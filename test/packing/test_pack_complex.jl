# Complex packing against test_pack_real.jl's reference layouts. `transform`
# must apply to the complex element before it is split (per real half, `conj`
# would be a silent no-op). Packing is exact, so comparisons are bitwise
# `isequal`, which also separates `+0.0` from `-0.0`.

# Exactly representable in Float32, with signed and unsigned zeros sprinkled in:
# `conj` and 1e's `-im` are sign flips.
function complex_value(::Type{T}, i) where {T}
    R = real(T)
    i % 17 == 0 && return T(R(0), R(0))
    i % 19 == 0 && return T(R(3), R(0))
    i % 23 == 0 && return T(R(0), R(-5))
    return T(R(10i + 1), R(-(10i + 2)))
end
complex_storage(::Type{T}, n) where {T} = [complex_value(T, i) for i in 1:n]

const COMPLEX_FORMATS = (
    (PlanarFormat(), PlanarFormat()), (OneEFormat(), PlanarFormat()),
    (InterleavedFormat(), PlanarFormat()), (OneEFormat(), OneEFormat()),
)

@testset "complex pack_a!/pack_b! ($T, $(typeof(fa)), $(typeof(fb))) vs reference" for
    T in (ComplexF64, ComplexF32), (fa, fb) in COMPLEX_FORMATS

    MR, NR = 4, 3
    kernel = ComplexKernelDescriptor(Val(MR), Val(NR), T, fa, fb)
    R = real(T)
    storage = complex_storage(T, 4000)
    calls = Ref(0)
    counting = z -> (calls[] += 1; 2z + one(T))   # nonzero at zero
    for (side, pack!, PD, fmt) in ((:a, pack_a!, MR, fa), (:b, pack_b!, NR, fb)), kc in (1, 5)
        fixtures = (
            (AffineAxis(0, 1, PD), AffineAxis(0, 97, kc)),
            (AffineAxis(40, -1, PD), AffineAxis(3000, -97, kc)),
            (ScatterAxis(SCATTER_LANES, PD), ScatterAxis(SCATTER_STEPS, kc)),
        )
        len = _ref_rpe(fmt) * PD * kc
        for (lane, step) in fixtures, valid in (PD, 1, 0)
            src, g = pack_fixture(side, storage, 500, resized(lane, valid), step)
            for f in (identity, conj)
                got, _ = pack_into(pack!, :vector, R, len, src, kernel, f)
                @test all(isequal.(got, ref_pack(fmt, T, PD, kc, valid, g, f)))
            end
            calls[] = 0
            got, canaries = pack_into(pack!, :view, R, len, src, kernel, counting)
            @test all(isequal.(got, ref_pack(fmt, T, PD, kc, valid, g, z -> 2z + one(T))))
            @test calls[] == valid * kc   # once per element, not per real half
            @test canaries
        end
    end
end

@testset "complex packing: validation before any write, kc == 0 is a no-op" begin
    kernel = ComplexKernelDescriptor(Val(4), Val(3), ComplexF64, PlanarFormat(), PlanarFormat())
    storage = fill(ComplexF64(3, 4), 100)
    src = SourceTile(storage, 0, AffineAxis(0, 1, 4), AffineAxis(0, 8, 2))
    # The buffer holds realtype(kernel), not scalartype(kernel).
    @test_throws ArgumentError pack_a!(zeros(ComplexF64, 1000), src, kernel, identity)
    @test_throws ArgumentError pack_b!(zeros(ComplexF64, 1000), src, kernel, identity)
    @test_throws ArgumentError pack_a!(zeros(Float32, 1000), src, kernel, identity)
    canary = fill(-42.0, 1000)
    @test_throws ArgumentError pack_a!(canary, SourceTile(storage, 0, AffineAxis(0, 1, 5), src.cols), kernel, identity)
    @test_throws ArgumentError pack_b!(canary, SourceTile(storage, 0, src.cols, AffineAxis(0, 1, 4)), kernel, identity)
    # A buffer sized as if it held complex elements is half as long as needed.
    @test_throws DimensionMismatch pack_a!(zeros(Float64, 4 * 2), src, kernel, identity)
    @test_throws BoundsError pack_a!(canary, SourceTile(storage, 90, src.rows, src.cols), kernel, identity)
    @test all(==(-42.0), canary)

    read = Ref(false)
    spy = z -> (read[] = true; z)
    packed = fill(-42.0, 8)
    @test pack_a!(packed, SourceTile(storage, 0, AffineAxis(0, 1, 3), AffineAxis(0, 8, 0)), kernel, spy) === packed
    @test pack_b!(packed, SourceTile(storage, 0, AffineAxis(0, 1, 0), AffineAxis(0, 8, 3)), kernel, spy) === packed
    @test all(==(-42.0), packed)
    @test !read[]
end

@testset "complex packing: zero steady-state allocation ($T, $(typeof(fa)))" for
    T in (ComplexF64, ComplexF32), (fa, fb) in COMPLEX_FORMATS[1:2]

    function run(::Type{T}, fa, fb) where {T}
        MR, NR, kc = 8, 6, 4
        kernel = ComplexKernelDescriptor(Val(MR), Val(NR), T, fa, fb)
        R = real(T)
        storage = rand(T, 4000)
        bufa = zeros(R, packed_a_length(kernel, kc))
        bufb = zeros(R, packed_b_length(kernel, kc))
        ro, co, ro_b = collect(0:(MR - 1)) .* 3, [0, 100, 250, 400], collect(0:(NR - 1)) .* 5
        fast = Int[]
        slow = Int[]
        GC.@preserve bufa bufb begin
            pa = packed_panel(bufa, 1, length(bufa))
            pb = packed_panel(bufb, 1, length(bufb))
            full_a = SourceTile(storage, 0, AffineAxis(0, 1, MR), AffineAxis(0, MR, kc))
            full_b = SourceTile(storage, 0, AffineAxis(0, 1, kc), AffineAxis(0, kc, NR))
            for f in (identity, conj)   # the contiguous fast path, where the ISA has it
                push!(fast, steady_pack_allocs(pack_a!, pa, full_a, kernel, f))
                push!(fast, steady_pack_allocs(pack_b!, pb, full_b, kernel, f))
            end
            srcs_a = (
                full_a, SourceTile(storage, 0, ScatterAxis(ro, MR), ScatterAxis(co, kc)),
                SourceTile(storage, 0, AffineAxis(0, 1, MR - 1), AffineAxis(0, MR, kc)),
                SourceTile(storage, 0, AffineAxis(0, 1, MR), AffineAxis(0, MR, 0)),
            )
            srcs_b = (
                full_b, SourceTile(storage, 0, ScatterAxis(co, kc), ScatterAxis(ro_b, NR)),
                SourceTile(storage, 0, AffineAxis(0, 1, kc), AffineAxis(0, kc, NR - 1)),
            )
            for s in srcs_a, f in (identity, conj)
                push!(slow, steady_pack_allocs(pack_a!, bufa, s, kernel, f))
            end
            for s in srcs_b, f in (identity, conj)
                push!(slow, steady_pack_allocs(pack_b!, bufb, s, kernel, f))
            end
            push!(slow, steady_pack_allocs(pack_a!, bufa, full_a, kernel, z -> 2z + oneunit(z)))
        end
        return fast, slow
    end
    fast, slow = run(T, fa, fb)
    @test all(iszero, fast)
    # Julia 1.10 does not keep the scalar complex path allocation-free.
    @test all(iszero, slow) skip = VERSION < v"1.11"
end
