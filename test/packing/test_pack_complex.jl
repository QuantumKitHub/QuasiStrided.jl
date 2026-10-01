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

@testset "complex pack! ($T, $(typeof(fa)), $(typeof(fb))) vs reference" for
    T in (ComplexF64, ComplexF32), (fa, fb) in COMPLEX_FORMATS

    MR, NR = 4, 3
    kernel = Descriptor(Val(MR), Val(NR), T, fa, fb)
    R = real(T)
    storage = complex_storage(T, 4000)
    calls = Ref(0)
    counting = z -> (calls[] += 1; 2z + one(T))   # nonzero at zero
    for (i, PD, fmt) in ((1, MR, fa), (2, NR, fb)), k_block_length in (1, 5)
        fixtures = (
            (AffineAxis(0, 1, PD), AffineAxis(0, 97, k_block_length)),
            (AffineAxis(40, -1, PD), AffineAxis(3000, -97, k_block_length)),
            (view(SCATTER_LANES, 1:PD), view(SCATTER_STEPS, 1:k_block_length)),
        )
        len = _ref_rpe(fmt) * PD * k_block_length
        for (lane, step) in fixtures, valid in (PD, 1, 0)
            src, g = pack_fixture(storage, 500, resized(lane, valid), step)
            for f in (identity, conj)
                got, _ = pack_into(:vector, R, len, src, sliver_spec(kernel, i), f)
                @test all(isequal.(got, ref_pack(fmt, T, PD, k_block_length, valid, g, f)))
            end
            calls[] = 0
            got, canaries = pack_into(:panel, R, len, src, sliver_spec(kernel, i), counting)
            @test all(isequal.(got, ref_pack(fmt, T, PD, k_block_length, valid, g, z -> 2z + one(T))))
            @test calls[] == valid * k_block_length   # once per element, not per real half
            @test canaries
        end
    end
end

@testset "complex packing: validation before any write, k_block_length == 0 is a no-op" begin
    kernel = Descriptor(Val(4), Val(3), ComplexF64, PlanarFormat(), PlanarFormat())
    storage = fill(ComplexF64(3, 4), 100)
    src = Tile(storage, 0, AffineAxis(0, 1, 3), AffineAxis(0, 8, 2))
    a, b = sliver_spec(kernel, 1), sliver_spec(kernel, 2)
    # The buffer holds realtype(kernel), not scalartype(kernel).
    @test_throws ArgumentError pack!(zeros(ComplexF64, 1000), src, a, identity)
    @test_throws ArgumentError pack!(zeros(ComplexF64, 1000), src, b, identity)
    @test_throws ArgumentError pack!(zeros(Float32, 1000), src, a, identity)
    canary = fill(-42.0, 1000)
    @test_throws ArgumentError pack!(canary, Tile(storage, 0, AffineAxis(0, 1, 5), src.cols), a, identity)
    @test_throws ArgumentError pack!(canary, Tile(storage, 0, AffineAxis(0, 1, 4), src.cols), b, identity)
    # A buffer sized as if it held complex elements is half as long as needed.
    @test_throws DimensionMismatch pack!(zeros(Float64, 4 * 2), src, a, identity)
    @test_throws BoundsError pack!(canary, Tile(storage, 90, src.rows, src.cols), a, identity)
    @test all(==(-42.0), canary)

    read = Ref(false)
    spy = z -> (read[] = true; z)
    packed = fill(-42.0, 8)
    for spec in (a, b)
        @test pack!(packed, Tile(storage, 0, AffineAxis(0, 1, 3), AffineAxis(0, 8, 0)), spec, spy) === packed
    end
    @test all(==(-42.0), packed)
    @test !read[]
end

@testset "complex packing: zero steady-state allocation ($T, $(typeof(fa)))" for
    T in (ComplexF64, ComplexF32), (fa, fb) in COMPLEX_FORMATS[1:2]

    function run(::Type{T}, fa, fb) where {T}
        MR, NR, k_block_length = 8, 6, 4
        kernel = Descriptor(Val(MR), Val(NR), T, fa, fb)
        storage = rand(T, 4000)
        steps, lanes = [0, 100, 250, 400], collect(0:(max(MR, NR) - 1)) .* 3
        fast = Int[]
        slow = Int[]
        for (i, L) in ((1, MR), (2, NR))
            spec = sliver_spec(kernel, i)
            buf = zeros(real(T), packed_length(spec, k_block_length))
            full = Tile(storage, 0, AffineAxis(0, 1, L), AffineAxis(0, L, k_block_length))
            srcs = (
                full, Tile(storage, 0, view(lanes, 1:L), view(steps, 1:k_block_length)),
                Tile(storage, 0, AffineAxis(0, 1, L - 1), AffineAxis(0, L, k_block_length)),
                Tile(storage, 0, AffineAxis(0, 1, L), AffineAxis(0, L, 0)),
            )
            GC.@preserve buf for f in (identity, conj)   # the contiguous fast path, where the ISA has it
                push!(fast, steady_pack_allocs(packed_panel(buf, 1, length(buf)), full, spec, f))
            end
            for s in srcs, f in (identity, conj)
                push!(slow, steady_pack_allocs(buf, s, spec, f))
            end
            push!(slow, steady_pack_allocs(buf, full, spec, z -> 2z + oneunit(z)))
        end
        return fast, slow
    end
    fast, slow = run(T, fa, fb)
    @test all(iszero, fast)
    # Julia 1.10 does not keep the scalar complex path allocation-free.
    @test all(iszero, slow) skip = VERSION < v"1.11"
end
