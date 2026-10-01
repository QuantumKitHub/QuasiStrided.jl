# pack_a!/pack_b! for real kernels against direct storage indexing. The
# reference layouts and fixtures here are shared with the two complex packing
# files included after this one.

using QuasiStrided: RealFormat, PlanarFormat, OneEFormat, InterleavedFormat, PackedPanel,
    packed_panel, ComplexKernelDescriptor, _copies_unchanged,
    _pack_a_contiguous_eligible

# Packed layouts written from the format definitions, not from the offset
# helpers. `g(t, p)` is the source element at lane `t`, K step `p`; lanes
# `>= valid` are padding and stay literal `+0.0`.
_ref_emit!(o, ::RealFormat, vr, p, t, z) = (o[vr * p + t + 1] = z)
function _ref_emit!(o, ::PlanarFormat, vr, p, t, z)
    o[2vr * p + t + 1] = real(z)
    return o[2vr * p + vr + t + 1] = imag(z)
end
function _ref_emit!(o, ::InterleavedFormat, vr, p, t, z)
    o[2vr * p + 2t + 1] = real(z)
    return o[2vr * p + 2t + 2] = imag(z)
end
function _ref_emit!(o, ::OneEFormat, vr, p, t, z)   # [[re, -im], [im, re]]
    b = 4vr * p
    o[b + 2t + 1], o[b + 2t + 2] = real(z), imag(z)
    return o[b + 2vr + 2t + 1], o[b + 2vr + 2t + 2] = -imag(z), real(z)
end
_ref_rpe(::RealFormat) = 1
_ref_rpe(::Union{PlanarFormat, InterleavedFormat}) = 2
_ref_rpe(::OneEFormat) = 4
function ref_pack(fmt, ::Type{T}, vr, k_block_length, valid, g, f) where {T}
    out = zeros(real(T), _ref_rpe(fmt) * vr * k_block_length)
    for p in 0:(k_block_length - 1), t in 0:(valid - 1)
        _ref_emit!(out, fmt, vr, p, t, convert(T, f(g(t, p))))
    end
    return out
end

resized(ax::AffineAxis, n) = AffineAxis(ax.base, ax.stride, n)
resized(ax::SubArray, n) = view(parent(ax), 1:n)

# An A (`lane` = rows) or B (`lane` = cols) source over `storage`, and its
# direct-indexing reader.
function pack_fixture(side, storage, base, lane, step)
    src = side === :a ? Tile(storage, base, lane, step) : Tile(storage, base, step, lane)
    return src, (t, p) -> storage[base + lane[t + 1] + step[p + 1] + 1]
end

# Packs `src` into a destination of kind `dst` (canaried unless a bare Vector)
# and returns the packed prefix and whether the canaries survived.
function pack_into(pack!, dst, R, len, src, kernel, f)
    buf = fill(R(-777), len + 12)
    if dst === :vector
        v = buf[1:len]
        @test pack!(v, src, kernel, f) === v
        return v, true
    end
    GC.@preserve buf begin
        d = dst === :view ? view(buf, 5:(4 + len)) : packed_panel(buf, 5, len)
        @test pack!(d, src, kernel, f) === d
    end
    return buf[5:(4 + len)], all(==(R(-777)), buf[1:4]) && all(==(R(-777)), buf[(5 + len):end])
end

const SCATTER_LANES = [5, 30, 1, 17, 9, 44, 2, 23, 11, 38, 7, 60, 14, 51, 3, 29]
const SCATTER_STEPS = [7, 900, 300, 1500, 60, 1210, 420]

@testset "pack_a!/pack_b! ($T): every stride kind and tail width vs direct indexing" for
    T in (Float64, Float32)

    MR, NR, k_block_length = 8, 6, 5
    kernel = KernelDescriptor(Val(MR), Val(NR), T)
    storage = T.(collect(1.0:2000.0))
    calls = Ref(0)
    counting = x -> (calls[] += 1; 3x + 1000)   # nonzero at zero: padding must bypass it
    lanes = Any[
        AffineAxis(0, 1, 8), AffineAxis(3, 7, 8), AffineAxis(40, -1, 8), AffineAxis(20, 0, 8),
        view(SCATTER_LANES, 1:8),
    ]
    steps = Any[AffineAxis(0, 64, k_block_length), AffineAxis(700, -97, k_block_length), view(SCATTER_STEPS, 1:k_block_length)]
    for (side, pack!, PD) in ((:a, pack_a!, MR), (:b, pack_b!, NR)),
            lane in lanes, step in steps, valid in (PD, 1, 0), dst in (:vector, :view)
        src, g = pack_fixture(side, storage, 11, resized(lane, valid), step)
        calls[] = 0
        got, canaries = pack_into(pack!, dst, T, PD * k_block_length, src, kernel, counting)
        @test got == ref_pack(RealFormat(), T, PD, k_block_length, valid, g, x -> 3x + 1000)
        @test calls[] == valid * k_block_length
        @test canaries
    end
    # Every tail width, at the minimal and a longer K depth.
    negate = x -> -x
    for (side, pack!, PD) in ((:a, pack_a!, MR), (:b, pack_b!, NR)), valid in 0:PD, kc1 in (1, 5)
        src, g = pack_fixture(side, storage, 0, AffineAxis(2, 3, valid), AffineAxis(0, 40, kc1))
        got, _ = pack_into(pack!, :vector, T, PD * kc1, src, kernel, negate)
        @test got == ref_pack(RealFormat(), T, PD, kc1, valid, g, negate)
    end
end

@testset "pack_a! contiguous fast path: fires exactly when eligible ($T, MR=$MR)" for
    T in (Float64, Float32), MR in (4, 16)

    kernel = KernelDescriptor(Val(MR), Val(3), T)
    vals = T.(collect(1.0:2000.0))
    mixed = (T === Float64 ? Float32 : Float64).(collect(1.0:2000.0) ./ 3)
    storages = @static isdefined(Base, :Memory) ? (vals, copyto!(Memory{T}(undef, 2000), vals), mixed) : (vals, mixed)
    koffs = [7, 900, 300, 1500]
    contig = collect(0:(MR - 1))
    for storage in storages
        steps = Any[AffineAxis(0, MR, 5), AffineAxis(1800, -MR, 6), view(koffs, 1:4)]
        lanes = Any[
            (AffineAxis(0, 1, MR), true), (AffineAxis(5, 1, MR), true),
            (AffineAxis(0, 2, MR), false), (AffineAxis(MR + 3, -1, MR), false),
            (AffineAxis(0, 1, MR - 1), false), (view(contig, 1:MR), false),
        ]
        for (lane, eligible) in lanes, step in steps, f in (identity, conj, x -> -x)
            src, g = pack_fixture(:a, storage, 11, lane, step)
            k_block_length = length(step)
            pp = packed_panel(zeros(T, 1), 1, 1)
            @test _pack_a_contiguous_eligible(pp, src, f, length(lane), Val(MR), T) ==
                (eligible && (f === identity || f === conj))
            @test !_pack_a_contiguous_eligible(zeros(T, 1), src, f, length(lane), Val(MR), T)
            got, canaries = pack_into(pack_a!, :panel, T, MR * k_block_length, src, kernel, f)
            @test got == ref_pack(RealFormat(), T, MR, k_block_length, length(lane), g, f)
            @test canaries
        end
    end
    @test _copies_unchanged(conj, T)
    @test !_copies_unchanged(conj, complex(T))   # conj is not the identity on complex
end

@testset "pack_a!/pack_b!: k_block_length == 0 reads and writes nothing" begin
    kernel = KernelDescriptor(Val(4), Val(3), Float64)
    storage = fill(3.0, 10)
    read = Ref(false)
    spy = x -> (read[] = true; x)
    packed = fill(-42.0, 8)
    @test pack_a!(packed, Tile(storage, 0, AffineAxis(0, 1, 3), AffineAxis(0, 1, 0)), kernel, spy) === packed
    @test pack_b!(packed, Tile(storage, 0, AffineAxis(0, 1, 0), AffineAxis(0, 1, 3)), kernel, spy) === packed
    @test packed == fill(-42.0, 8)
    @test !read[]
end

@testset "pack_a!/pack_b!: invalid metadata rejected before mutation" begin
    kernel = KernelDescriptor(Val(4), Val(3), Float64)
    kernel32 = KernelDescriptor(Val(4), Val(3), Float32)
    storage = fill(9.0, 20)
    tile(m, n, base = 0) = Tile(storage, base, AffineAxis(0, 1, m), AffineAxis(0, 1, n))
    for (pack!, bad, ok, short) in ((pack_a!, tile(5, 3), tile(4, 3), 11), (pack_b!, tile(3, 4), tile(3, 3), 8))
        canary = fill(-1.0, 100)
        @test_throws ArgumentError pack!(canary, bad, kernel, identity)     # m > MR / n > NR
        @test_throws ArgumentError pack!(canary, ok, kernel32, identity)    # eltype mismatch
        @test_throws DimensionMismatch pack!(fill(-1.0, short), ok, kernel, identity)
        @test_throws BoundsError pack!(canary, tile(size(ok)..., 18), kernel, identity)
        @test canary == fill(-1.0, 100)
    end
end

@testset "pack_a!/pack_b!: zero steady-state allocation" begin
    nontrivial(x) = 2x + 1
    # Every destination kind, affine/scattered axes, tails and
    # k_block_length == 0; the ScalarKernel/SIMDKernel forwarding methods on a subset.
    function run(ctor, ::Type{T}, MR, NR, full) where {T}
        kernel = ctor(Val(MR), Val(NR), T)
        k_block_length = 7
        storage = rand(T, 4000)
        koffs = [0, MR, 3MR, 2MR, 5MR, 4MR, 6MR]
        lanes = collect(0:(max(MR, NR) - 1)) .* 3
        bufa = zeros(T, MR * k_block_length + 8)
        bufb = zeros(T, NR * k_block_length + 8)
        bytes = Int[]
        GC.@preserve bufa bufb begin
            pa = packed_panel(bufa, 1, MR * k_block_length)
            dsts_a = (bufa, view(bufa, 3:(2 + MR * k_block_length)), pa)
            dsts_b = (bufb, view(bufb, 3:(2 + NR * k_block_length)), packed_panel(bufb, 1, NR * k_block_length))
            pk = view(koffs, 1:k_block_length)
            srcs_a = (
                Tile(storage, 0, AffineAxis(0, 1, MR), AffineAxis(0, MR, k_block_length)),
                Tile(storage, 0, AffineAxis(0, 1, MR), pk),
                Tile(storage, 0, AffineAxis(0, 2, MR - 1), AffineAxis(0, 2MR, k_block_length)),
                Tile(storage, 0, view(lanes, 1:MR), view(koffs, 1:k_block_length)),
                Tile(storage, 0, AffineAxis(0, 1, MR), AffineAxis(0, MR, 0)),
            )
            srcs_b = (
                Tile(storage, 0, AffineAxis(0, 1, k_block_length), AffineAxis(0, k_block_length, NR)),
                Tile(storage, 0, pk, AffineAxis(0, k_block_length, NR - 1)),
                Tile(storage, 0, view(koffs, 1:k_block_length), view(lanes, 1:NR)),
                Tile(storage, 0, AffineAxis(0, 1, 0), AffineAxis(0, k_block_length, NR)),
            )
            if !full
                srcs_a, srcs_b, dsts_a, dsts_b = srcs_a[1:1], srcs_b[1:1], dsts_a[1:2], dsts_b[1:2]
            end
            for d in dsts_a
                for s in srcs_a
                    push!(bytes, steady_pack_allocs(pack_a!, d, s, kernel, identity))
                end
                push!(bytes, steady_pack_allocs(pack_a!, d, srcs_a[1], kernel, nontrivial))
            end
            for d in dsts_b, s in srcs_b
                push!(bytes, steady_pack_allocs(pack_b!, d, s, kernel, identity))
            end
            push!(bytes, steady_pack_allocs(pack_b!, dsts_b[1], srcs_b[1], kernel, nontrivial))
            push!(bytes, steady_pack_allocs(pack_a!, pa, srcs_a[1], kernel, conj))
        end
        return bytes
    end
    @test all(iszero, run(KernelDescriptor, Float64, 16, 6, true))
    @test all(iszero, run(KernelDescriptor, Float32, 32, 6, true))
    @test all(iszero, run(ScalarKernel, Float64, 8, 6, false))
    @test all(iszero, run(SIMDKernel, Float64, 8, 6, false))
end
