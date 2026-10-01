# pack! for real kernels against direct storage indexing. The reference
# layouts and fixtures here are shared with the two complex packing files
# included after this one.

using QuasiStrided: RealFormat, PlanarFormat, OneEFormat, InterleavedFormat, PackedPanel,
    packed_panel, packed_length, copies_unchanged, real_contiguous_eligible

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

# A sliver source with lanes along `lane` and K steps along `step` (B's tile
# transposed), and its direct-indexing reader.
function pack_fixture(storage, base, lane, step)
    return Tile(storage, base, lane, step), (t, p) -> storage[base + lane[t + 1] + step[p + 1] + 1]
end

# Packs `src` into a destination of kind `dst` (canaried unless a bare Vector)
# and returns the packed prefix and whether the canaries survived.
function pack_into(dst, R, len, src, spec, f)
    buf = fill(R(-777), len + 12)
    if dst === :vector
        v = buf[1:len]
        @test pack!(v, src, spec, f) === v
        return v, true
    end
    GC.@preserve buf begin
        d = packed_panel(buf, 5, len)
        @test pack!(d, src, spec, f) === d
    end
    return buf[5:(4 + len)], all(==(R(-777)), buf[1:4]) && all(==(R(-777)), buf[(5 + len):end])
end

const SCATTER_LANES = [5, 30, 1, 17, 9, 44, 2, 23, 11, 38, 7, 60, 14, 51, 3, 29]
const SCATTER_STEPS = [7, 900, 300, 1500, 60, 1210, 420]

@testset "pack! ($T): every stride kind and tail width vs direct indexing" for T in (Float64, Float32)
    MR, NR, k_block_length = 8, 6, 5
    kernel = Descriptor(Val(MR), Val(NR), T)
    storage = T.(collect(1.0:2000.0))
    calls = Ref(0)
    counting = x -> (calls[] += 1; 3x + 1000)   # nonzero at zero: padding must bypass it
    lanes = Any[
        AffineAxis(0, 1, 8), AffineAxis(3, 7, 8), AffineAxis(40, -1, 8), AffineAxis(20, 0, 8),
        view(SCATTER_LANES, 1:8),
    ]
    steps = Any[AffineAxis(0, 64, k_block_length), AffineAxis(700, -97, k_block_length), view(SCATTER_STEPS, 1:k_block_length)]
    for (spec, L) in ((sliver_spec(kernel, 1), MR), (sliver_spec(kernel, 2), NR)),
            lane in lanes, step in steps, valid in (L, 1, 0), dst in (:vector, :panel)
        src, g = pack_fixture(storage, 11, resized(lane, valid), step)
        calls[] = 0
        got, canaries = pack_into(dst, T, L * k_block_length, src, spec, counting)
        @test got == ref_pack(RealFormat(), T, L, k_block_length, valid, g, x -> 3x + 1000)
        @test calls[] == valid * k_block_length
        @test canaries
    end
    # Every tail width, at the minimal and a longer K depth.
    negate = x -> -x
    for (spec, L) in ((sliver_spec(kernel, 1), MR), (sliver_spec(kernel, 2), NR)), valid in 0:L, kc1 in (1, 5)
        src, g = pack_fixture(storage, 0, AffineAxis(2, 3, valid), AffineAxis(0, 40, kc1))
        got, _ = pack_into(:vector, T, L * kc1, src, spec, negate)
        @test got == ref_pack(RealFormat(), T, L, kc1, valid, g, negate)
    end
end

@testset "pack!: B packs as A packs its transposed tile" begin
    kernel = Descriptor(Val(6), Val(6), Float64)
    storage = collect(1.0:2000.0)
    tile_b = Tile(storage, 3, view(SCATTER_STEPS, 1:5), AffineAxis(0, 1, 6))   # K x N
    got_b, _ = pack_into(:panel, Float64, 30, transpose(tile_b), sliver_spec(kernel, 2), identity)
    got_a, _ = pack_into(:panel, Float64, 30, transpose(tile_b), sliver_spec(kernel, 1), identity)
    @test got_b == got_a
end

@testset "pack! contiguous fast path: fires exactly when eligible ($T, L=$L, side $i)" for
    T in (Float64, Float32), (L, i) in ((4, 1), (16, 1), (6, 2))

    kernel = Descriptor(Val(L), Val(L), T)
    vals = T.(collect(1.0:2000.0))
    mixed = (T === Float64 ? Float32 : Float64).(collect(1.0:2000.0) ./ 3)
    storages = @static isdefined(Base, :Memory) ? (vals, copyto!(Memory{T}(undef, 2000), vals), mixed) : (vals, mixed)
    koffs = [7, 900, 300, 1500]
    contig = collect(0:(L - 1))
    spec = sliver_spec(kernel, i)
    for storage in storages
        steps = Any[AffineAxis(0, L, 5), AffineAxis(1800, -L, 6), view(koffs, 1:4)]
        lanes = Any[
            (AffineAxis(0, 1, L), true), (AffineAxis(5, 1, L), true),
            (AffineAxis(0, 2, L), false), (AffineAxis(L + 3, -1, L), false),
            (AffineAxis(0, 1, L - 1), false), (view(contig, 1:L), false),
        ]
        for (lane, eligible) in lanes, step in steps, f in (identity, conj, x -> -x)
            src, g = pack_fixture(storage, 11, lane, step)
            k_block_length = length(step)
            @test real_contiguous_eligible(src, spec, f, length(lane)) ==
                (eligible && (f === identity || f === conj))
            got, canaries = pack_into(:panel, T, L * k_block_length, src, spec, f)
            @test got == ref_pack(RealFormat(), T, L, k_block_length, length(lane), g, f)
            @test canaries
        end
    end
    @test copies_unchanged(conj, T)
    @test !copies_unchanged(conj, complex(T))   # conj is not the identity on complex
end

@testset "pack!: k_block_length == 0 reads and writes nothing" begin
    kernel = Descriptor(Val(4), Val(3), Float64)
    storage = fill(3.0, 10)
    read = Ref(false)
    spy = x -> (read[] = true; x)
    packed = fill(-42.0, 8)
    for i in 1:2
        @test pack!(packed, Tile(storage, 0, AffineAxis(0, 1, 3), AffineAxis(0, 1, 0)), sliver_spec(kernel, i), spy) === packed
    end
    @test packed == fill(-42.0, 8)
    @test !read[]
end

@testset "pack!: invalid metadata rejected before mutation" begin
    kernel = Descriptor(Val(4), Val(3), Float64)
    kernel32 = Descriptor(Val(4), Val(3), Float32)
    storage = fill(9.0, 20)
    tile(m, n, base = 0) = Tile(storage, base, AffineAxis(0, 1, m), AffineAxis(0, 1, n))
    for (i, bad, ok, short) in ((1, tile(5, 3), tile(4, 3), 11), (2, tile(4, 3), tile(3, 3), 8))
        spec = sliver_spec(kernel, i)
        canary = fill(-1.0, 100)
        @test_throws ArgumentError pack!(canary, bad, spec, identity)                      # lanes > L
        @test_throws ArgumentError pack!(canary, ok, sliver_spec(kernel32, i), identity)   # eltype mismatch
        @test_throws DimensionMismatch pack!(fill(-1.0, short), ok, spec, identity)
        @test_throws BoundsError pack!(canary, tile(size(ok)..., 18), spec, identity)
        @test canary == fill(-1.0, 100)
    end
end

@testset "pack!: zero steady-state allocation" begin
    nontrivial(x) = 2x + 1
    # Every destination kind, affine/scattered axes, tails and
    # k_block_length == 0; the ScalarKernel/SIMDKernel forwarding methods on a subset.
    function run(ctor, ::Type{T}, MR, NR, full) where {T}
        kernel = ctor(Val(MR), Val(NR), T)
        k_block_length = 7
        storage = rand(T, 4000)
        koffs = [0, MR, 3MR, 2MR, 5MR, 4MR, 6MR]
        lanes = collect(0:(max(MR, NR) - 1)) .* 3
        bytes = Int[]
        for (i, L) in ((1, MR), (2, NR))
            spec = sliver_spec(kernel, i)
            buf = zeros(T, L * k_block_length + 8)
            pk = view(koffs, 1:k_block_length)
            srcs = (
                Tile(storage, 0, AffineAxis(0, 1, L), AffineAxis(0, L, k_block_length)),
                Tile(storage, 0, AffineAxis(0, 1, L), pk),
                Tile(storage, 0, AffineAxis(0, 2, L - 1), AffineAxis(0, 2L, k_block_length)),
                Tile(storage, 0, view(lanes, 1:L), pk),
                Tile(storage, 0, AffineAxis(0, 1, L), AffineAxis(0, L, 0)),
            )
            GC.@preserve buf for d in (buf, packed_panel(buf, 1, L * k_block_length))
                for s in (full ? srcs : srcs[1:1]), f in (identity, conj)
                    push!(bytes, steady_pack_allocs(d, s, spec, f))
                end
                push!(bytes, steady_pack_allocs(d, srcs[1], spec, nontrivial))
            end
        end
        return bytes
    end
    @test all(iszero, run(Descriptor, Float64, 16, 6, true))
    @test all(iszero, run(Descriptor, Float32, 32, 6, true))
    @test all(iszero, run(ScalarKernel, Float64, 8, 6, false))
    @test all(iszero, run(SIMDKernel, Float64, 8, 6, false))
end
