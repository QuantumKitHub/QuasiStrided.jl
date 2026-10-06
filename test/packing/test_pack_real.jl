# pack! for real kernels against direct storage indexing.

include("helpers.jl")

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
