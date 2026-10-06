# The stages driven by hand, end to end: AxisGroup -> BlockDescriptor -> tile
# axes -> pack! -> ScalarKernel execute_tile!, with beta applied on
# the first K panel only.

include("../helpers.jl")

@testset "manual pipeline: AxisGroup -> tiles -> packing -> ScalarKernel" begin
    A, B, Cref = _worked_fixture()
    M = AxisGroup((3, 2), ((1, 15), (1, 12)))  # A, C
    N = AxisGroup((4,), ((5,), (3,)))          # B, C
    K = AxisGroup((5,), ((3,), (1,)))          # A, B
    @test (axis_length(M), axis_length(N), axis_length(K)) == (6, 4, 5)

    kernel = ScalarKernel(Val(8), Val(6), Float64)  # one tile covers M and N
    bufs(g) = (zeros(Int, axis_length(g)), zeros(Int, axis_length(g)))
    mb, nb, kb = bufs(M), bufs(N), bufs(K)
    (dM_A, dM_C) = block_descriptors!(mb, M, 0, 6)
    (dN_B, dN_C) = block_descriptors!(nb, N, 0, 4)
    # Scatter axes borrow pointers into the offset buffers.
    GC.@preserve mb nb kb begin
        axis(d, buf) = QuasiStrided.axis_of(d, buf, 0)
        row_A, row_C = axis(dM_A, mb[1]), axis(dM_C, mb[2])
        col_B, col_C = axis(dN_B, nb[1]), axis(dN_C, nb[2])

        Cstorage = zeros(length(Cref))
        destination = Tile(Cstorage, 0, row_C, col_C)
        @test size(destination) == (6, 4)
        packed_a, packed_b = zeros(tile_size(kernel, 1) * 5), zeros(tile_size(kernel, 2) * 5)

        Cstart = rand(MersenneTwister(1234), size(Cref)...)
        # One panel covering K, then panels of 2, 2, 1 with nontrivial alpha/beta.
        for (panels, alpha, beta) in ((((0, 5),), 1.0, 0.0), (((0, 2), (2, 2), (4, 1)), 2.5, 0.75))
            copyto!(Cstorage, vec(Cstart))
            for (idx, (first, len)) in enumerate(panels)
                (dK_A, dK_B) = block_descriptors!(kb, K, first, len)
                pack!(packed_a, Tile(vec(A), 0, row_A, axis(dK_A, kb[1])), sliver_spec(kernel, 1), identity)
                pack!(packed_b, Tile(vec(B), 0, col_B, axis(dK_B, kb[2])), sliver_spec(kernel, 2), identity)
                execute_tile!(kernel, destination, packed_a, packed_b, len, alpha, idx == 1 ? beta : 1.0)
            end
            @test reshape(Cstorage, size(Cref)) ≈ alpha .* Cref .+ beta .* Cstart
        end
    end
end

@testset "tile-level bounds checks and alpha/beta shortcuts" begin
    k = ScalarKernel(Val(2), Val(2), Float64)
    # Out-of-bounds source or destination storage is rejected before any access.
    packed = zeros(4)
    for base in (10_000, -10_000)
        src = Tile([1.0, 2.0, 3.0, 4.0], base, AffineAxis(0, 1, 2), AffineAxis(0, 2, 2))
        @test_throws BoundsError pack!(packed, src, sliver_spec(k, 1), identity)
        @test_throws BoundsError pack!(packed, src, sliver_spec(k, 2), identity)
    end
    @test all(iszero, packed)
    canary = fill(999.0, 4)
    dst = Tile(canary, 3, AffineAxis(0, 1, 2), AffineAxis(0, 0, 1))
    @test_throws BoundsError execute_tile!(k, dst, zeros(4), zeros(4), 0, 1.0, 2.0)
    @test all(==(999.0), canary)

    # Undersized packed buffers are rejected, except by the k_block_length = 0 / alpha = 0
    # short-circuits, which never read them.
    dst = Tile(zeros(4), 0, AffineAxis(0, 1, 2), AffineAxis(0, 2, 2))
    @test_throws DimensionMismatch execute_tile!(k, dst, zeros(4), zeros(4), 50, 1.0, 0.0)
    @test execute_tile!(k, dst, zeros(1), zeros(1), 0, 1.0, 0.0) === dst
    @test execute_tile!(k, dst, zeros(1), zeros(1), 5, 0.0, 1.0) === dst

    @test checked_tile_storage_bounds(0, AffineAxis(0, 1, 3), AffineAxis(0, 3, 2), 6) === nothing
    @test checked_tile_storage_bounds(1_000_000, AffineAxis(0, 1, 0), AffineAxis(0, 1, 5), 1) === nothing
    @test_throws BoundsError checked_tile_storage_bounds(0, AffineAxis(0, 1, 3), AffineAxis(0, 3, 2), 5)
    @test_throws BoundsError checked_tile_storage_bounds(-1, AffineAxis(0, 1, 3), AffineAxis(0, 3, 2), 6)

    kernel = ScalarKernel(Val(3), Val(2), Float64)
    pa, pb = Float64[1, 2, 3, 4, 5, 6], Float64[10, 20, 30, 40]
    tile(s) = Tile(s, 0, AffineAxis(0, 1, 3), AffineAxis(0, 3, 2))
    s = fill(NaN, 6)
    execute_tile!(kernel, tile(s), pa, pb, 2, 0.0, 0.0)   # alpha = beta = 0: zeros
    @test all(iszero, s)
    s = [1.0, NaN, 3.0, 4.0, NaN, 6.0]
    execute_tile!(kernel, tile(s), pa, pb, 2, 0.0, 1.0)   # alpha = 0, beta = 1: no-op
    @test isequal(s, [1.0, NaN, 3.0, 4.0, NaN, 6.0])
    s = fill(NaN, 6)
    execute_tile!(kernel, tile(s), pa, pb, 2, 1.0, 0.0)   # beta = 0 never reads C
    @test all(isfinite, s)
    s = [42.0]
    execute_tile!(kernel, Tile(s, 0, AffineAxis(0, 1, 0), AffineAxis(0, 1, 0)), pa, pb, 2, 1.0, 0.0)
    @test s == [42.0]
    @test_throws ArgumentError execute_tile!(
        kernel, Tile(zeros(20), 0, AffineAxis(0, 1, 4), AffineAxis(0, 4, 2)), pa, pb, 2, 1.0, 0.0
    )
    # Canaries outside the tile survive; scattered columns agree with affine ones.
    s = fill(-1.0, 8)
    execute_tile!(kernel, tile(s), pa, pb, 2, 1.0, 0.0)
    @test s[7] == -1.0 && s[8] == -1.0
    s6 = zeros(6)
    execute_tile!(kernel, Tile(s6, 0, AffineAxis(0, 1, 3), view([0, 3], 1:2)), pa, pb, 2, 1.0, 0.0)
    @test s6 == s[1:6]
    s32 = zeros(Float32, 6)
    execute_tile!(ScalarKernel(Val(3), Val(2), Float32), tile(s32), Float32.(pa), Float32.(pb), 2, 1.0f0, 0.0f0)
    @test s32 ≈ s[1:6]
end
