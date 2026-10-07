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
