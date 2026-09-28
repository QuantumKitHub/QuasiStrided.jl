using QuasiStrided: SIMDKernel, lanewidth, _vector_store_eligible, kernel_shapes
using StridedViews: StridedView

@testset "SIMDKernel" begin
    full = ((Float64, (8, 6, 4)), (Float32, (16, 6, 8)))
    for (T, (MR, NR, W)) in full
        mk_contract(SIMDKernel(Val(MR), Val(NR), T, Val(W)))
    end
    # Every shipped shape, plus 28 accumulator vectors (past the point where a
    # dynamically indexed tuple is heap-allocated).
    for (T, s) in ((T, s) for T in (Float64, Float32) for s in kernel_shapes(T))
        (T, s) in full || mk_contract(SIMDKernel(Val(s[1]), Val(s[2]), T, Val(s[3])); full = false)
    end
    mk_contract(SIMDKernel(Val(16), Val(7), Float64, Val(4)); full = false)

    @test lanewidth(SIMDKernel(Val(8), Val(6), Float64)) == 4
    @test lanewidth(SIMDKernel(Val(16), Val(6), Float32)) == 8
    @test_throws ArgumentError SIMDKernel(Val(6), Val(4), Float64, Val(4))
    @test_throws ArgumentError SIMDKernel(Val(6), Val(4), Float64, Val(0))

    @testset "_vector_store_eligible: unit-stride rows into 1-D dense storage of T" begin
        for T in (Float64, Float32)
            m, n = 8, 8
            rows, cols = AffineAxis(0, 1, m), AffineAxis(0, m, n)
            mem = parent(StridedView(zeros(T, m * n)))  # `Memory{T}` on Julia >= 1.11
            @test _vector_store_eligible(DestinationTile(mem, 0, rows, cols), T)
            @test _vector_store_eligible(DestinationTile(zeros(T, m * n), 0, rows, cols), T)
            @test !_vector_store_eligible(DestinationTile(view(zeros(T, m * n + 4), 3:(m * n + 2)), 0, rows, cols), T)
            @test !_vector_store_eligible(DestinationTile(zeros(T, m, n), 0, rows, cols), T)
            @test !_vector_store_eligible(DestinationTile(zeros(T, 2m * n), 0, AffineAxis(0, 2, m), AffineAxis(0, 2m, n)), T)
            @test !_vector_store_eligible(DestinationTile(mem, 0, ScatterAxis(collect(0:(m - 1)), m), cols), T)
            @test !_vector_store_eligible(DestinationTile(mem, 0, rows, cols), T === Float64 ? Float32 : Float64)
        end
    end
end
