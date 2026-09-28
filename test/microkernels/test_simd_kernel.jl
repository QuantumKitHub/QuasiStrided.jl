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

    @testset "_vector_store_eligible: unit-stride rows into 1-D dense real storage" begin
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
            @test _vector_store_eligible(DestinationTile(mem, 0, rows, cols), T === Float64 ? Float32 : Float64)
            @test !_vector_store_eligible(DestinationTile(zeros(complex(T), m * n), 0, rows, cols), T)
        end
    end

    @testset "vector store into converted storage: $T kernel, $S C" for (T, S) in ((Float64, Float32), (Float32, Float64))
        k = SIMDKernel(Val(8), Val(3), T)
        rng = MersenneTwister(5)
        for (m, n) in ((8, 3), (7, 3), (3, 2)), (alpha, beta) in mk_alphabeta(T)
            acc = map(v -> typeof(v)(ntuple(_ -> T(2rand(rng) - 1), lanewidth(k))), zero_accumulator(k))
            cold = S.(2 .* rand(rng, m * n) .- 1)
            fast = mk_dense(cold)
            dfast = DestinationTile(fast, 0, AffineAxis(0, 1, m), AffineAxis(0, m, n))
            @test _vector_store_eligible(dfast, T)
            store_tile!(dfast, acc, alpha, beta, k)
            scal = copy(cold)
            store_tile!(DestinationTile(scal, 0, ScatterAxis(collect(0:(m - 1)), m), AffineAxis(0, m, n)), acc, alpha, beta, k)
            @test mk_close(fast, scal, Float32)
        end
    end
end
