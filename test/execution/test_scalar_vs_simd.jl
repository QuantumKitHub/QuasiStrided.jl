@testset "driver: ScalarKernel and SIMDKernel agree through execute!" begin
    rng = MersenneTwister(20260908)
    Amat, Bmat, Cstart = rand(rng, 11, 13), rand(rng, 13, 9), rand(rng, 11, 9)
    for (m, n, kc) in ((Val(8), Val(6), 5), (Val(4), Val(3), 4))
        Cs, Cv = copy(Cstart), copy(Cstart)
        execute!(_mm_plan(Cs, Amat, Bmat; kernel = ScalarKernel(m, n, Float64), kc = kc), 2.5, 0.75)
        execute!(_mm_plan(Cv, Amat, Bmat; kernel = SIMDKernel(m, n, Float64), kc = kc), 2.5, 0.75)
        @test Cs ≈ 2.5 .* (Amat * Bmat) .+ 0.75 .* Cstart
        @test isapprox(Cs, Cv; atol = 1.0e-10)
    end
end
