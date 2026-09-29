# Mixed operand eltypes: compute type, conversion on load, one rounding on store.

using LinearAlgebra: norm

# `(Cv, Av, indA, Bv, indB, indC)` of shape `(M, K, N)`; `:scattered` takes each
# operand from `scattered_fixture` of its own eltype.
function _mixed_fixture(TA, TB, TC, M, K, N, layout, seed)
    layout === :scattered &&
        return (scattered_fixture(TC)[1], scattered_fixture(TA)[2:3]..., scattered_fixture(TB)[4:6]...)
    rng = MersenneTwister(seed)
    A, B, C = randn(rng, TA, M, K), randn(rng, TB, K, N), randn(rng, TC, M, N)
    return (StridedView(C), StridedView(A), (1, 2), StridedView(B), (2, 3), (1, 3))
end

# The reference in Float64/ComplexF64 from the exactly promoted inputs.
function _mixed_ref(fx, alpha, beta; kw...)
    Cv, Av, iA, Bv, iB, iC = fx
    R = promote_type(eltype(Av), eltype(Bv), eltype(Cv)) <: Complex ? ComplexF64 : Float64
    return _brute_ref(R.(Array(Av)), iA, R.(Array(Bv)), iB, R.(Array(Cv)), iC, alpha, beta; kw...)
end

@testset "mixed eltypes: $TA x $TB -> $TC, accumulator = $acc, $layout" for (TA, TB, TC, acc, layout, MKN, beta, conjA, conjB, path) in (
        (Float32, Float64, Float64, nothing, :dense, (23, 37, 19), 0, false, false, QuasiStrided._NestPath),
        (Float64, Float32, Float32, nothing, :dense, (32, 40, 12), 1, false, false, QuasiStrided._NestPath),
        (Float64, ComplexF64, ComplexF64, nothing, :dense, (21, 33, 17), 0.3 - 0.7im, false, true, QuasiStrided._NestPath),
        (ComplexF32, Float64, ComplexF64, nothing, :scattered, (32, 32, 32), 0.5, true, false, QuasiStrided._NestPath),
        (ComplexF32, ComplexF64, ComplexF64, nothing, :dense, (19, 29, 23), 1, true, true, QuasiStrided._NestPath),
        (Float64, Float64, ComplexF64, nothing, :dense, (18, 25, 14), 0.25 + 0.5im, false, false, QuasiStrided._NestPath),
        (ComplexF64, Float32, ComplexF64, Float32, :dense, (17, 31, 13), 0, false, false, QuasiStrided._NestPath),
        (Float32, Float64, Float64, nothing, :dense, (1, 70, 29), 0.5, false, false, QuasiStrided._DotPath),
        (Float64, Float32, Float64, nothing, :dense, (37, 1, 21), 1, false, false, QuasiStrided._OuterPath),
    )
    fx = _mixed_fixture(TA, TB, TC, MKN..., layout, 7)
    alpha = 1.5
    ref = _mixed_ref(fx, alpha, beta; conjA, conjB)
    plan = plan_contract(fx...; conjA, conjB, accumulator = acc)
    T = acc === nothing ? promote_type(TA, TB, TC) : (TC <: Complex ? Complex{acc} : acc)
    @test plan isa ContractPlan{T}
    @test _path_of(plan) isa path
    execute!(plan, alpha, beta)
    K = layout === :scattered ? 32 : MKN[2]
    rtol = 10 * sqrt(K) * max(eps(real(T)), eps(real(TC)))
    @test norm(Array(fx[1]) - ref) <= rtol * norm(ref)
end

@testset "mixed eltypes: steady-state execute! allocates nothing" begin
    rng = MersenneTwister(3)
    Amat, Bmat, Cmat = randn(rng, Float32, 40, 30), randn(rng, ComplexF64, 30, 20), zeros(ComplexF64, 40, 20)
    @test _steady_allocs!(execute!, _mm_plan(Cmat, Amat, Bmat), Cmat) == 0 skip = (VERSION < v"1.11")
end

# One K panel: between panels the partial sums round-trip through C.
@testset "accumulator = Float64: Float32 operands within one ulp" begin
    rng = MersenneTwister(5)
    A, B, C0 = randn(rng, Float32, 64, 512), randn(rng, Float32, 512, 48), randn(rng, Float32, 64, 48)
    ref = Float32.(Float64.(A) * Float64.(B) .+ 0.5 .* Float64.(C0))
    C64, C32 = copy(C0), copy(C0)
    execute!(_mm_plan(C64, A, B; accumulator = Float64, kc = 512), 1, 0.5)
    execute!(_mm_plan(C32, A, B), 1, 0.5)
    @test all(abs.(C64 .- ref) .<= eps.(ref))
    @test maximum(abs.(C64 .- ref)) < maximum(abs.(C32 .- ref))
end
