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

const _CR, _RC = QuasiStrided.ComplexRealKernel, QuasiStrided.RealComplexKernel
const _PROMOTED = Union{QuasiStrided.PlanarKernel, QuasiStrided.FMAddSubKernel}

@testset "mixed eltypes: $TA x $TB -> $TC, accumulator = $acc, $layout" for (TA, TB, TC, acc, layout, MKN, beta, conjA, conjB, path, K) in (
        (Float32, Float64, Float64, nothing, :dense, (23, 37, 19), 0, false, false, QuasiStrided._NestPath, SIMDKernel),
        (Float64, Float32, Float32, nothing, :dense, (32, 40, 12), 1, false, false, QuasiStrided._NestPath, SIMDKernel),
        (Float64, ComplexF64, ComplexF64, nothing, :dense, (21, 33, 17), 0.3 - 0.7im, false, true, QuasiStrided._NestPath, _RC),
        (ComplexF32, Float64, ComplexF64, nothing, :scattered, (32, 32, 32), 0.5, true, false, QuasiStrided._NestPath, _CR),
        (ComplexF32, ComplexF64, ComplexF64, nothing, :dense, (19, 29, 23), 1, true, true, QuasiStrided._NestPath, _PROMOTED),
        (Float64, Float64, ComplexF64, nothing, :dense, (18, 25, 14), 0.25 + 0.5im, false, false, QuasiStrided._NestPath, _PROMOTED),
        (ComplexF64, Float32, ComplexF64, Float32, :dense, (17, 31, 13), 0, false, false, QuasiStrided._NestPath, _CR),
        (Float32, Float64, Float64, nothing, :dense, (1, 70, 29), 0.5, false, false, QuasiStrided._DotPath, SIMDKernel),
        (Float64, Float32, Float64, nothing, :dense, (37, 1, 21), 1, false, false, QuasiStrided._OuterPath, SIMDKernel),
    )
    fx = _mixed_fixture(TA, TB, TC, MKN..., layout, 7)
    alpha = 1.5
    ref = _mixed_ref(fx, alpha, beta; conjA, conjB)
    plan = plan_contract(fx...; conjA, conjB, accumulator = acc)
    T = acc === nothing ? promote_type(TA, TB, TC) : (TC <: Complex ? Complex{acc} : acc)
    @test plan isa ContractPlan{T}
    @test plan.kernel isa K
    @test _path_of(plan) isa path
    execute!(plan, alpha, beta)
    Qk = layout === :scattered ? 32 : MKN[2]
    rtol = 10 * sqrt(Qk) * max(eps(real(T)), eps(real(TC)))
    @test norm(Array(fx[1]) - ref) <= rtol * norm(ref)
end

@testset "a named mixed-domain kernel with a complex RealFormat side throws at plan time" begin
    Cv, Av, Bv = (StridedView(zeros(ComplexF64, 8, 8)) for _ in 1:3)
    for kernel in (_CR(Val(4), Val(6), ComplexF64, Val(4)), _RC(Val(8), Val(3), ComplexF64, Val(4)))
        @test_throws ArgumentError plan_contract(Cv, Av, (1, 2), Bv, (2, 3), (1, 3); kernel)
    end
end

@testset "mixed eltypes: steady-state execute! allocates nothing" begin
    rng = MersenneTwister(3)
    Amat, Bmat, Cmat = randn(rng, Float32, 40, 30), randn(rng, ComplexF64, 30, 20), zeros(ComplexF64, 40, 20)
    @test _steady_allocs!(execute!, _mm_plan(Cmat, Amat, Bmat), Cmat) == 0 skip = (VERSION < v"1.11")
end

@testset "Float32 C, Float64 compute type, K over several kc blocks: within one ulp ($TA, $layout, beta = $beta)" for (TA, acc, layout) in (
            (Float32, Float64, :dense), (Float64, nothing, :dense), (Float32, Float64, :scattered),
        ), beta in (0, 0.5)
    fx = if layout === :scattered
        Cv = scattered_fixture(Float32, 32, 512)[1]
        copyto!(Cv, randn(MersenneTwister(4), Float32, size(Cv)))
        (Cv, scattered_fixture(TA, 32, 512)[2:end]...)
    else
        rng = MersenneTwister(5)
        A, B, C = randn(rng, TA, 64, 512), randn(rng, TA, 512, 48), randn(rng, Float32, 64, 48)
        (StridedView(C), StridedView(A), (1, 2), StridedView(B), (2, 3), (1, 3))
    end
    ref = Float32.(_mixed_ref(fx, 1, beta))
    C32 = copy(fx[1])
    execute!(plan_contract(C32, map(x -> x isa StridedView ? StridedView(Float32.(x)) : x, fx[2:end])...), 1, beta)
    plan = plan_contract(fx...; accumulator = acc)
    @test _path_of(plan) isa QuasiStrided._PanelPath
    @test axis_length(plan.kgroup) > plan.blocking.kc
    execute!(plan, 1, beta)
    err = abs.(Array(fx[1]) .- ref)
    @test all(err .<= eps.(ref))
    @test maximum(err) < maximum(abs.(Array(C32) .- ref))
end

@testset "Float32 C, Float64 compute type: steady-state execute! allocates nothing" begin
    rng = MersenneTwister(6)
    Amat, Bmat, Cmat = randn(rng, Float32, 64, 700), randn(rng, Float32, 700, 40), zeros(Float32, 64, 40)
    plan = _mm_plan(Cmat, Amat, Bmat; accumulator = Float64)
    @test _path_of(plan) isa QuasiStrided._PanelPath
    @test _steady_allocs!(execute!, plan, Cmat) == 0 skip = (VERSION < v"1.11")
end
