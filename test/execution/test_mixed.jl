# Mixed operand eltypes: compute type, conversion on load, one rounding on store.

using LinearAlgebra: norm

# `(Cv, Av, indA, Bv, indB, indC)` of shape `(M, K, N)`; `:scattered` takes each
# operand from `scattered_fixture` of its own eltype, `:split` is intensli_7 with
# M^4 rows.
function _mixed_fixture(TA, TB, TC, M, K, N, layout, seed)
    layout === :split && return _sp_views(TC, _SP_I7, (M, M, M, K, M, N), TA)
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
        (Float64, ComplexF64, ComplexF64, nothing, :scattered, (32, 32, 32), 0.3 - 0.7im, false, true, QuasiStrided._NestPath, _RC),
        (ComplexF32, Float64, ComplexF64, nothing, :scattered, (32, 32, 32), 0.5, true, false, QuasiStrided._NestPath, _CR),
        (Float64, ComplexF64, ComplexF64, nothing, :split, (8, 3, 5), 0.5, false, true, _NestPath{false, <:Any, (true, false)}, _RC),
        (ComplexF32, ComplexF64, ComplexF64, nothing, :dense, (19, 29, 23), 1, true, true, QuasiStrided._NestPath, _PROMOTED),
        (Float64, Float64, ComplexF64, nothing, :dense, (18, 25, 14), 0.25 + 0.5im, false, false, QuasiStrided._NestPath, _PROMOTED),
        (ComplexF64, Float32, ComplexF64, Float32, :dense, (17, 31, 13), 0, false, false, QuasiStrided._NestPath, _CR),
        (Float32, Float64, Float64, nothing, :dense, (1, 70, 29), 0.5, false, false, QuasiStrided._DotPath, SIMDKernel),
        (Float64, Float32, Float64, nothing, :dense, (37, 1, 21), 1, false, false, QuasiStrided._OuterPath, SIMDKernel),
    )
    fx = _mixed_fixture(TA, TB, TC, MKN..., layout, 7)
    alpha = 1.5
    ref = _mixed_ref(fx, alpha, beta; conjA, conjB)
    plan = (layout === :split ? _sp_forced_plan : plan_contract)(fx...; conjA, conjB, accumulator = acc)
    T = acc === nothing ? promote_type(TA, TB, TC) : (TC <: Complex ? Complex{acc} : acc)
    @test plan isa ContractPlan{T}
    @test plan.kernel isa K
    @test _path_of(plan) isa path
    execute!(plan, alpha, beta)
    k_length = layout === :scattered ? 32 : MKN[2]
    rtol = 10 * sqrt(k_length) * max(eps(real(T)), eps(real(TC)))
    @test norm(Array(fx[1]) - ref) <= rtol * norm(ref)
end

@testset "a named mixed-domain kernel with a complex RealFormat side throws at plan time" begin
    Cv, Av, Bv = (StridedView(zeros(ComplexF64, 8, 8)) for _ in 1:3)
    for kernel in (_CR(Val(4), Val(6), ComplexF64, Val(4)), _RC(Val(8), Val(3), ComplexF64, Val(4)))
        @test_throws ArgumentError plan_contract(Cv, Av, (1, 2), Bv, (2, 3), (1, 3); kernel)
    end
end

# Along K, 1, 2^-30 and -1 fall in different K blocks: the 2^-30 and the small
# beta*C survive only if the partial sums stay in the compute type between blocks.
@testset "one rounding to $TC across K blocks: $TA x $TB, M = $M, $case, beta = $beta" for (TA, TB, TC, acc, M, N, K, case, beta) in (
        (Float64, Float64, Float32, nothing, 13, 40, 20, :manual_n_block_16, 0.5),
        (Float32, Float32, Float32, Float64, 13, 9, 20, :packed_b, 1),
        (Float64, ComplexF64, ComplexF32, nothing, 13, 9, 20, :strided_c, 0.5),
        (Float32, Float32, Float32, Float64, 7, 5, 3000, :backend, 0),
        (Float64, Float64, Float32, nothing, 1, 9, 20, :strided_c, 1),
        (Float64, Float64, Float32, nothing, 13, 9, 20, :strided_c, 0),
    )
    b = TB <: Complex ? TB(1, 1) : one(TB)
    A = zeros(TA, M, K)
    A[:, 1] .= 1
    A[:, K ÷ 2 + 1] .= 2.0^-30
    A[:, K] .= -1
    Av, Bv = StridedView(A), StridedView(fill(b, K, N))
    case === :packed_b && (Bv = permutedims(StridedView(permutedims(Array(Bv))), (2, 1)))
    Cv = case === :strided_c ? StridedView(zeros(TC, 2M, 3N))[2:2:2M, 1:3:3N] : StridedView(zeros(TC, M, N))
    fill!(Cv, iszero(beta) ? NaN : 2.0^-28)
    if case === :backend
        @test K ÷ 2 > QuasiStrided._resolved_defaults(Float64).real_row.k_block
        backend = QuasiStrided.QuasiStridedBackend(accumulator = acc)
        TO.tensorcontract!(Cv, Av, ((1,), (2,)), false, Bv, ((1,), (2,)), false, ((1, 2), ()), 1, beta, backend)
    else
        allocator = case === :manual_n_block_16 ? TO.ManualAllocator() : TO.DefaultAllocator()
        n_block = case === :manual_n_block_16 ? 16 : nothing
        plan = plan_contract(Cv, Av, (1, 2), Bv, (2, 3), (1, 3); accumulator = acc, k_block = 4, n_block, allocator)
        @test _path_of(plan) isa QuasiStrided._PanelPath
        @test axis_length(plan.mgroup) == M
        case === :manual_n_block_16 && @test N > plan.blocking.n_block && !(plan.workspace.c_panel isa Vector)
        case === :packed_b && @test _path_of(plan) isa QuasiStrided._PanelPath{<:QuasiStrided._NestPath{false}}
        TB <: Complex && @test plan.kernel isa _RC
        execute!(plan, 1, beta)
        QuasiStrided.release!(plan.workspace, allocator)
    end
    @test all(==(TC(2.0^-30 * b + beta * 2.0^-28)), Array(Cv))
end

@testset "steady-state execute! allocates nothing: $TA x $TB -> $TC, accumulator = $acc" for (TA, TB, TC, acc, K) in (
        (Float32, ComplexF64, ComplexF64, nothing, 30), (Float32, Float32, Float32, Float64, 700),
    )
    rng = MersenneTwister(3)
    Amat, Bmat, Cmat = randn(rng, TA, 64, K), randn(rng, TB, K, 40), zeros(TC, 64, 40)
    plan = _mm_plan(Cmat, Amat, Bmat; accumulator = acc, k_block = 256)
    @test _path_of(plan) isa (TC === Float32 ? QuasiStrided._PanelPath : QuasiStrided._NestPath)
    @test _steady_allocs!(execute!, plan, Cmat) == 0 skip = (VERSION < v"1.11")
end
