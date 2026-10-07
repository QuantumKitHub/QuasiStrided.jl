# The C panel: partial sums of a narrower C stay in the compute type across K blocks.

include("helpers.jl")

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
        @test K ÷ 2 > default_blocking(auto_kernel(Float64, M)).k_block
        backend = QuasiStrided.QuasiStridedBackend(accumulator = acc)
        TO.tensorcontract!(Cv, Av, ((1,), (2,)), false, Bv, ((1,), (2,)), false, ((1, 2), ()), 1, beta, backend)
    else
        allocator = case === :manual_n_block_16 ? TO.ManualAllocator() : TO.DefaultAllocator()
        n_block = case === :manual_n_block_16 ? 16 : nothing
        plan = plan_contract(Cv, Av, (1, 2), Bv, (2, 3), (1, 3); accumulator = acc, k_block = 4, n_block, allocator)
        @test _path_of(plan) isa QuasiStrided.PanelPath
        @test axis_length(plan.mgroup) == M
        case === :manual_n_block_16 && @test N > plan.blocking.n_block && !(plan.workspace.c_panel isa Vector)
        case === :packed_b && @test _path_of(plan) isa QuasiStrided.PanelPath{<:QuasiStrided.NestPath{false}}
        TB <: Complex && @test plan.kernel isa RC
        execute!(plan, 1, beta)
        QuasiStrided.release!(plan.workspace, allocator)
    end
    @test all(==(TC(2.0^-30 * b + beta * 2.0^-28)), Array(Cv))
end
