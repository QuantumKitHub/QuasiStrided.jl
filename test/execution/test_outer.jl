# The outer-product path: `execute!` on a real `K == 1` plan with unit-stride M
# in A and C streams `alpha * A * B[n]` column by column. Results match the
# nest and the reference (`≈`: the two differ on a signed zero).

include("helpers.jl")

@testset "outer path: which plans it takes ($T)" for T in (Float64, Float32, ComplexF64)
    W = _lanes(T)
    for variant in (:plain, :bstrided, :coffset, :multiM)
        p = plan_contract(_outer_maker(T, 16, 9, 1; variant)()...)
        @test _outer_takes(p) == (T <: Real)
    end
    T <: Real || continue
    @test !_outer_takes(plan_contract(_outer_maker(T, 16, 9, 1; variant = :agap)()...))
    # C stored [b, a]: the planner swaps roles, so the 9-long `b` becomes M.
    pc = plan_contract(_outer_maker(T, 16, 9, 1; variant = :cpermC)()...)
    @test axis_length(pc.mgroup) == 9 && _outer_takes(pc) == (9 >= W)
    # M below one vector (N kept below W too, so the roles cannot swap).
    @test !_outer_takes(plan_contract(_outer_maker(T, W - 1, W - 1, 1)()...))
    @test !_outer_takes(_mm_plan(zeros(T, 16, 9), randn(T, 16, 2), randn(T, 2, 9)))
end

@testset "outer path: matches the nest and the reference ($T)" for T in (Float64, Float32)
    W = _lanes(T)
    for (idx, (M, N)) in enumerate(((W, 1), (2W + 1, 13), (63, 63))), variant in (:plain, :bstrided, :coffset),
            (alpha, beta) in ((1.0, 0.0), (2.5, -0.75), (1.0, 1.0))
        mk = _outer_maker(T, M, N, 10 + idx; variant)
        C_out, plan = _run_fresh(execute!, mk, alpha, beta)
        C_nest, _ = _run_nest(mk, alpha, beta)
        @test _outer_takes(plan) == (axis_length(plan.mgroup) >= W)
        @test C_out ≈ C_nest
        @test C_out ≈ _ref_of(mk, alpha, beta)
    end
    # Composite M, several N blocks; and a singleton M beside a long N, which
    # swaps roles and runs on the 9-long N with a scalar tail.
    for (mk, kw) in ((_outer_maker(T, 32, 40, 50; variant = :multiM), (n_block = 6,)), (_outer_maker(T, 1, 9, 77), (;)))
        C_out, plan = _run_fresh(execute!, mk, 1.5, 0.5; kw...)
        @test _outer_takes(plan) == (axis_length(plan.mgroup) >= W)
        @test C_out ≈ _ref_of(mk, 1.5, 0.5)
    end
    # beta = 0 never reads a NaN C.
    mk = _outer_maker(T, 2W + 3, 9, 4; Cfill = NaN)
    C_out, _ = _run_fresh(execute!, mk, 2.0, 0.0)
    @test C_out ≈ _ref_of(mk, 2.0, 0.0)
end

@testset "outer path: bounds, allocations, the backend ($T)" for T in (Float64, Float32)
    W = _lanes(T)
    M, N = 2W, 5
    # One element short on each operand: rejected before any write.
    for (sa, sb, sc) in ((M - 1, N, M * N), (M, N - 1, M * N), (M, N, M * N - 1))
        plan = plan_contract(
            StridedView(zeros(T, sc), (M, N), (1, M), 0), StridedView(randn(T, sa), (M,), (1,), 0), (1,),
            StridedView(randn(T, sb), (N,), (1,), 0), (2,), (1, 2)
        )
        @test _outer_takes(plan)
        @test_throws BoundsError execute!(plan, 1.0, 0.0)
        @test all(iszero, plan.Cstorage)
    end
    plan = plan_contract(_outer_maker(T, 64, 50, 6)()...; n_block = 12)
    @test _outer_takes(plan)
    execute!(plan, 1.0, 0.0)
    @test (@allocated execute!(plan, 1.0, 0.0)) == 0 skip = (VERSION < v"1.11")
    a, b, C = randn(T, 63), randn(T, 63), zeros(T, 63, 63)
    pA, pB, pAB = TO.contract_indices((:a,), (:b,), (:a, :b))
    TO.tensorcontract!(C, a, pA, false, b, pB, false, pAB, 1.0, 0.0, QuasiStridedBackend())
    @test C ≈ a * transpose(b)
end
