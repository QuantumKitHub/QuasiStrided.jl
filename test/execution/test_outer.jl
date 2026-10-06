# The outer-product path: `execute!` on a real `K == 1` plan with unit-stride M
# in A and C streams `alpha * A * B[n]` column by column. Results match the
# nest and the reference (`≈`: the two differ on a signed zero).

_outer_takes(plan) = _path_of(plan) isa QuasiStrided.OuterPath

# `C[a, b] = A[a] * B[b]` at extents `(M, N)`:
#   :bstrided  B every other element of a longer vector
#   :cpermC    C stored [b, a] (M not unit-stride in C)
#   :agap      A every other element (M not unit-stride in A)
#   :coffset   C a view with an offset, M unit-stride
#   :multiM    M a composite of two labels that fold to one ramp
function _outer_maker(::Type{T}, M, N, seed; variant = :plain, Cfill = nothing) where {T}
    return function ()
        rng = MersenneTwister(seed)
        a, b = randn(rng, T, M), randn(rng, T, N)
        C = Cfill === nothing ? randn(rng, T, M, N) : fill(convert(T, Cfill), M, N)
        gapped(v, r) = (big = fill(convert(T, NaN), 2 * length(v)); big[r] .= v; StridedView(view(big, r)))
        Av = variant === :agap ? gapped(a, 1:2:(2M)) : StridedView(a)
        Bv = variant === :bstrided ? gapped(b, 2:2:(2N)) : StridedView(b)
        variant === :multiM &&
            return (StridedView(reshape(C, 4, M ÷ 4, N)), StridedView(reshape(a, 4, M ÷ 4)), (1, 2), Bv, (3,), (1, 2, 3))
        Cv = if variant === :cpermC
            permutedims(StridedView(permutedims(C, (2, 1))), (2, 1))
        elseif variant === :coffset
            big = fill(convert(T, NaN), M + 3, N + 2)
            big[2:(M + 1), 2:(N + 1)] .= C
            StridedView(view(big, 2:(M + 1), 2:(N + 1)))
        else
            StridedView(C)
        end
        return (Cv, Av, (1,), Bv, (2,), (1, 2))
    end
end

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

@testset "the path modes override path selection" begin
    pd(; kw...) = plan_contract(_mm_maker(Float64, 1, 64, 9, 1)()...; kw...)
    po(; kw...) = plan_contract(_outer_maker(Float64, 16, 9, 1)()...; kw...)
    pu(; kw...) = plan_contract(_mm_maker(Float64, 20, 12, 9, 1)()...; kw...)
    ps(; kw...) = plan_contract(_mm_maker(Float64, 20, 12, 9, 1; B = :transposed)()...; kw...)
    @test _dot_takes(pd()) && _outer_takes(po()) && _ub_takes(pu()) && !_ub_takes(ps())
    @test !_dot_takes(pd(; path_modes = _NEST_ONLY)) && !_outer_takes(po(; path_modes = _NEST_ONLY)) &&
        !_ub_takes(pu(; path_modes = _NEST_ONLY))
    @test _ub_takes(ps(; path_modes = QuasiStrided.PathModes(unpacked_b = :always)))
end
