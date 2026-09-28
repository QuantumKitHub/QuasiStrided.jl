# The outer-product path (src/execution/outer.jl): `execute!` on a real plan
# with `K == 1` and unit-stride M in A and C streams `alpha * A * B[n]` column
# by column instead of running the nest. Checked here:
#
#   1. the outer-product path takes exactly the eligible plans (real `T`,
#      `K == 1`, unit-ramp M in both A and C, `Qm >= W`, dense A/C storage),
#      declines the rest untouched, and obeys `_OUTER_MODE`.
#   2. On eligible plans the result matches the nest (`_OUTER_MODE[] =
#      :never`), the oracle and a brute-force reference: `≈`, since the nest's
#      FMA from `+0.0` and this path's plain product differ on a signed zero.
#   3. beta = 0 never reads C; out-of-bounds operands are rejected before any
#      write; the call is allocation-free.

const _outer_mode = QuasiStrided._OUTER_MODE

function _outer_run(mk, mode::Symbol, alpha, beta; plankw...)
    Cv, Av, indA, Bv, indB, indC = mk()
    plan = plan_contract(Cv, Av, indA, Bv, indB, indC; plankw...)
    old = _outer_mode[]
    _outer_mode[] = mode
    try
        @test execute!(plan, alpha, beta) === plan.Cstorage
    finally
        _outer_mode[] = old
    end
    return Array(Cv), plan
end

# Whether the automatic path would take `plan` (without running it).
_outer_eligible(plan) =
    axis_length(plan.kgroup) == 1 && QuasiStrided._outer_applicable(plan, axis_length(plan.mgroup))

# `C[a, b] = A[a] * B[b]` (the suite's `*_1_0_1_gemm_ready`) at extents
# `(M, N)`, with variants:
#   :plain      dense everything
#   :bstrided   B every other element of a longer vector
#   :cpermC     C stored [b, a] and viewed back (M not unit-stride in C)
#   :agap       A every other element (M not unit-stride in A)
#   :coffset    C a view with an offset into a larger parent, M unit-stride
#   :multiM     M a composite of two labels, C[a1,a2,b] with (a1,a2) folding
function _outer_maker(::Type{T}, M, N, seed; variant = :plain, Cfill = nothing) where {T}
    return function ()
        rng = MersenneTwister(seed)
        a = randn(rng, T, M)
        b = randn(rng, T, N)
        C = Cfill === nothing ? randn(rng, T, M, N) : fill(convert(T, Cfill), M, N)
        Av = if variant === :agap
            big = fill(convert(T, NaN), 2M)
            big[1:2:(2M)] .= a
            StridedView(view(big, 1:2:(2M)))
        else
            StridedView(a)
        end
        Bv = if variant === :bstrided
            big = fill(convert(T, NaN), 2N)
            big[2:2:(2N)] .= b
            StridedView(view(big, 2:2:(2N)))
        else
            StridedView(b)
        end
        Cv = if variant === :cpermC
            permutedims(StridedView(permutedims(C, (2, 1))), (2, 1))
        elseif variant === :coffset
            big = fill(convert(T, NaN), M + 3, N + 2)
            big[2:(M + 1), 2:(N + 1)] .= C
            StridedView(view(big, 2:(M + 1), 2:(N + 1)))
        else
            StridedView(C)
        end
        @assert isequal(Array(Av), a) && isequal(Array(Bv), b) && isequal(Array(Cv), C)
        if variant === :multiM
            M1 = 4
            @assert M % M1 == 0
            return (
                StridedView(reshape(C, M1, M ÷ M1, N)), StridedView(reshape(a, M1, M ÷ M1)), (1, 2),
                Bv, (3,), (1, 2, 3),
            )
        end
        return (Cv, Av, (1,), Bv, (2,), (1, 2))
    end
end

const _OUTER_AB = ((1.0, 0.0), (2.5, -0.75), (1.0, 1.0), (-0.5, 1.25))

@testset "outer path: which plans it takes ($T)" for T in (Float64, Float32, ComplexF64, ComplexF32)
    W = QuasiStrided._dot_lanewidth(T)
    for variant in (:plain, :bstrided, :coffset, :multiM)
        p = plan_contract(_outer_maker(T, 16, 9, 1; variant = variant)()...)
        @test axis_length(p.kgroup) == 1
        @test _outer_eligible(p) == (T <: Real)
    end
    if T <: Real
        # M not unit-stride in A: declined.
        @test !_outer_eligible(plan_contract(_outer_maker(T, 16, 9, 1; variant = :agap)()...))
        # C stored [b, a]: the planner swaps the roles, so the label that is
        # unit-stride in C becomes M (here the 9-long `b`, whose operand is
        # dense too) and the path applies iff that M still fills a vector.
        pc = plan_contract(_outer_maker(T, 16, 9, 1; variant = :cpermC)()...)
        @test axis_length(pc.mgroup) == 9
        @test _outer_eligible(pc) == (9 >= W)
        # Below one vector of M: declined. N is kept below W too: where W = 2
        # (NEON Float64) `W - 1` is a singleton axis, which makes C's N axis
        # unit-stride as well, so with a long N the planner legitimately
        # swaps the roles and takes the 9-long N as M (as :cpermC above).
        @test !_outer_eligible(plan_contract(_outer_maker(T, W - 1, W - 1, 1)()...))
        # With a long N the bound applies to the M the plan ends up with.
        p1 = plan_contract(_outer_maker(T, W - 1, 9, 1)()...)
        @test axis_length(p1.mgroup) in (W - 1, 9)
        @test _outer_eligible(p1) == (axis_length(p1.mgroup) >= W)
        # K > 1: not an outer product.
        @test !_outer_eligible(_mm_plan(zeros(T, 16, 9), randn(T, 16, 2), randn(T, 2, 9)))
        # The mode override.
        p = plan_contract(_outer_maker(T, 16, 9, 1)()...)
        _outer_mode[] = :never
        @test !_outer_eligible(p)
        _outer_mode[] = :auto
        @test _outer_eligible(p)
    end
end

@testset "outer path: matches the nest, the oracle and a reference ($T)" for T in (Float64, Float32)
    W = QuasiStrided._dot_lanewidth(T)
    shapes = ((W, 1), (W, 7), (W + 3, 5), (2W + 1, 13), (63, 63), (128, 128), (40, 3))
    for (idx, (M, N)) in enumerate(shapes), variant in (:plain, :bstrided, :coffset, :cpermC, :agap), (alpha, beta) in _OUTER_AB
        mk = _outer_maker(T, M, N, 10 + idx; variant = variant)
        C_nest, _ = _outer_run(mk, :never, alpha, beta)
        C_out, plan = _outer_run(mk, :auto, alpha, beta)
        # :agap is never eligible (A has stride 2); :cpermC only when the
        # planner swapped the roles so that M is unit-stride in C as well.
        @test _outer_eligible(plan) == (
            variant !== :agap && all(==((1,)), plan.mgroup.strides) && axis_length(plan.mgroup) >= W
        )
        @test C_out ≈ C_nest
        Cv0, Av0, iA, Bv0, iB, iC = mk()
        @test C_out ≈ _hp_ref(Array(Av0), iA, Array(Bv0), iB, Array(Cv0), iC, alpha, beta)
        C_tw = let (Cv, Av, iA, Bv, iB, iC) = mk()
            execute_tilewise!(plan_contract(Cv, Av, iA, Bv, iB, iC), alpha, beta)
            Array(Cv)
        end
        @test C_out ≈ C_tw
    end
    # Composite M (two labels folding to one unit ramp) and several N blocks.
    for (M, N) in ((32, 40), (128, 7))
        mk = _outer_maker(T, M, N, 50 + M; variant = :multiM)
        C_nest, _ = _outer_run(mk, :never, 1.5, 0.5; nc = 6)
        C_out, plan = _outer_run(mk, :auto, 1.5, 0.5; nc = 6)
        @test _outer_eligible(plan)
        @test C_out ≈ C_nest
        Cv0, Av0, iA, Bv0, iB, iC = mk()
        @test C_out ≈ _hp_ref(Array(Av0), iA, Array(Bv0), iB, Array(Cv0), iC, 1.5, 0.5)
    end
    # A singleton M beside a long N: C's N axis is unit-stride too, so the
    # planner swaps the roles and the path runs on the 9-long N, with a scalar
    # tail wherever W does not divide 9 -- the plan NEON Float64 (W = 2)
    # takes for `M = W - 1` above.
    for (alpha, beta) in _OUTER_AB
        mk = _outer_maker(T, 1, 9, 77)
        C_nest, _ = _outer_run(mk, :never, alpha, beta)
        C_out, plan = _outer_run(mk, :auto, alpha, beta)
        @test axis_length(plan.mgroup) == 9
        @test _outer_eligible(plan) == (9 >= W)
        @test C_out ≈ C_nest
        Cv0, Av0, iA, Bv0, iB, iC = mk()
        @test C_out ≈ _hp_ref(Array(Av0), iA, Array(Bv0), iB, Array(Cv0), iC, alpha, beta)
    end
end

@testset "outer path: beta = 0 never reads C, bounds, allocations ($T)" for T in (Float64, Float32)
    W = QuasiStrided._dot_lanewidth(T)
    mk = _outer_maker(T, 2W + 3, 9, 4; Cfill = NaN)
    C_out, plan = _outer_run(mk, :auto, 2.0, 0.0)
    @test _outer_eligible(plan)
    @test all(isfinite, C_out)
    Cv0, Av0, _, Bv0, _, _ = mk()
    @test C_out ≈ 2.0 .* (Array(Av0) * transpose(Array(Bv0)))

    # One element short on each operand: rejected before any write.
    M, N = 2W, 5
    mkA = () -> (StridedView(zeros(T, M, N)), StridedView(randn(T, M - 1), (M,), (1,), 0), (1,), StridedView(randn(T, N)), (2,), (1, 2))
    mkB = () -> (StridedView(zeros(T, M, N)), StridedView(randn(T, M)), (1,), StridedView(randn(T, N - 1), (N,), (1,), 0), (2,), (1, 2))
    mkC = () -> (StridedView(zeros(T, M * N - 1), (M, N), (1, M), 0), StridedView(randn(T, M)), (1,), StridedView(randn(T, N)), (2,), (1, 2))
    for mk in (mkA, mkB, mkC)
        Cv, Av, iA, Bv, iB, iC = mk()
        plan = plan_contract(Cv, Av, iA, Bv, iB, iC)
        @test _outer_eligible(plan)
        @test_throws BoundsError execute!(plan, 1.0, 0.0)
        @test all(iszero, plan.Cstorage)
    end

    Cv, Av, iA, Bv, iB, iC = _outer_maker(T, 64, 50, 6)()
    plan = plan_contract(Cv, Av, iA, Bv, iB, iC; oracle = false, nc = 12)
    @test _outer_eligible(plan)
    execute!(plan, 1.0, 0.0)
    execute!(plan, 1.0, 0.0)
    @test (@allocated execute!(plan, 1.0, 0.0)) == 0 skip = (VERSION < v"1.11")
end

@testset "outer path: through the TensorOperations backend ($T)" for T in (Float64, Float32, ComplexF64)
    a = randn(T, 63); b = randn(T, 63)
    C1 = zeros(T, 63, 63); C2 = zeros(T, 63, 63)
    pA, pB, pAB = TO.contract_indices((:a,), (:b,), (:a, :b))
    _outer_mode[] = :never
    TO.tensorcontract!(C1, a, pA, false, b, pB, false, pAB, 1.0, 0.0, QuasiStridedBackend())
    _outer_mode[] = :auto
    TO.tensorcontract!(C2, a, pA, false, b, pB, false, pAB, 1.0, 0.0, QuasiStridedBackend())
    @test C2 ≈ C1
    @test C2 ≈ a * transpose(b)
end
