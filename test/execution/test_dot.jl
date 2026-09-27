# The dot-product path (src/execution/dot.jl): `execute!` on a plan with
# `M == 1` or `N == 1` whose matrix operand is K-contiguous runs a K-vectorized
# gemv instead of the five-loop nest. Checked here:
#
#   1. the dot path takes exactly the eligible plans (degenerate free
#      extent, unit-ramp K on the matrix operand, `DenseVector` storage,
#      `Qk >= W`), declines the rest without touching C, and obeys `_DOT_MODE`.
#   2. On eligible plans the result matches the nest (`_DOT_MODE[] = :never`),
#      the oracle and a brute-force reference to a tolerance (the K summation
#      order differs, so never `==`), over all four eltypes, conj flags and
#      views, alpha/beta regimes, permuted vector and output operands, K tails
#      and several K blocks, and the scalar-output case `M == N == 1`.
#   3. beta = 0 never reads C; out-of-bounds operands are rejected before any
#      write; the call is allocation-free.
#
# `_hp_ref` (test/execution/test_halfpack.jl) is the brute-force reference.

const _dot_mode = QuasiStrided._DOT_MODE

const _DOT_TYPES = (Float64, Float32, ComplexF64, ComplexF32)

# Run the plan under `mode`; returns the output array.
function _dot_run(mk, mode::Symbol, alpha, beta; plankw...)
    Cv, Av, indA, Bv, indB, indC = mk()
    plan = plan_contract(Cv, Av, indA, Bv, indB, indC; plankw...)
    old = _dot_mode[]
    _dot_mode[] = mode
    try
        @test execute!(plan, alpha, beta) === plan.Cstorage
    finally
        _dot_mode[] = old
    end
    return Array(Cv), plan
end

# Whether the automatic path would take `plan` (without running it).
function _dot_eligible(plan)
    Qm = axis_length(plan.mgroup); Qn = axis_length(plan.ngroup); Qk = axis_length(plan.kgroup)
    return QuasiStrided._dot_applicable(plan, Qm, Qn, Qk)
end

# The suite's `C[cde] = A[ab] * B[abcde]` shape (M = 1, matrix B K-fastest)
# at leg dimension `d`, plus variants:
#   :plain      as is
#   :permvec    A stored [b,a]: the VECTOR's K map is not a ramp (matrix is)
#   :permC      C stored [e,d,c] and viewed back: scattered output offsets
#   :offsetB    B a view with a nonzero offset into a larger parent
function _dot_gemv_maker(::Type{T}, d, seed; variant = :plain, Cfill = nothing) where {T}
    return function ()
        rng = MersenneTwister(seed)
        A = randn(rng, T, d, d)
        B = randn(rng, T, d, d, d, d, d)
        C = Cfill === nothing ? randn(rng, T, d, d, d) : fill(convert(T, Cfill), d, d, d)
        Av = variant === :permvec ? permutedims(StridedView(permutedims(A, (2, 1))), (2, 1)) : StridedView(A)
        Bv = if variant === :offsetB
            big = fill(convert(T, NaN), d, d, d, d, d + 2)
            big[:, :, :, :, 2:(d + 1)] .= B
            StridedView(view(big, :, :, :, :, 2:(d + 1)))
        else
            StridedView(B)
        end
        Cv = variant === :permC ? permutedims(StridedView(permutedims(C, (3, 2, 1))), (3, 2, 1)) : StridedView(C)
        @assert isequal(Array(Av), A) && isequal(Array(Bv), B) && isequal(Array(Cv), C)
        return (Cv, Av, (1, 2), Bv, (1, 2, 3, 4, 5), (3, 4, 5))
    end
end

# N = 1 with the matrix operand A K-fastest: C[a,b] = A[k,a,b] * v[k].
function _dot_n1_maker(::Type{T}, d, K, seed) where {T}
    return function ()
        rng = MersenneTwister(seed)
        A = randn(rng, T, K, d, d)
        v = randn(rng, T, K)
        C = randn(rng, T, d, d)
        return (StridedView(C), StridedView(A), (3, 1, 2), StridedView(v), (3,), (1, 2))
    end
end

const _DOT_AB = ((1.0, 0.0), (2.5, -0.75), (1.0, 1.0), (-0.5, 1.25))

@testset "dot path: which plans it takes ($T)" for T in _DOT_TYPES
    W = QuasiStrided._dot_lanewidth(T)
    @test W >= 2
    # The suite shape, both variants of the vector operand: eligible.
    for variant in (:plain, :permvec, :permC, :offsetB)
        p = plan_contract(_dot_gemv_maker(T, 4, 1; variant = variant)()...)
        @test 1 in (axis_length(p.mgroup), axis_length(p.ngroup))
        @test _dot_eligible(p)
    end
    # N = 1 with A K-fastest: eligible; with A K-slowest: not.
    @test _dot_eligible(plan_contract(_dot_n1_maker(T, 4, 16, 2)()...))
    pk = plan_contract(
        StridedView(zeros(T, 4, 4)), StridedView(randn(T, 4, 4, 16)), (1, 2, 3),
        StridedView(randn(T, 16)), (3,), (1, 2)
    )
    @test !_dot_eligible(pk)
    # Below one vector of K: declined.
    ps = plan_contract(
        StridedView(zeros(T, 1, 9)), StridedView(randn(T, 1, W - 1)), (1, 2),
        StridedView(randn(T, W - 1, 9)), (2, 3), (1, 3)
    )
    @test !_dot_eligible(ps)
    # Neither extent degenerate: declined.
    @test !_dot_eligible(_mm_plan(zeros(T, 3, 5), randn(T, 3, 20), randn(T, 20, 5)))
    # The mode override.
    p = plan_contract(_dot_gemv_maker(T, 4, 1)()...)
    _dot_mode[] = :never
    @test !_dot_eligible(p)
    _dot_mode[] = :auto
    @test _dot_eligible(p)
    # Non-DenseVector matrix storage is declined (raw-pointer loads need it).
    K = 4W
    wrapped = StridedView(_HPDenseMat(randn(T, K, 5)), (K, 5), (1, K), 0)
    pw = plan_contract(StridedView(zeros(T, 1, 5)), StridedView(randn(T, 1, K)), (1, 2), wrapped, (2, 3), (1, 3))
    @test !_dot_eligible(pw)
end

@testset "dot path: the suite's M = 1 gemv, all variants and regimes ($T)" for T in _DOT_TYPES
    for d in (2, 4, 6), variant in (:plain, :permvec, :permC, :offsetB), (alpha, beta) in _DOT_AB
        mk = _dot_gemv_maker(T, d, 10 * d; variant = variant)
        C_nest, _ = _dot_run(mk, :never, alpha, beta)
        C_dot, plan = _dot_run(mk, :auto, alpha, beta)
        @test _dot_eligible(plan) == (d * d >= QuasiStrided._dot_lanewidth(T))
        @test C_dot ≈ C_nest
        Cv0, Av0, iA, Bv0, iB, iC = mk()
        @test C_dot ≈ _hp_ref(Array(Av0), iA, Array(Bv0), iB, Array(Cv0), iC, alpha, beta)
        C_tw = let (Cv, Av, iA, Bv, iB, iC) = mk()
            execute_tilewise!(plan_contract(Cv, Av, iA, Bv, iB, iC), alpha, beta)
            Array(Cv)
        end
        @test C_dot ≈ C_tw
    end
end

@testset "dot path: N = 1, scalar output, K tails and several K blocks ($T)" for T in _DOT_TYPES
    W = QuasiStrided._dot_lanewidth(T)
    for (d, K) in ((3, W), (4, W + 1), (5, 3W - 1), (2, 1000)), (alpha, beta) in _DOT_AB
        mk = _dot_n1_maker(T, d, K, 3 * K + d)
        # `kc = 64` forces several K blocks at K = 1000; `mc = 7` several
        # output blocks against a 25-long free composite.
        C_nest, _ = _dot_run(mk, :never, alpha, beta; kc = 64, mc = 7)
        C_dot, plan = _dot_run(mk, :auto, alpha, beta; kc = 64, mc = 7)
        @test _dot_eligible(plan)
        @test C_dot ≈ C_nest
        Cv0, Av0, iA, Bv0, iB, iC = mk()
        @test C_dot ≈ _hp_ref(Array(Av0), iA, Array(Bv0), iB, Array(Cv0), iC, alpha, beta)
    end
    # Scalar output C[] = sum_k a[k] b[k], K not a multiple of W.
    K = 5W + 3
    mk0 = () -> begin
        rng = MersenneTwister(99)
        (StridedView(fill(randn(rng, T))), StridedView(randn(rng, T, K)), (1,), StridedView(randn(rng, T, K)), (1,), ())
    end
    C_nest, _ = _dot_run(mk0, :never, 1.5, -0.5)
    C_dot, plan = _dot_run(mk0, :auto, 1.5, -0.5)
    @test _dot_eligible(plan)
    @test C_dot ≈ C_nest
    Cv0, Av0, _, Bv0, _, _ = mk0()
    @test C_dot[] ≈ 1.5 * sum(Array(Av0) .* Array(Bv0)) - 0.5 * Cv0[]
end

@testset "dot path: conjugation flags and views ($T)" for T in (ComplexF64, ComplexF32)
    alpha, beta = T(0.75, -1.0), T(0.5, 0.5)
    for conjA in (false, true), conjB in (false, true), wrapA in (false, true), wrapB in (false, true)
        mk = _dot_gemv_maker(T, 4, 5)
        run(mode) = let (Cv, Av, iA, Bv, iB, iC) = mk()
            _dot_mode[] = mode
            p = plan_contract(Cv, wrapA ? conj(Av) : Av, iA, wrapB ? conj(Bv) : Bv, iC === () ? iB : iB, iC; conjA = conjA, conjB = conjB)
            @test mode === :never || _dot_eligible(p)
            execute!(p, alpha, beta)
            _dot_mode[] = :auto
            Array(Cv)
        end
        Cn = run(:never)
        Cd = run(:auto)
        @test Cd ≈ Cn
        Cv0, Av0, iA, Bv0, iB, iC = mk()
        @test Cd ≈ _hp_ref(
            Array(Av0), iA, Array(Bv0), iB, Array(Cv0), iC, alpha, beta;
            conjA = conjA ⊻ wrapA, conjB = conjB ⊻ wrapB
        )
    end
    # N = 1 with a conjugated MATRIX operand (A): the conj is folded into the
    # vector and an output conj.
    for conjA in (false, true), conjB in (false, true)
        mk = _dot_n1_maker(T, 3, 20, 8)
        Cn, _ = _dot_run(mk, :never, alpha, beta; conjA = conjA, conjB = conjB)
        Cd, plan = _dot_run(mk, :auto, alpha, beta; conjA = conjA, conjB = conjB)
        @test _dot_eligible(plan)
        @test Cd ≈ Cn
        Cv0, Av0, iA, Bv0, iB, iC = mk()
        @test Cd ≈ _hp_ref(Array(Av0), iA, Array(Bv0), iB, Array(Cv0), iC, alpha, beta; conjA = conjA, conjB = conjB)
    end
end

@testset "dot path: beta = 0 never reads C ($T)" for T in _DOT_TYPES
    mk = _dot_gemv_maker(T, 4, 6; Cfill = NaN)
    C_dot, plan = _dot_run(mk, :auto, 2.0, 0.0)
    @test _dot_eligible(plan)
    @test all(isfinite, C_dot)
    Cv0, Av0, iA, Bv0, iB, iC = mk()
    @test C_dot ≈ _hp_ref(Array(Av0), iA, Array(Bv0), iB, zeros(T, size(Cv0)), iC, 2.0, 0.0)
end

@testset "dot path: out-of-bounds operands are rejected before any write ($T)" for T in (Float64, ComplexF64)
    K, N = 24, 5
    # Matrix (B) one element short, vector one short, C one short.
    mkB = () -> (
        StridedView(zeros(T, 1, N)), StridedView(randn(T, 1, K)), (1, 2),
        StridedView(randn(T, K * N - 1), (K, N), (1, K), 0), (2, 3), (1, 3),
    )
    mkA = () -> (
        StridedView(zeros(T, 1, N)), StridedView(randn(T, K - 1), (1, K), (1, 1), 0), (1, 2),
        StridedView(randn(T, K, N)), (2, 3), (1, 3),
    )
    mkC = () -> (
        StridedView(zeros(T, N - 1), (1, N), (1, 1), 0), StridedView(randn(T, 1, K)), (1, 2),
        StridedView(randn(T, K, N)), (2, 3), (1, 3),
    )
    for mk in (mkB, mkA, mkC)
        Cv, Av, iA, Bv, iB, iC = mk()
        plan = plan_contract(Cv, Av, iA, Bv, iB, iC)
        @test _dot_eligible(plan)
        @test_throws BoundsError execute!(plan, 1.0, 0.0)
        @test all(iszero, plan.Cstorage)
    end
end

@testset "dot path: allocation-free in steady state ($T)" for T in _DOT_TYPES
    Cv, Av, iA, Bv, iB, iC = _dot_gemv_maker(T, 6, 4)()
    plan = plan_contract(Cv, Av, iA, Bv, iB, iC; oracle = false)
    @test _dot_eligible(plan)
    execute!(plan, 1.0, 0.0)
    execute!(plan, 1.0, 0.0)
    @test (@allocated execute!(plan, 1.0, 0.0)) == 0 skip = (VERSION < v"1.11")
    # Several K blocks and output blocks too.
    Cv, Av, iA, Bv, iB, iC = _dot_n1_maker(T, 5, 300, 4)()
    plan = plan_contract(Cv, Av, iA, Bv, iB, iC; oracle = false, kc = 64, mc = 8)
    @test _dot_eligible(plan)
    execute!(plan, 1.0, 0.0)
    @test (@allocated execute!(plan, 1.0, 0.0)) == 0 skip = (VERSION < v"1.11")
end

@testset "dot path: through the TensorOperations backend ($T)" for T in _DOT_TYPES
    d = 6
    A = randn(T, d, d); B = randn(T, d, d, d, d, d)
    C1 = zeros(T, d, d, d); C2 = zeros(T, d, d, d)
    pA, pB, pAB = TO.contract_indices((:a, :b), (:a, :b, :c, :d, :e), (:c, :d, :e))
    _dot_mode[] = :never
    TO.tensorcontract!(C1, A, pA, false, B, pB, false, pAB, 1.0, 0.0, QuasiStridedBackend())
    _dot_mode[] = :auto
    TO.tensorcontract!(C2, A, pA, false, B, pB, false, pAB, 1.0, 0.0, QuasiStridedBackend())
    @test C2 ≈ C1
    @test vec(C2) ≈ transpose(reshape(B, d * d, d^3)) * vec(A)
end
