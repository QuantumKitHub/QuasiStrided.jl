# The dot-product path: `execute!` on `M == 1` or `N == 1` with a K-contiguous
# matrix operand runs a K-vectorized gemv. Results match the nest and the
# reference to a tolerance (the K summation order differs).

include("helpers.jl")

const _DOT_TYPES = (Float64, Float32, ComplexF64, ComplexF32)

# `C[cde] = A[ab] * B[abcde]` (M = 1, matrix B K-fastest) at leg dimension `d`:
#   :permvec  A stored [b,a]: the vector's K map is not a ramp (the matrix's is)
#   :permC    C stored [e,d,c]: scattered output offsets
#   :offsetB  B a view with a nonzero offset
function _dot_gemv_maker(::Type{T}, d, seed; variant = :plain, Cfill = nothing) where {T}
    return function ()
        rng = MersenneTwister(seed)
        A, B = randn(rng, T, d, d), randn(rng, T, d, d, d, d, d)
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
        return (Cv, Av, (1, 2), Bv, (1, 2, 3, 4, 5), (3, 4, 5))
    end
end

# N = 1 with the matrix operand A K-fastest: C[a,b] = A[k,a,b] * v[k].
_dot_n1_maker(::Type{T}, d, K, seed) where {T} = function ()
    rng = MersenneTwister(seed)
    return (StridedView(randn(rng, T, d, d)), StridedView(randn(rng, T, K, d, d)), (3, 1, 2), StridedView(randn(rng, T, K)), (3,), (1, 2))
end

# The dot path against the nest and the reference.
function _dot_check(mk, alpha, beta; conjA = false, conjB = false, plankw...)
    C_dot, plan = _run_fresh(execute!, mk, alpha, beta; conjA, conjB, plankw...)
    C_nest, _ = _run_nest(mk, alpha, beta; conjA, conjB, plankw...)
    @test _dot_takes(plan)
    @test C_dot ≈ C_nest
    return @test C_dot ≈ _ref_of(mk, alpha, beta; conjA, conjB)
end

@testset "dot path: which plans it takes ($T)" for T in _DOT_TYPES
    W = _lanes(T)
    for variant in (:plain, :permvec, :permC, :offsetB)
        @test _dot_takes(plan_contract(_dot_gemv_maker(T, 4, 1; variant)()...))
    end
    @test _dot_takes(plan_contract(_dot_n1_maker(T, 4, 16, 2)()...))
    # The matrix operand A K-slowest; K below one vector; no degenerate extent;
    # non-DenseVector matrix storage.
    declined = (
        (StridedView(zeros(T, 4, 4)), StridedView(randn(T, 4, 4, 16)), (1, 2, 3), StridedView(randn(T, 16)), (3,), (1, 2)),
        _mm_maker(T, 1, W - 1, 9, 3)(), _mm_maker(T, 3, 20, 5, 4)(), _mm_maker(T, 1, 4W, 5, 5; B = :wrapped)(),
    )
    for f in declined
        @test !_dot_takes(plan_contract(f...))
    end
end

@testset "dot path: matches the nest and the reference ($T)" for T in _DOT_TYPES
    ab = ((1.0, 0.0), (2.5, -0.75), (1.0, 1.0))
    for d in (2, 6), variant in (:plain, :permvec, :permC, :offsetB), (alpha, beta) in ab
        mk = _dot_gemv_maker(T, d, 10 * d; variant)
        if d * d >= _lanes(T)
            _dot_check(mk, alpha, beta)
        else
            @test !_dot_takes(plan_contract(mk()...))
        end
    end
    # N = 1: K tails, several K blocks (`k_block = 64`) and output blocks (`m_block = 7`).
    W = _lanes(T)
    for (d, K) in ((3, W), (5, 3W - 1), (2, 1000)), (alpha, beta) in ab
        _dot_check(_dot_n1_maker(T, d, K, 3K + d), alpha, beta; k_block = 64, m_block = 7)
    end
    # A scalar output, C[] = sum_k a[k] b[k].
    K = 5W + 3
    mk0 = () -> (rng = MersenneTwister(99); (StridedView(fill(randn(rng, T))), StridedView(randn(rng, T, K)), (1,), StridedView(randn(rng, T, K)), (1,), ()))
    _dot_check(mk0, 1.5, -0.5)
    # beta = 0 never reads a NaN C.
    C, _ = _run_fresh(execute!, _dot_gemv_maker(T, 4, 6; Cfill = NaN), 2.0, 0.0)
    @test all(isfinite, C)
    if T <: Complex
        # The flags and conj views, on the matrix (B) and on the vector.
        for conjA in (false, true), conjB in (false, true), wrap in (false, true)
            mk = _dot_gemv_maker(T, 4, 5)
            mkw = wrap ? () -> (f = mk(); (f[1], conj(f[2]), f[3], conj(f[4]), f[5], f[6])) : mk
            _dot_check(mkw, T(0.75, -1.0), T(0.5, 0.5); conjA, conjB)
        end
        # N = 1 with a conjugated matrix operand (A).
        for conjA in (false, true), conjB in (false, true)
            _dot_check(_dot_n1_maker(T, 3, 20, 8), T(0.75, -1.0), T(0.5, 0.5); conjA, conjB)
        end
    end
end

@testset "dot path: out-of-bounds operands are rejected before any write ($T)" for T in (Float64, ComplexF64)
    K, N = 24, 5
    # Matrix (B) one element short, vector one short, C one short.
    for (sa, sb, sc) in ((K, K * N - 1, N), (K - 1, K * N, N), (K, K * N, N - 1))
        plan = plan_contract(
            StridedView(zeros(T, sc), (1, N), (1, 1), 0), StridedView(randn(T, sa), (1, K), (1, 1), 0), (1, 2),
            StridedView(randn(T, sb), (K, N), (1, K), 0), (2, 3), (1, 3)
        )
        @test _dot_takes(plan)
        @test_throws BoundsError execute!(plan, 1.0, 0.0)
        @test all(iszero, plan.Cstorage)
    end
end

@testset "dot path: allocation-free, and through the backend ($T)" for T in _DOT_TYPES
    for (mk, kw) in ((_dot_gemv_maker(T, 6, 4), (;)), (_dot_n1_maker(T, 5, 300, 4), (k_block = 64, m_block = 8)))
        plan = plan_contract(mk()...; kw...)
        @test _dot_takes(plan)
        execute!(plan, 1.0, 0.0)
        @test (@allocated execute!(plan, 1.0, 0.0)) == 0 skip = (VERSION < v"1.11")
    end
    # The backend plans and runs in one call.
    d = 6
    A, B, C = randn(T, d, d), randn(T, d, d, d, d, d), zeros(T, d, d, d)
    pA, pB, pAB = TO.contract_indices((:a, :b), (:a, :b, :c, :d, :e), (:c, :d, :e))
    TO.tensorcontract!(C, A, pA, false, B, pB, false, pAB, 1.0, 0.0, QuasiStridedBackend())
    @test vec(C) ≈ transpose(reshape(B, d * d, d^3)) * vec(A)
end
