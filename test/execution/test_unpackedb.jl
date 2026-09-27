# The unpacked-B path (src/execution/unpackedb.jl): `execute!` runs the
# microkernel against B read in place through an `UnpackedBView` instead of a
# packed panel. Three properties are checked:
#
#   1. `_use_unpacked_b` selects exactly the plans it should: an eligible
#      kernel (SIMD/planar/fmaddsub, never 1m or scalar), `Qm <=
#      _UNPACKED_B_MMAX`, and a K composite that is an affine ramp with B step
#      +-1; and `_UNPACKED_B_MODE` overrides it both ways.
#   2. The result is BITWISE identical (`==`) to the packed path: the K-step
#      arithmetic is the same `_accumulate_step`, in the same order, on the
#      same B values -- only the address they are read from differs. Padding
#      columns alias a valid column instead of holding zeros, and are never
#      stored. Also `≈` the oracle and a brute-force reference.
#   3. Under `:always` the packed-B buffer is never written (NaN poison), and
#      the call stays allocation-free.
#
# `_hp_ref` (test/execution/test_halfpack.jl, included before this file) is the
# brute-force reference.

const _ub_mode = QuasiStrided._UNPACKED_B_MODE
const _ub_use = QuasiStrided._use_unpacked_b

# Run `execute!` under `mode` on a fresh fixture from `mk()`; returns the
# output array and whether the packed-B buffer was left untouched.
function _ub_run(mk, mode::Symbol, alpha, beta; plankw...)
    Cv, Av, indA, Bv, indB, indC = mk()
    plan = plan_contract(Cv, Av, indA, Bv, indB, indC; plankw...)
    fill!(plan.workspace.packed_b, NaN)
    old = _ub_mode[]
    _ub_mode[] = mode
    try
        ret = execute!(plan, alpha, beta)
        @test ret === plan.Cstorage
    finally
        _ub_mode[] = old
    end
    return Array(Cv), all(isnan, plan.workspace.packed_b), plan
end

# Matmul fixture with a choice of B layout; NaN gaps poison C if read.
#   :kunit    B[k,n] dense                      (K step +1)
#   :kneg     B reversed along k                (K step -1)
#   :nunit    B stored [n,k], viewed transposed (K step N)
#   :kgap     every other k of a 2K-row B       (K step 2)
function _ub_mm_maker(::Type{T}, Ma, Ka, Na, seed; blayout = :kunit, Cfill = nothing) where {T}
    return function ()
        rng = MersenneTwister(seed)
        Amat = randn(rng, T, Ma, Ka)
        Bmat = randn(rng, T, Ka, Na)
        Cmat = Cfill === nothing ? randn(rng, T, Ma, Na) : fill(convert(T, Cfill), Ma, Na)
        Bv = if blayout === :kunit
            StridedView(Bmat)
        elseif blayout === :kneg
            big = fill(convert(T, NaN), Ka + 4, Na)
            big[(Ka + 2):-1:3, :] .= Bmat
            StridedView(view(big, (Ka + 2):-1:3, :))
        elseif blayout === :nunit
            permutedims(StridedView(permutedims(Bmat, (2, 1))), (2, 1))
        elseif blayout === :kgap
            big = fill(convert(T, NaN), 2Ka, Na)
            big[1:2:(2Ka), :] .= Bmat
            StridedView(view(big, 1:2:(2Ka), :))
        else
            error("unknown blayout $blayout")
        end
        @assert Array(Bv) == Bmat
        return (StridedView(Cmat), StridedView(Amat), (1, 2), Bv, (2, 3), (1, 3))
    end
end

const _UB_TYPES = (Float64, Float32, ComplexF64, ComplexF32)
const _UB_AB = ((1.0, 0.0), (2.5, -0.75), (1.0, 1.0), (-0.5, 1.25))

@testset "unpacked B: which plans read B in place ($T)" for T in _UB_TYPES
    mk(blayout, M) = _ub_mm_maker(T, M, 12, 9, 1; blayout = blayout)
    for blayout in (:kunit, :kneg)
        p = plan_contract(mk(blayout, 20)()...)
        @test _ub_use(p)
        @test QuasiStrided._unpacked_b_kernel_eligible(p.kernel)
        # The mode override wins both ways.
        _ub_mode[] = :never
        @test !_ub_use(p)
        _ub_mode[] = :always
        @test _ub_use(p)
        _ub_mode[] = :auto
    end
    # A strided or gapped K in B stays packed ...
    for blayout in (:nunit, :kgap)
        @test !_ub_use(plan_contract(mk(blayout, 20)()...))
    end
    # ... as does a K = 1 outer product (rank-0 K group, step 0) ...
    p = plan_contract(
        StridedView(zeros(T, 20, 9)), StridedView(randn(T, 20)), (1,), StridedView(randn(T, 9)), (2,), (1, 2)
    )
    @test !_ub_use(p)
    # ... and an M extent above the cutoff.
    MMAX = QuasiStrided._UNPACKED_B_MMAX
    @test _ub_use(plan_contract(mk(:kunit, MMAX)()...))
    @test !_ub_use(plan_contract(mk(:kunit, MMAX + 1)()...))
    # Kernels that address the panel directly never take the view.
    if T <: Real
        ps = plan_contract(mk(:kunit, 20)()...; kernel = ScalarKernel(Val(4), Val(3), T))
        @test !_ub_use(ps)
    else
        po = plan_contract(mk(:kunit, 20)()...; kernel = QuasiStrided.OneMKernel(Val(4), Val(6), T, Val(4)))
        @test !_ub_use(po)
    end
    # A scattered (non-ramp) K composite is ineligible: B[k1,k2,n] against
    # A[m,k2,k1] -- both K maps ramps on their own, not together.
    d = 3
    Aarr = randn(T, 7, d, d)
    Barr = randn(T, d, d, 5)
    Cm = zeros(T, 7, 5)
    pk = plan_contract(StridedView(Cm), StridedView(Aarr), (1, 3, 2), StridedView(Barr), (2, 3, 4), (1, 4))
    @test !first(QuasiStrided.affine_ramp(pk.kgroup))
    @test !_ub_use(pk)
end

@testset "unpacked B: bitwise the packed path on matmul shapes ($T)" for T in _UB_TYPES
    shapes = ((1, 1, 1), (5, 3, 7), (16, 16, 16), (17, 9, 13), (40, 7, 30), (33, 64, 5), (2, 100, 1))
    for (idx, (M, K, N)) in enumerate(shapes), blayout in (:kunit, :kneg), (alpha, beta) in _UB_AB
        mk = _ub_mm_maker(T, M, K, N, 10 + idx; blayout = blayout)
        C_packed, _, _ = _ub_run(mk, :never, alpha, beta)
        C_unp, untouched, plan = _ub_run(mk, :always, alpha, beta)
        @test untouched
        @test C_unp == C_packed
        Cv0, Av0, _, Bv0, _, _ = mk()
        C_tw = let (Cv, Av, iA, Bv, iB, iC) = mk()
            execute_tilewise!(plan_contract(Cv, Av, iA, Bv, iB, iC), alpha, beta)
            Array(Cv)
        end
        @test C_unp ≈ C_tw
        @test C_unp ≈ _hp_ref(Array(Av0), (1, 2), Array(Bv0), (2, 3), Array(Cv0), (1, 3), alpha, beta)
        # The automatic rule picks the view for every one of these but the
        # K = 1 shape (a rank-0 K group has step 0, not +-1).
        @test _ub_use(plan) == (K > 1)
    end
end

@testset "unpacked B: several jc/pc/ic blocks, tails and explicit kernels ($T)" for T in _UB_TYPES
    p0 = plan_contract(_ub_mm_maker(T, 64, 8, 8, 2)()...)
    kernels = if T <: Real
        (p0.kernel, SIMDKernel(Val(8), Val(6), T), SIMDKernel(Val(2 * lanewidth(p0.kernel)), Val(3), T))
    else
        W = QuasiStrided.lanewidth(p0.kernel)
        (p0.kernel, QuasiStrided.PlanarKernel(Val(W), Val(5), T, Val(W)), QuasiStrided.FMAddSubKernel(Val(W), Val(5), T, Val(W)))
    end
    for k in kernels, (mc, kc, nc) in ((8, 3, 6), (mr(k), 5, nr(k)), (3 * mr(k), 64, 1), (40, 1, 100))
        for (M, K, N) in ((11, 17, 19), (4, 25, 3), (1, 3, 1), (2 * mr(k) + 1, 13, 2 * nr(k) + 1))
            mk = _ub_mm_maker(T, M, K, N, 7 * K + N + M)
            C_packed, _, _ = _ub_run(mk, :never, 1.5, 0.5; kernel = k, mc = mc, kc = kc, nc = nc)
            C_unp, untouched, _ = _ub_run(mk, :always, 1.5, 0.5; kernel = k, mc = mc, kc = kc, nc = nc)
            @test untouched
            @test C_unp == C_packed
            Cv0, Av0, _, Bv0, _, _ = mk()
            @test C_unp ≈ _hp_ref(Array(Av0), (1, 2), Array(Bv0), (2, 3), Array(Cv0), (1, 3), 1.5, 0.5)
        end
    end
end

@testset "unpacked B: strided/gapped K, scattered N, permuted C, conj ($T)" for T in _UB_TYPES
    alpha, beta = T <: Complex ? (T(0.75, -1.0), T(0.5, 0.5)) : (T(0.75), T(0.5))
    # (1) K strided in B: ineligible by the rule, but must be CORRECT when
    # forced -- the view works for any K axis.
    for blayout in (:nunit, :kgap)
        mk = _ub_mm_maker(T, 21, 14, 11, 5; blayout = blayout)
        C_packed, _, _ = _ub_run(mk, :never, alpha, beta)
        C_unp, untouched, _ = _ub_run(mk, :always, alpha, beta)
        @test untouched && C_unp == C_packed
    end
    # (2) Non-ramp N composite: C[m,n1,n2] = A[m,k] B[k,n1,n2] with C reversed
    # along n2, so N slivers straddle irregular offsets (the buffer path).
    M, K, N1, N2 = 20, 6, 13, 3
    mk2 = function ()
        rng = MersenneTwister(77)
        Amat = randn(rng, T, M, K)
        Barr = randn(rng, T, K, N1, N2)
        Cfull = randn(rng, T, M, N1, N2)
        Cr = view(Cfull, :, :, N2:-1:1)
        return (StridedView(Cr), StridedView(Amat), (1, 2), StridedView(Barr), (2, 3, 4), (1, 3, 4))
    end
    C_packed, _, _ = _ub_run(mk2, :never, alpha, beta)
    C_unp, untouched, plan = _ub_run(mk2, :always, alpha, beta)
    @test !first(QuasiStrided.affine_ramp(plan.ngroup))
    @test _ub_use(plan)
    @test untouched && C_unp == C_packed
    Cv0, Av0, _, Bv0, _, _ = mk2()
    @test C_unp ≈ _hp_ref(Array(Av0), (1, 2), Array(Bv0), (2, 3, 4), Array(Cv0), (1, 3, 4), alpha, beta)
    # (3) Scattered K composite forced on: correct through the PtrScatterAxis
    # K axis.
    d = 3
    mk3 = function ()
        rng = MersenneTwister(78)
        Aarr = randn(rng, T, 7, d, d)
        Barr = randn(rng, T, d, d, 5)
        Cm = randn(rng, T, 7, 5)
        return (StridedView(Cm), StridedView(Aarr), (1, 3, 2), StridedView(Barr), (2, 3, 4), (1, 4))
    end
    C_packed, _, _ = _ub_run(mk3, :never, alpha, beta)
    C_unp, untouched, _ = _ub_run(mk3, :always, alpha, beta)
    @test untouched && C_unp == C_packed
    Cv0, Av0, _, Bv0, _, _ = mk3()
    @test C_unp ≈ _hp_ref(Array(Av0), (1, 3, 2), Array(Bv0), (2, 3, 4), Array(Cv0), (1, 4), alpha, beta)
    # (4) Permuted A / negative-stride B / offset C (the scattered fixture).
    C_packed = let (Cv, Av, iA, Bv, iB, iC) = scattered_fixture(T)
        _ub_mode[] = :never
        execute!(plan_contract(Cv, Av, iA, Bv, iB, iC), alpha, beta)
        _ub_mode[] = :auto
        Array(Cv)
    end
    C_unp = let (Cv, Av, iA, Bv, iB, iC) = scattered_fixture(T)
        _ub_mode[] = :always
        execute!(plan_contract(Cv, Av, iA, Bv, iB, iC), alpha, beta)
        _ub_mode[] = :auto
        Array(Cv)
    end
    @test C_unp == C_packed
    # (5) Conjugation: the flag, a conj-wrapped view, and both (which cancel).
    if T <: Complex
        for conjA in (false, true), conjB in (false, true), wrap in (false, true)
            mk = _ub_mm_maker(T, 9, 7, 8, 91)
            run(mode) = let (Cv, Av, iA, Bv, iB, iC) = mk()
                _ub_mode[] = mode
                p = plan_contract(Cv, Av, iA, wrap ? conj(Bv) : Bv, iB, iC; conjA = conjA, conjB = conjB)
                execute!(p, alpha, beta)
                _ub_mode[] = :auto
                Array(Cv)
            end
            Cn = run(:never)
            Ca = run(:always)
            @test Ca == Cn
            Cv0, Av0, _, Bv0, _, _ = mk()
            @test Ca ≈ _hp_ref(
                Array(Av0), (1, 2), Array(Bv0), (2, 3), Array(Cv0), (1, 3), alpha, beta;
                conjA = conjA, conjB = conjB ⊻ wrap
            )
        end
    end
end

@testset "unpacked B: beta = 0 never reads C, K = 0 and alpha = 0 untouched ($T)" for T in _UB_TYPES
    mk = _ub_mm_maker(T, 10, 6, 7, 3; Cfill = NaN)
    C_unp, untouched, _ = _ub_run(mk, :always, 2.0, 0.0)
    @test untouched && all(isfinite, C_unp)
    Cv0, Av0, _, Bv0, _, _ = mk()
    @test C_unp ≈ 2.0 .* (Array(Av0) * Array(Bv0))
    # alpha = 0 and K = 0 take the beta-only short circuit before the choice.
    Cv, Av, iA, Bv, iB, iC = _ub_mm_maker(T, 10, 6, 7, 3)()
    C0 = Array(Cv)
    _ub_mode[] = :always
    execute!(plan_contract(Cv, Av, iA, Bv, iB, iC), 0.0, 0.5)
    _ub_mode[] = :auto
    @test Array(Cv) ≈ 0.5 .* C0
end

@testset "unpacked B: out-of-bounds B is still rejected before any write ($T)" for T in (Float64, ComplexF64)
    M, K, N = 10, 6, 5
    Cm = zeros(T, M, N)
    Bshort = StridedView(randn(T, K * N - 1), (K, N), (1, K), 0)   # one element short
    plan = plan_contract(StridedView(Cm), StridedView(randn(T, M, K)), (1, 2), Bshort, (2, 3), (1, 3))
    @test _ub_use(plan)
    _ub_mode[] = :always
    @test_throws BoundsError execute!(plan, 1.0, 0.0)
    _ub_mode[] = :auto
    @test all(iszero, Cm)
end

@testset "unpacked B: allocation-free in steady state ($T)" for T in _UB_TYPES
    Cmat = zeros(T, 40, 37)
    plan = _mm_plan(Cmat, randn(T, 40, 50), randn(T, 50, 37); kc = 16, nc = 12)
    @test _ub_use(plan)
    _ub_mode[] = :always
    @test _steady_allocs!(execute!, plan, Cmat) == 0 skip = (VERSION < v"1.11")
    _ub_mode[] = :auto
    @test Cmat ≈ reshape(plan.Astorage, 40, 50) * reshape(plan.Bstorage, 50, 37)
end

@testset "unpacked B: the TensorOperations backend result is unchanged ($T)" for T in _UB_TYPES
    A = randn(T, 12, 9); B = randn(T, 9, 14)
    C1 = zeros(T, 12, 14); C2 = zeros(T, 12, 14)
    _ub_mode[] = :never
    TO.tensorcontract!(C1, A, ((1,), (2,)), false, B, ((1,), (2,)), false, ((1, 2), ()), 1.0, 0.0, QuasiStridedBackend())
    _ub_mode[] = :auto
    TO.tensorcontract!(C2, A, ((1,), (2,)), false, B, ((1,), (2,)), false, ((1, 2), ()), 1.0, 0.0, QuasiStridedBackend())
    @test C2 == C1
    @test C2 ≈ A * B
end
