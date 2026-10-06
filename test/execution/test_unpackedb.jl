# The unpacked-B path: `execute!` reads B in place through an `UnpackedBView`
# where `unpacked_b_rule` says so. The result must be bitwise the packed nest's
# (same arithmetic, same values, different address), and the packed-B buffer
# must stay untouched.

include("helpers.jl")

# Poison the packed-B buffer, then run.
function _ub_run!(plan, alpha, beta)
    fill!(plan.workspace.packed_b, NaN)
    return execute!(plan, alpha, beta)
end
_ub_untouched(plan) = all(isnan, plan.workspace.packed_b)

# Checks the automatic unpacked run of `mk` against the packed nest (bitwise)
# and the reference.
function _ub_check(mk, alpha, beta; conjA = false, conjB = false, plankw...)
    C_unp, plan = _run_fresh(_ub_run!, mk, alpha, beta; conjA, conjB, plankw...)
    C_packed, _ = _run_nest(mk, alpha, beta; conjA, conjB, plankw...)
    @test _ub_takes(plan)
    @test _ub_untouched(plan)
    @test C_unp == C_packed
    return @test C_unp ≈ _ref_of(mk, alpha, beta; conjA, conjB)
end

const _UB_TYPES = (Float64, Float32, ComplexF64, ComplexF32)

@testset "unpacked B: which plans read B in place ($T)" for T in _UB_TYPES
    planof(M; B = :dense, kw...) = plan_contract(_mm_maker(T, M, 12, 9, 1; B)()...; kw...)
    @test _ub_takes(planof(20)) && _ub_takes(planof(20; B = :reversed))
    # K strided in B stays packed, as does an M above the cutoff.
    @test !_ub_takes(planof(20; B = :transposed)) && !_ub_takes(planof(20; B = :gap))
    MMAX = QuasiStrided._UNPACKED_B_MMAX
    @test _ub_takes(planof(MMAX)) && !_ub_takes(planof(MMAX + 1))
    # K = 1 (a rank-0 K group, step 0).
    Kv = (StridedView(zeros(T, 20, 9)), StridedView(randn(T, 20)), (1,), StridedView(randn(T, 9)), (2,), (1, 2))
    @test !_ub_takes(plan_contract(Kv...))
    # A non-ramp K composite: B[k1,k2,n] against A[m,k2,k1].
    pk = plan_contract(
        StridedView(zeros(T, 7, 5)), StridedView(randn(T, 7, 3, 3)), (1, 3, 2),
        StridedView(randn(T, 3, 3, 5)), (2, 3, 4), (1, 4)
    )
    @test !first(QuasiStrided.affine_ramp(pk.kgroup)) && !_ub_takes(pk)
    # Kernels that address the packed panel directly.
    k = T <: Real ? ScalarKernel(Val(4), Val(3), T) : QuasiStrided.OneMKernel(Val(4), Val(6), T, Val(4))
    @test !_ub_takes(planof(20; kernel = k))
end

@testset "unpacked B: bitwise the packed nest ($T)" for T in _UB_TYPES
    for (idx, (M, K, N)) in enumerate(((2, 3, 1), (17, 9, 13), (33, 64, 5))), B in (:dense, :reversed),
            (alpha, beta) in ((1.0, 0.0), (2.5, -0.75), (1.0, 1.0))
        _ub_check(_mm_maker(T, M, K, N, 10 + idx; B), alpha, beta)
    end
    # Several N/K/M blocks with tails, explicit kernel shapes.
    k0 = plan_contract(_mm_maker(T, 64, 8, 8, 2)()...).kernel
    W = lanewidth(k0)
    kernels = T <: Real ? (k0, SIMDKernel(Val(8), Val(6), T)) :
        (k0, QuasiStrided.PlanarKernel(Val(W), Val(5), T, Val(W)), QuasiStrided.FMAddSubKernel(Val(W), Val(5), T, Val(W)))
    for k in kernels, (m_block, k_block, n_block) in ((8, 3, 6), (40, 1, 100))
        _ub_check(_mm_maker(T, 2 * tile_size(k, 1) + 1, 13, 2 * tile_size(k, 2) + 1, 3), 1.5, 0.5; kernel = k, m_block, k_block, n_block)
    end
    # beta = 0 never reads a NaN C.
    _ub_check(_mm_maker(T, 10, 6, 7, 3; Cfill = NaN), 2.0, 0.0)
    # A non-ramp N composite (C reversed along n2): irregular column offsets.
    mkN = () -> (
        StridedView(view(randn(MersenneTwister(77), T, 20, 13, 3), :, :, 3:-1:1)),
        StridedView(randn(MersenneTwister(78), T, 20, 6)), (1, 2),
        StridedView(randn(MersenneTwister(79), T, 6, 13, 3)), (2, 3, 4), (1, 3, 4),
    )
    @test !first(QuasiStrided.affine_ramp(plan_contract(mkN()...).ngroup))
    _ub_check(mkN, 0.75, 0.5)
    # Permuted A, negative-stride B, offset C.
    _ub_check(() -> scattered_fixture(T), 0.75, 0.5)
    if T <: Complex
        for conjA in (false, true), conjB in (false, true)
            _ub_check(_mm_maker(T, 9, 7, 8, 91), T(0.75, -1.0), T(0.5, 0.5); conjA, conjB)
        end
        # A conj-wrapped B composes with the flag by XOR.
        mkW = () -> (f = _mm_maker(T, 9, 7, 8, 91)(); (f[1:3]..., conj(f[4]), f[5:6]...))
        C, _ = _run_fresh(execute!, mkW, 1.0, 0.0; conjB = true)
        @test C ≈ _ref_of(_mm_maker(T, 9, 7, 8, 91), 1.0, 0.0)
    end
end

@testset "unpacked B: out-of-bounds B is rejected before any write ($T)" for T in (Float64, ComplexF64)
    Cm = zeros(T, 10, 5)
    Bshort = StridedView(randn(T, 6 * 5 - 1), (6, 5), (1, 6), 0)
    plan = plan_contract(StridedView(Cm), StridedView(randn(T, 10, 6)), (1, 2), Bshort, (2, 3), (1, 3))
    @test _ub_takes(plan)
    @test_throws BoundsError execute!(plan, 1.0, 0.0)
    @test all(iszero, Cm)
end

@testset "unpacked B: allocation-free, and through the backend ($T)" for T in _UB_TYPES
    Amat, Bmat, Cmat = randn(T, 40, 50), randn(T, 50, 37), zeros(T, 40, 37)
    plan = _mm_plan(Cmat, Amat, Bmat; k_block = 16, n_block = 12)
    @test _ub_takes(plan)
    allocs = _steady_allocs!(execute!, plan, Cmat)
    @test allocs == 0 skip = (VERSION < v"1.11")
    @test Cmat ≈ Amat * Bmat
    # The backend plans and runs in one call.
    fill!(Cmat, 0)
    TO.tensorcontract!(Cmat, Amat, ((1,), (2,)), false, Bmat, ((1,), (2,)), false, ((1, 2), ()), 1.0, 0.0, QuasiStridedBackend())
    @test Cmat ≈ Amat * Bmat
end
