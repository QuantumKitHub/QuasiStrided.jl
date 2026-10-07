include("helpers.jl")

@testset "classify_labels" begin
    for (indA, indB, indC, want) in (
            ((1, 2), (2, 3), (1, 3), ((1,), (3,), (2,))),                          # GEMM
            ((1, 2), (2, 3), (3, 1), ((1,), (3,), (2,))),                          # transposed C
            ((1, 2, 3, 4), (3, 5, 6, 7), (4, 6, 7, 1, 2, 5), ((1, 2, 4), (5, 6, 7), (3,))),
            ((1, 2), (3, 1, 4, 5), (3, 2, 4, 5), ((2,), (3, 4, 5), (1,))),
            ((1, 2, 3, 4), (3, 4, 5, 6), (1, 2, 5, 6), ((1, 2), (5, 6), (3, 4))),
            ((1,), (1, 2), (2,), ((), (2,), (1,))),                                # no M
            ((1, 2), (3, 4), (1, 2, 3, 4), ((1, 2), (3, 4), ())),                  # outer product
        )
        got = QuasiStrided.classify_labels(indA, indB, indC)
        @test got === want
        @test map(length, got) == QuasiStrided.group_ranks(length(indA), length(indB), length(indC))
    end
    @test_throws ArgumentError QuasiStrided.classify_labels((1, 1), (1, 2), (1, 2))
    @test_throws ArgumentError QuasiStrided.classify_labels((1, 2), (2, 3), (1, 2, 3))  # all three
    @test_throws ArgumentError QuasiStrided.classify_labels((1, 2), (3, 4), (1, 3))     # dangling in A
    @test_throws ArgumentError QuasiStrided.classify_labels((1, 2), (3, 4), (1, 2, 3, 9))
end

@testset "label order: order_free_labels sorts by |C-stride|, stably" begin
    Cv = _lo_view((5, 3, 2, 4), (12, 1, 60, 3))
    indC = (10, 20, 30, 40)
    labels = (10, 20, 30, 40)
    @test _lo_order(labels, indC, Cv) === (20, 40, 10, 30)
    @test _lo_order((40, 10), indC, Cv) === (40, 10)  # a subset: only its own members
    @test _lo_order((30, 20), indC, Cv) === (20, 30)

    # Ties keep input order, whichever way the input is given.
    Ct = _lo_view((2, 3, 4), (1, 1, 1))
    @test _lo_order((7, 8, 9), (7, 8, 9), Ct) === (7, 8, 9)
    @test _lo_order((9, 7, 8), (7, 8, 9), Ct) === (9, 7, 8)

    # Negative strides sort by magnitude.
    Cn = _lo_view((3, 4), (-1, 4))
    @test _lo_order((20, 10), (10, 20), Cn) === (10, 20)
    Cn2 = _lo_view((3, 4), (4, -1))
    @test _lo_order((10, 20), (10, 20), Cn2) === (20, 10)

    @test _lo_order((), indC, Cv) === ()
    @test _lo_order((30,), indC, Cv) === (30,)

    # Against a stable `sortperm`, randomized.
    Random.seed!(20260921)
    for trial in 1:100
        nd = rand(1:5)
        C = StridedView(randn(ntuple(_ -> rand(1:4), nd)))
        st = Base.strides(C)
        labels = shuffle(collect(1:nd))[1:rand(1:nd)]
        got = _lo_order(Tuple(labels), ntuple(identity, nd), C)
        @test got isa NTuple{length(labels), Int}
        @test collect(got) == labels[sortperm([abs(st[l]) for l in labels]; alg = Base.Sort.DEFAULT_STABLE)]
    end
end

@testset "label order: leading_unit_run / prefer_swap" begin
    d = 4
    C6 = StridedView(zeros(Float64, d, d, d, d, d, d))  # strides 1, d, d^2, ...
    indC = (1, 2, 3, 4, 5, 6)                            # a,b,c,i,j,k
    a, b, c, i, j, k = indC

    @test _lo_run((a, i, j), indC, C6) == d          # ccsd_t_1's sorted M: run stops at i
    @test _lo_run((a, b, k), indC, C6) == d^2        # ccsd_t_3's sorted N: b is C-adjacent to a
    @test _lo_run((a, b, c, i, j, k), indC, C6) == d^6
    @test _lo_run((b, i, j), indC, C6) == 1          # no unit-stride head
    @test _lo_run((a, c, b), indC, C6) == d          # sorted order is the caller's job
    @test _lo_run((), indC, C6) == 1

    # A descending axis is not a unit-stride run.
    @test _lo_run((10, 20), (10, 20), _lo_view((3, 4), (-1, 3))) == 1
    # Singleton axes are skipped whatever their stride; an empty axis ends it.
    @test _lo_run((10, 20), (10, 20), _lo_view((1, 6), (5, 1))) == 6
    @test _lo_run((10, 20), (10, 20), _lo_view((6, 1), (1, 17))) == 6
    @test _lo_run((10, 20), (10, 20), _lo_view((0, 6), (1, 1))) == 0

    # The swap rule on the ccsd_t shapes at dim 4, with `m_tile` played by hand.
    # ccsd_t_2: sorted M = (b,i,j) (run 1), sorted N = (a,c,k) (run 4).
    @test _lo_swap((b, i, j), (a, c, k), indC, C6, 4)        # 4-wide kernel: swap
    @test !_lo_swap((b, i, j), (a, c, k), indC, C6, 8)       # 8-wide: 4 < 8, do not swap
    # ccsd_t_3: sorted N = (a,b,k) (run 16): the adjacent label extends it.
    @test _lo_swap((c, i, j), (a, b, k), indC, C6, 8)
    @test _lo_swap((c, i, j), (a, b, k), indC, C6, 16)
    @test !_lo_swap((c, i, j), (a, b, k), indC, C6, 32)
    # ccsd_t_1: M already has the run; never swap, whatever m_tile says.
    @test !_lo_swap((a, i, j), (b, c, k), indC, C6, 4)
    @test !_lo_swap((a, i, j), (b, c, k), indC, C6, 8)
    # Two-m_tile form: each orientation is judged against the kernel it would run.
    @test _lo_swap((b, i, j), (a, c, k), indC, C6, 8, 4)
    @test !_lo_swap((b, i, j), (a, c, k), indC, C6, 8, 8)
    @test !_lo_swap((a, i, j), (b, c, k), indC, C6, 4, 4)
    # Nothing to swap onto.
    @test !_lo_swap((b, i, j), (), indC, C6, 1)
end

@testset "label order: leading_unit_run's full-coverage condition is a single-map affine_ramp on C's own strides" begin
    # The whole composite is one run iff C's own single-map group is a unit
    # ramp. Empty and all-singleton composites are excluded: `affine_ramp`
    # reports step 0 there while the run is trivially 1.
    rng = Random.MersenneTwister(0x01E3A3E1)
    ntested = 0
    while ntested < 300
        D = rand(rng, 1:5)
        indCr = ntuple(identity, D)
        lens = ntuple(_ -> rand(rng, (1, 2, 3, 5)), D)
        strides = ntuple(_ -> rand(rng, (-3, -1, 1, 2, 3, 7)), D)
        Cr = _lo_view(lens, strides)
        nlabels = rand(rng, 1:D)
        order = Tuple(Random.shuffle(rng, collect(1:D))[1:nlabels])
        m_length = prod(lens[l] for l in order)
        m_length == 1 && continue
        ntested += 1

        run = QuasiStrided.leading_unit_run(order, indCr, Cr)

        clens = ntuple(d -> lens[order[d]], nlabels)
        cstrides = ntuple(d -> strides[order[d]], nlabels)
        cgroup = QuasiStrided.AxisGroup(clens, (cstrides,))
        (isramp, steps) = QuasiStrided.affine_ramp(cgroup)

        @test (run == m_length) == (isramp && steps[1] == 1)
    end
end

@testset "label order: pinning test on the ccsd_t shapes (composite order and swap)" begin
    d = 5
    for (name, IA, IB) in LO_CASES
        (indA, indB, indC), _ = _lo_labels(IA, IB)
        A = randn(d, d, d, d)
        B = randn(d, d, d, d)
        C = zeros(d, d, d, d, d, d)
        Av, Bv, Cv = StridedView(A), StridedView(B), StridedView(C)
        cst = Base.strides(Cv)
        cstride(l) = cst[findfirst(==(l), indC)]
        mlab, nlab, klab = QuasiStrided.classify_labels(indA, indB, indC)
        @test length(klab) == 1
        kA = Base.strides(Av)[findfirst(==(klab[1]), indA)]
        kB = Base.strides(Bv)[findfirst(==(klab[1]), indB)]

        msorted = Tuple(sort(collect(mlab); by = cstride))
        nsorted = Tuple(sort(collect(nlab); by = cstride))
        @test _lo_order(mlab, indC, Cv) == msorted
        @test _lo_order(nlab, indC, Cv) == nsorted
        mrun = _lo_run(msorted, indC, Cv)
        nrun = _lo_run(nsorted, indC, Cv)
        # ccsd_t_1 carries C's unit axis on A (run d); the others carry it on B.
        @test (name == "ccsd_t_1") == (mrun == d)
        @test nrun == (name == "ccsd_t_1" ? 1 : name == "ccsd_t_3" ? d^2 : d)

        # Named kernels, host-independent: m_tile = 4 swaps 2/3/4, m_tile = 8 only 3.
        for (kernel, expect_swap) in (
                (ScalarKernel(Val(4), Val(3), Float64), name != "ccsd_t_1"),
                (SIMDKernel(Val(8), Val(6), Float64), name == "ccsd_t_3"),
            )
            plan = plan_contract(Cv, Av, indA, Bv, indB, indC; kernel = kernel)
            swapped = plan.Astorage === parent(Bv)
            @test swapped == expect_swap
            @test swapped == _lo_swap(msorted, nsorted, indC, Cv, tile_size(kernel, 1))
            if swapped
                # B feeds M: mgroup's maps are (B, C) over the sorted N labels.
                @test plan.mgroup.strides[2] == Tuple(cstride(l) for l in nsorted)
                @test plan.ngroup.strides[2] == Tuple(cstride(l) for l in msorted)
                @test plan.kgroup.strides == ((kB,), (kA,))
                @test plan.Bstorage === parent(Av)
                @test plan.Abase == offset(Bv) && plan.Bbase == offset(Av)
            else
                @test plan.mgroup.strides[2] == Tuple(cstride(l) for l in msorted)
                @test plan.ngroup.strides[2] == Tuple(cstride(l) for l in nsorted)
                @test plan.kgroup.strides == ((kA,), (kB,))
                @test plan.Bstorage === parent(Bv)
            end
            # The C map of each composite is ascending, whichever operand fed it.
            @test issorted(abs.(plan.mgroup.strides[2]))
            @test issorted(abs.(plan.ngroup.strides[2]))
            @test plan.mgroup.lengths == ntuple(_ -> d, 3)
            @test plan.ngroup.lengths == ntuple(_ -> d, 3)
        end

        # Default kernel: the same rule at this machine's `m_tile`.
        plan = plan_contract(Cv, Av, indA, Bv, indB, indC)
        m_tile = tile_size(plan.kernel, 1)
        @test (plan.Astorage === parent(Bv)) == (mrun < m_tile && nrun >= m_tile)
    end
end

@testset "label order: plain matmul is unchanged; a transposed output swaps" begin
    Ma, Ka, Na = 9, 4, 12
    Amat, Bmat = randn(Ma, Ka), randn(Ka, Na)
    kernel = SIMDKernel(Val(8), Val(6), Float64)
    # One view per operand: on Julia 1.10 `parent` of two fresh views of the
    # same array need not be `===`.
    Av, Bv = StridedView(Amat), StridedView(Bmat)

    Cmat = zeros(Ma, Na)
    p = plan_contract(StridedView(Cmat), Av, (1, 2), Bv, (2, 3), (1, 3); kernel = kernel)
    @test p.Astorage === parent(Av)
    @test p.mgroup.strides == ((1,), (1,))
    @test p.ngroup.strides == ((Ka,), (Ma,))
    execute!(p, 1.0, 0.0)
    @test Cmat ≈ Amat * Bmat

    # Into C's transpose: N's 12-wide run >= m_tile = 8, so B feeds M.
    Ct = zeros(Na, Ma)
    pt = plan_contract(StridedView(Ct), Av, (1, 2), Bv, (2, 3), (3, 1); kernel = kernel)
    @test pt.Astorage === parent(Bv)
    @test pt.mgroup.strides == ((Ka,), (1,))      # (B, C) maps over label 3
    @test pt.ngroup.strides == ((1,), (Na,))      # (A, C) maps over label 1
    @test pt.kgroup.strides == ((1,), (Ma,))      # (B, A) maps over label 2
    execute!(pt, 1.0, 0.0)
    @test Ct ≈ transpose(Amat * Bmat)
    # ... but not when the run is too short for the kernel.
    Ct2 = zeros(6, Ma)
    Bv6 = StridedView(Bmat[:, 1:6])
    pt2 = plan_contract(StridedView(Ct2), Av, (1, 2), Bv6, (2, 3), (3, 1); kernel = kernel)
    @test pt2.Astorage === parent(Av)
end

@testset "label order: correctness on ccsd_t shapes, permuted/sliced C, alpha/beta, conj" begin
    Random.seed!(0x1ABE_10DE)
    d = 6
    # C is a sliced, permuted view of a padded (c, k, a, j, b, i) array.
    perm = (3, 5, 1, 6, 4, 2)  # output axis p takes physical axis perm[p]
    for T in (Float64, ComplexF64)
        rtol = 200 * d * eps(real(T))
        for (name, IA, IB) in LO_CASES, (conjA, conjB) in ((false, false), (true, true))
            (T <: Real) && conjA && continue  # conj is the identity on the real path
            (indA, indB, indC), (pA, pB, pAB) = _lo_labels(IA, IB)
            A = randn(T, d, d, d, d)
            B = randn(T, d, d, d, d)
            Av = conjA ? StridedView(A, size(A), strides(A), 0, conj) : StridedView(A)
            Bv = StridedView(B)
            Cbig = randn(T, d + 1, d + 2, d, d + 1, d, d + 3)
            Csub = view(Cbig, 1:d, 2:(d + 1), :, 2:(d + 1), :, 3:(d + 2))
            Cv = permutedims(StridedView(Csub), perm)
            @test offset(Cv) != 0
            @test !issorted(Base.strides(Cv))
            Cstart = copy(Cv)
            alpha = T <: Complex ? T(1.3, -0.4) : T(1.3)
            beta = T <: Complex ? T(0.7, 0.2) : T(0.7)

            Cref = _lo_reference(Cstart, Av, indA, Bv, indB, indC; conjA, conjB, alpha, beta)

            plan = plan_contract(Cv, Av, indA, Bv, indB, indC; conjA = conjA, conjB = conjB)
            execute!(plan, alpha, beta)
            @test isapprox(copy(Cv), Cref; rtol = rtol)
        end
    end
end

@testset "label order: the swap never fires for complex kernels (guarded by T <: Real, deliberately deferred)" begin
    # ccsd_t_3 would swap for a real dtype at this m_tile; the transforms follow
    # the flag XOR A's `conj` op.
    d = 4
    for T in (ComplexF64, ComplexF32)
        W = QuasiStrided.default_lanewidth(real(T))
        kernel = QuasiStrided.PlanarKernel(Val(W), Val(8), T, Val(W))
        @test tile_size(kernel, 1) <= 16
        (indA, indB, indC), _ = _lo_labels(LO_CASES[3][2], LO_CASES[3][3])
        A = randn(T, d, d, d, d)
        B = randn(T, d, d, d, d)
        C = randn(T, d, d, d, d, d, d)
        Av = StridedView(A, size(A), strides(A), 0, conj)
        Bv = StridedView(B)
        Cv = StridedView(C)
        alpha, beta = T(0.5, 1.5), T(-1.0, 0.25)
        for (conjA, conjB) in ((true, true), (true, false), (false, true))
            Cstart = copy(C)
            Cref = _lo_reference(Cstart, Av, indA, Bv, indB, indC; conjA, conjB, alpha, beta)
            plan = plan_contract(Cv, Av, indA, Bv, indB, indC; kernel = kernel, conjA = conjA, conjB = conjB)
            @test plan.Astorage === parent(Av)   # the swap did NOT fire (complex)
            @test plan.atransform === (conjA ? identity : conj)
            @test plan.btransform === (conjB ? conj : identity)
            execute!(plan, alpha, beta)
            @test isapprox(C, Cref; rtol = 200 * d * eps(real(T)))
            copyto!(C, Cstart)
        end
    end
end

@testset "label order: the adapter path reaches the reordered plan" begin
    d = 6
    for (name, IA, IB) in LO_CASES
        (indA, indB, indC), (pA, pB, pAB) = _lo_labels(IA, IB)
        A, B, C = randn(d, d, d, d), randn(d, d, d, d), randn(d, d, d, d, d, d)
        Cref = _lo_reference(C, StridedView(A), indA, StridedView(B), indB, indC; alpha = 0.9, beta = -0.5)
        TO.tensorcontract!(C, A, pA, false, B, pB, false, pAB, 0.9, -0.5, QuasiStrided.QuasiStridedBackend())
        @test isapprox(C, Cref; rtol = 200 * d * eps())
    end
end

# Contracted-label order. Labels a=1 b=2 c=3 d=4 e=5 with distinct extents.
# `KO_EXT` fits any L2; `KO_BIG` makes B's lines touched before `d` advances
# (b*c*e*64 B) exceed THIS host's L2 share, with e >= 40, so the model reorders
# K wherever the suite runs.
function _ko_big(l2 = QuasiStrided.l2_core_bytes(target_profile()))
    e = max(40, fld(l2, 40 * 40 * 64) + 1)
    return Dict(1 => 9, 2 => 40, 3 => 40, 4 => 5, 5 => e)
end
const KO_EXT = Dict(1 => 9, 2 => 3, 3 => 4, 4 => 5, 5 => 6)
const KO_BIG = _ko_big()
_ko_array(::Type{T}, ind, ext = KO_EXT) where {T} = randn(T, Tuple(ext[l] for l in ind)...)
_ko_stride(v, ind, l) = Base.strides(v)[findfirst(==(l), ind)]

function _ko_order(Av, indA, Bv, indB, Cv, indC; l2bytes)
    mlabels, nlabels, klabels = QuasiStrided.classify_labels(indA, indB, indC)
    morder = QuasiStrided.order_free_labels(mlabels, indC, Cv)
    norder = QuasiStrided.order_free_labels(nlabels, indC, Cv)
    ext(ls) = prod((size(Cv, findfirst(==(l), indC)) for l in ls); init = 1)
    return QuasiStrided.order_contract_labels(
        klabels, indA, Av, morder, indB, Bv, norder, ext(mlabels), ext(nlabels), l2bytes
    )
end

@testset "K order: cost model picks among indA / A-sorted / B-sorted" begin
    T = Float64
    Av = StridedView(_ko_array(T, (1, 2, 3, 4)))   # A[a,b,c,d]: a unit-stride, 9 >= a line
    Cv = StridedView(zeros(T, KO_EXT[1], KO_EXT[5]))

    # contract_scrambled, B[d,c,b,e]: B-sorted once the lines in flight exceed L2.
    Bs = StridedView(_ko_array(T, (4, 3, 2, 5)))
    @test _ko_order(Av, (1, 2, 3, 4), Bs, (4, 3, 2, 5), Cv, (1, 5); l2bytes = 0) == (4, 3, 2)
    @test _ko_order(Av, (1, 2, 3, 4), Bs, (4, 3, 2, 5), Cv, (1, 5); l2bytes = 1 << 20) == (2, 3, 4)

    # gemm_ready B[b,c,d,e] and b_permuted B[b,e,c,d]: B-sorted == indA order.
    for indB in ((2, 3, 4, 5), (2, 5, 3, 4))
        Bv = StridedView(_ko_array(T, indB))
        @test _ko_order(Av, (1, 2, 3, 4), Bv, indB, Cv, (1, 5); l2bytes = 0) == (2, 3, 4)
    end

    # TRG-shaped: sorting by B would give A a page-crossing K walk to save
    # refetches of a small B, so indA order stays.
    At = StridedView(randn(T, 8, 8, 8, 8, 8))
    Bt = StridedView(randn(T, 8, 8, 8))
    Ct = StridedView(zeros(T, 8, 8, 8, 8))
    for l2bytes in (0, 1 << 20)
        @test _ko_order(At, (1, 2, 3, 4, 5), Bt, (5, 6, 2), Ct, (1, 3, 4, 6); l2bytes) == (2, 5)
    end

    p = plan_contract(StridedView(zeros(T, 3, 5)), StridedView(randn(T, 3, 4)), (1, 2), StridedView(randn(T, 4, 5)), (2, 3), (1, 3))
    @test p.kgroup.strides == ((3,), (1,))
    # A singleton K axis is skipped as `kfast` and as `u`.
    A1 = StridedView(randn(T, 9, 1, 4, 5))
    B1 = StridedView(randn(T, 1, 5, 4, 6))
    o = _ko_order(A1, (1, 2, 3, 4), B1, (2, 4, 3, 5), Cv, (1, 5); l2bytes = 0)
    @test sort(collect(o)) == [2, 3, 4]
    @test filter(!=(2), o) == (4, 3)  # B-sorted among the non-singletons
end

@testset "K order: static-length tuple, inferred, allocation-free" begin
    T = Float64
    ext = KO_BIG
    Av = StridedView(_ko_array(T, (1, 2, 3, 4), ext))
    Bs = StridedView(_ko_array(T, (4, 3, 2, 5), ext))
    Cv = StridedView(zeros(T, ext[1], ext[5]))
    klabels, morder, norder = (2, 3, 4), (1,), (5,)
    f(l2) = QuasiStrided.order_contract_labels(klabels, (1, 2, 3, 4), Av, morder, (4, 3, 2, 5), Bs, norder, ext[1], ext[5], l2)
    g() = QuasiStrided.order_contract_labels(klabels, (1, 2, 3, 4), Av, morder, (4, 3, 2, 5), Bs, norder, ext[1], ext[5])
    @test @inferred(f(0)) === (4, 3, 2)
    @test @inferred(g()) isa NTuple{3, Int}
    f(0); g()
    @test (@allocated f(0)) == 0 skip = (VERSION < v"1.11")
    @test (@allocated g()) == 0 skip = (VERSION < v"1.11")
    h1() = QuasiStrided.order_contract_labels((2,), (1, 2), Av, (1,), (2, 5), Bs, (5,), 9, 40)
    h0() = QuasiStrided.order_contract_labels((), (1,), Av, (1,), (5,), Bs, (5,), 9, 40)
    @test @inferred(h1()) === (2,)
    @test @inferred(h0()) === ()
end

@testset "K order: plan_contract flips contract_scrambled at an L2-exceeding size" begin
    T = Float64
    ext = KO_BIG
    @test ext[2] * ext[3] * ext[5] * 64 > QuasiStrided.l2_core_bytes(target_profile())
    Av = StridedView(_ko_array(T, (1, 2, 3, 4), ext))
    Bs = StridedView(_ko_array(T, (4, 3, 2, 5), ext))   # B[d,c,b,e]
    Cv = StridedView(zeros(T, ext[1], ext[5]))
    plan = plan_contract(Cv, Av, (1, 2, 3, 4), Bs, (4, 3, 2, 5), (1, 5))
    @test plan.kgroup.lengths == (5, 40, 40)
    @test plan.kgroup.strides[2] == (1, 5, 200)
    @test plan.kgroup.strides[1] == Tuple(_ko_stride(Av, (1, 2, 3, 4), l) for l in (4, 3, 2))

    # gemm_ready at the same size keeps indA order.
    Bg = StridedView(_ko_array(T, (2, 3, 4, 5), ext))
    p = plan_contract(Cv, Av, (1, 2, 3, 4), Bg, (2, 3, 4, 5), (1, 5))
    @test p.kgroup.lengths == (40, 40, 5)
    @test p.kgroup.strides[1] == (9, 360, 14400)

    # Swapped (C stored as (e, a)): the same K order with the maps exchanged.
    Ct = StridedView(zeros(T, ext[5], ext[1]))
    kernel = SIMDKernel(Val(8), Val(6), T)
    pt = plan_contract(Ct, Av, (1, 2, 3, 4), Bs, (4, 3, 2, 5), (5, 1); kernel = kernel)
    @test pt.Astorage === parent(Bs)
    @test pt.kgroup.lengths == (5, 40, 40)
    @test pt.kgroup.strides[1] == (1, 5, 200)                     # B's map first
    @test pt.kgroup.strides[2] == (14400, 360, 9)                 # then A's
end

@testset "K order: correctness on scrambled K (alpha/beta, conj, both orientations)" begin
    Random.seed!(0x5C7A_0B1E)
    ext = KO_BIG
    for T in (Float64, ComplexF64)
        rtol = 500 * eps(real(T))
        alpha = T <: Complex ? T(1.3, -0.4) : T(1.3)
        beta = T <: Complex ? T(0.7, 0.2) : T(0.7)
        for (indA, indB, indC) in (
                    ((1, 2, 3, 4), (4, 3, 2, 5), (1, 5)),   # scrambled, C[a,e] (reordered K)
                    ((1, 2, 3, 4), (4, 3, 2, 5), (5, 1)),   # scrambled, C[e,a] (swap for real T)
                    ((4, 3, 2, 1), (2, 3, 4, 5), (1, 5)),   # A transposed, K inner in both
                    ((4, 3, 2, 1), (3, 2, 4, 5), (5, 1)),   # both scrambled, C transposed
                ), (conjA, conjB) in ((false, false), (true, true))
            (T <: Real) && conjA && continue
            A = _ko_array(T, indA, ext)
            B = _ko_array(T, indB, ext)
            Av = conjA ? StridedView(A, size(A), strides(A), 0, conj) : StridedView(A)
            Bv = StridedView(B)
            C = randn(T, Tuple(ext[l] for l in indC)...)
            Cstart = copy(C)
            Cv = StridedView(C)
            Cref = _lo_reference(Cstart, Av, indA, Bv, indB, indC; conjA, conjB, alpha, beta)
            plan = plan_contract(Cv, Av, indA, Bv, indB, indC; conjA = conjA, conjB = conjB)
            execute!(plan, alpha, beta)
            @test isapprox(C, Cref; rtol = rtol)
        end
    end
end
