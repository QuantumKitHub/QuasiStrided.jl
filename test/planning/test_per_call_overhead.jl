# Per-call planning cost and bounds-check placement: cheaper planning,
# once-per-block bounds validation, closed-form affine blocks.
#
# Three things are pinned here, each of which the rest of the suite would only
# catch indirectly:
#
#   1. the planning helpers (`_classify_labels`, `_order_free_labels`,
#      `_build_pair_group`) against straightforward reference implementations,
#   2. the once-per-macro-block storage-bounds check: that it REJECTS exactly
#      what per-sliver checks would reject, accepts exactly what they would
#      accept, and runs once per block (counted, not assumed),
#   3. `affine_ramp` and the closed-form block description built on it, against
#      the buffer-materializing path -- descriptor by descriptor.

using Test
using Random
using StridedViews: StridedView, offset

const QS = QuasiStrided

const _pcf_plan = QuasiStrided.plan_contract
const _pcf_exec = QuasiStrided.execute!
const _pcf_exec_tw = QuasiStrided.execute_tilewise!
const _pcf_Plan = QuasiStrided.ContractPlan

# ===========================================================================
# 1. Planning helpers
# ===========================================================================

# Reference `_classify_labels`, written the obvious way (Sets + push!).
function _ref_classify(indA, indB, indC)
    setA, setB, setC = Set(indA), Set(indB), Set(indC)
    m = Int[]; k = Int[]; n = Int[]
    for l in indA
        (l in setB) && (l in setC) && error("all three")
        (l in setB) ? push!(k, l) : ((l in setC) ? push!(m, l) : error("dangling"))
    end
    for l in indB
        (l in setA) && continue
        (l in setC) ? push!(n, l) : error("dangling")
    end
    return m, n, k
end

@testset "per-call floor: _classify_labels matches the Set-based reference" begin
    cases = (
        ((1, 2), (2, 3), (1, 3)),                       # plain GEMM
        ((1, 2), (2, 3), (3, 1)),                       # transposed C
        ((1, 2, 3, 4), (3, 5, 6, 7), (4, 6, 7, 1, 2, 5)),  # ccsd_t_1 shape
        ((1, 2), (3, 1, 4, 5), (3, 2, 4, 5)),           # ao2mo_2 shape
        ((1, 2, 3, 4), (3, 4, 5, 6), (1, 2, 5, 6)),     # dim15_2_2_2 shape
        ((1,), (1, 2), (2,)),                           # no M labels
        ((1, 2), (3, 4), (1, 2, 3, 4)),                 # pure outer product
    )
    for (indA, indB, indC) in cases
        got = QS._classify_labels(indA, indB, indC)
        want = _ref_classify(indA, indB, indC)
        # Statically sized tuples, one per composite, with the ranks fixed by
        # the label tuples' lengths (`_group_ranks`); same labels, same order.
        @test map(collect, got) == want
        @test all(l -> l isa Tuple{Vararg{Int}}, got)
        @test isconcretetype(typeof(got))
        # Trimmed to the real counts, not left at their `NA`/`NB` worst case.
        @test length(got[1]) + length(got[3]) == length(indA)
        @test length(got[2]) + length(got[3]) == length(indB)
        @test (length(got[1]), length(got[2]), length(got[3])) ==
            QS._group_ranks(length(indA), length(indB), length(indC))
    end

    # Rejections.
    @test_throws ArgumentError QS._classify_labels((1, 1), (1, 2), (1, 2))
    @test_throws ArgumentError QS._classify_labels((1, 2), (2, 3), (1, 2, 3))  # all three
    @test_throws ArgumentError QS._classify_labels((1, 2), (3, 4), (1, 3))     # dangling in A
    @test_throws ArgumentError QS._classify_labels((1, 2), (3, 4), (1, 2, 3, 9))
end

@testset "per-call floor: _order_free_labels is a stable ascending |C-stride| sort" begin
    Random.seed!(20260921)
    for trial in 1:200
        nd = rand(1:5)
        dims = ntuple(_ -> rand(1:4), nd)
        C = StridedView(randn(dims))
        indC = ntuple(identity, nd)
        st = Base.strides(C)
        labels = shuffle(collect(indC))[1:rand(1:nd)]
        got = QS._order_free_labels(Tuple(labels), indC, C)
        want = labels[sortperm([abs(st[l]) for l in labels]; alg = Base.Sort.DEFAULT_STABLE)]
        @test collect(got) == want
        # Same statically sized tuple type in as out (planning/test_plan_contract.jl's
        # label-order pinning reads `_classify_labels`'s output after ordering it).
        @test got isa NTuple{length(labels), Int}
    end

    # Ties keep input order, explicitly: a size-1 axis of a square array gives
    # equal |stride| only when the strides really are equal, so build the tie
    # by hand with a 1-element axis.
    C = StridedView(randn(1, 1, 3))
    @test QS._order_free_labels((2, 1), (1, 2, 3), C) === (2, 1)
    @test QS._order_free_labels((1, 2), (1, 2, 3), C) === (1, 2)
end

@testset "per-call floor: _build_pair_group matches a direct construction" begin
    Random.seed!(1234)
    for trial in 1:200
        nd1 = rand(1:4)
        nd2 = rand(nd1:5)
        shared = rand(1:nd1)                      # how many labels the group has
        dims1 = ntuple(_ -> rand(1:4), nd1)
        v1 = StridedView(randn(dims1))
        ind1 = ntuple(identity, nd1)
        labels = shuffle(collect(ind1))[1:shared]
        # v2 carries the same labels (same lengths) plus filler.
        ind2 = (labels..., ntuple(d -> 100 + d, nd2 - shared)...)
        dims2 = (
            ntuple(d -> size(v1, findfirst(==(labels[d]), ind1)::Int), shared)...,
            ntuple(_ -> rand(1:4), nd2 - shared)...,
        )
        v2 = StridedView(randn(dims2))

        # The label tuple's length IS the rank (`_group_ranks`), so the group
        # type is concrete without any runtime rank ladder.
        g = QS._build_pair_group(Tuple(labels), ind1, v1, ind2, v2)
        s1 = Base.strides(v1)
        s2 = Base.strides(v2)
        p1 = ntuple(d -> findfirst(==(labels[d]), ind1)::Int, shared)
        p2 = ntuple(d -> findfirst(==(labels[d]), ind2)::Int, shared)
        @test g.lengths == ntuple(d -> size(v1, p1[d]), shared)
        @test g.strides[1] == ntuple(d -> s1[p1[d]], shared)
        @test g.strides[2] == ntuple(d -> s2[p2[d]], shared)
        @test g isa AxisGroup{shared, 2}
        @test isconcretetype(typeof(g))
        @test Base.return_types(
            QS._build_pair_group, (typeof(Tuple(labels)), typeof(ind1), typeof(v1), typeof(ind2), typeof(v2))
        ) == [AxisGroup{shared, 2}]
    end

    # A matched label whose lengths disagree is still a DimensionMismatch.
    v1 = StridedView(randn(3, 4))
    v2 = StridedView(randn(5, 4))
    @test_throws DimensionMismatch QS._build_pair_group((1,), (1, 2), v1, (1, 2), v2)

    # Rank zero is reachable and well formed (an outer product's K group).
    g0 = QS._build_pair_group((), (1, 2), v1, (1, 2), v2)
    @test g0 isa AxisGroup{0, 2}
    @test axis_length(g0) == 1
end

@testset "per-call floor: plan_contract allocation stays well under the old floor" begin
    # A fully allocation-free planner is out of reach because each
    # composite's RANK is a value property, so this pins a generous ceiling
    # rather than zero. It catches a Set/`ntuple(f, ::Int)`-style planner
    # (~5-7 KB per call on these shapes), which would blow through it by 2-3x.
    A = randn(64, 64); B = randn(64, 64); C = zeros(64, 64)
    Av, Bv, Cv = StridedView(A), StridedView(B), StridedView(C)
    p = _pcf_plan(Cv, Av, (1, 2), Bv, (2, 3), (1, 3))
    ws = p.workspace
    f() = _pcf_plan(Cv, Av, (1, 2), Bv, (2, 3), (1, 3); workspace = ws)
    f()
    @test (@allocated f()) <= 3000
end

# ===========================================================================
# 2. Once-per-macro-block storage-bounds validation
# ===========================================================================

@testset "per-call floor: checked_span_bounds is equivalent to the per-tile check" begin
    # The safety argument, stated as a property: for a rectangular region
    # split into slivers along one axis, checking the UNION range once accepts
    # exactly the inputs that checking each sliver accepts. Randomized over
    # positive/negative/zero strides and both regular and scattered axes.
    Random.seed!(99)
    for trial in 1:400
        nrow = rand(1:9)
        ncol = rand(1:5)
        rowoffs = [rand(-20:20) for _ in 1:nrow]
        coloffs = [rand(-20:20) for _ in 1:ncol]
        base = rand(-5:30)
        len = rand(1:60)
        reg = rand(1:4)                      # sliver height
        nsliv = cld(nrow, reg)

        cols = ScatterAxis(coloffs, ncol)
        persliver = true
        for s in 0:(nsliv - 1)
            first = s * reg
            cnt = min(reg, nrow - first)
            rows = ScatterAxis(view(rowoffs, (first + 1):(first + cnt)), cnt)
            ok = try
                checked_tile_storage_bounds(base, rows, cols, len)
                true
            catch e
                e isa BoundsError || rethrow()
                false
            end
            persliver &= ok
        end

        blockrange = (minimum(rowoffs), maximum(rowoffs))
        colrange = (minimum(coloffs), maximum(coloffs))
        blockok = try
            QS.checked_span_bounds(base, blockrange, colrange, len)
            true
        catch e
            e isa BoundsError || rethrow()
            false
        end
        @test blockok == persliver
    end

    # The empty conventions of `axis_offset_range` pass through unchanged.
    @test QS.checked_span_bounds(0, (0, -1), (0, 0), 1) === nothing
    @test QS.checked_span_bounds(0, (0, 0), (0, -1), 1) === nothing
    @test_throws BoundsError QS.checked_span_bounds(0, (0, 0), (0, 0), 0)
end

@testset "per-call floor: _classify_slivers! accumulates the TRUE block range" begin
    # The test above proves `checked_span_bounds` is equivalent to the
    # per-sliver checks GIVEN a correct aggregate range, because it computes
    # that range itself. This one closes the other half: that
    # `_classify_slivers!` -- the thing that actually produces the range the
    # driver hands to `checked_span_bounds` -- accumulates the true min/max
    # over the whole block and not, say, the first or the last sliver's.
    #
    # That failure mode is invisible everywhere else in this file: a too-SMALL
    # aggregate range throws nothing and still computes the right values on
    # every in-bounds input, so it is the silently-under-validated direction
    # and needs a direct assertion. Buffers are written by hand rather than
    # through `fill_offsets!`, so the irregular/scattered case -- which the
    # ramp-vs-classify test below cannot reach, since ramps are regular by
    # construction -- is exercised on purpose.
    Random.seed!(20260922)
    for trial in 1:500
        blocklen = rand(1:24)
        reg = rand(1:5)
        nsliv = cld(blocklen, reg)
        buf1 = [rand(-40:40) for _ in 1:blocklen]
        buf2 = [rand(-40:40) for _ in 1:blocklen]
        d1 = Vector{BlockDescriptor}(undef, nsliv)
        d2 = Vector{BlockDescriptor}(undef, nsliv)
        (r1, r2) = QS._classify_slivers!(d1, d2, buf1, buf2, blocklen, reg, nsliv)
        @test r1 == (minimum(buf1), maximum(buf1))
        @test r2 == (minimum(buf2), maximum(buf2))
        # Sanity: with random contents and a wide enough sliver, at least some
        # of these descriptors really are irregular, i.e. the scan branch of
        # `descriptor_offset_range` is the one under test.
        @test all(s -> d1[s].count == min(reg, blocklen - (s - 1) * reg), 1:nsliv)
    end

    # Both extremes planted in an INTERIOR sliver, with the first and last
    # slivers deliberately unremarkable. A first-only or last-only
    # accumulation passes every other assertion in this file and fails here.
    blocklen, reg = 18, 6                      # slivers 1:6, 7:12, 13:18
    buf1 = fill(1, blocklen); buf1[8] = 500; buf1[9] = -500   # sliver 2, irregular
    buf2 = fill(2, blocklen)
    buf2[7:12] .= [0, -7, -14, -21, -28, -35]                 # sliver 2, REGULAR, stride -7
    d1 = Vector{BlockDescriptor}(undef, 3)
    d2 = Vector{BlockDescriptor}(undef, 3)
    (r1, r2) = QS._classify_slivers!(d1, d2, buf1, buf2, blocklen, reg, 3)
    @test !d1[2].regular && d1[2].count == 6        # scan branch
    @test d2[2].regular && d2[2].stride == -7       # affine branch, negative stride
    @test r1 == (-500, 500)
    @test r2 == (-35, 2)
    # ... and the first/last slivers alone would have said something else.
    @test QS.descriptor_offset_range(d1[1], buf1, 0) == (1, 1)
    @test QS.descriptor_offset_range(d1[3], buf1, 12) == (1, 1)
end

@testset "per-call floor: descriptor_offset_range agrees with axis_offset_range" begin
    Random.seed!(7)
    for trial in 1:300
        n = rand(1:8)
        buf = [rand(-30:30) for _ in 1:n]
        rand() < 0.4 && (buf = [3 + 5 * (t - 1) for t in 1:n])   # force a regular run
        d = describe_block(buf, 0, n)
        ax = QS._axis_of(d, buf, 0)
        @test QS.descriptor_offset_range(d, buf, 0) == axis_offset_range(ax)
    end
    @test QS.descriptor_offset_range(BlockDescriptor(0, 0, 0, true), Int[], 0) == (0, -1)
end

# A storage wrapper that counts `length` calls. `checked_span_bounds` is
# reached from `_execute_nest!` through exactly one `length(plan.Xstorage)`
# per `execute!`, where a per-sliver/per-tile check would call it once per
# sliver/tile. Not a `DenseVector`, so the vectorized pack/store fast paths stay
# off -- which is fine: the question here is how many times the bounds check
# runs, not which inner loop does.
mutable struct CountingStorage{T} <: AbstractVector{T}
    data::Vector{T}
    n::Int
end
CountingStorage(v::Vector{T}) where {T} = CountingStorage{T}(v, 0)
Base.size(c::CountingStorage) = size(c.data)
Base.length(c::CountingStorage) = (c.n += 1; length(c.data))
Base.@propagate_inbounds Base.getindex(c::CountingStorage, i::Int) = c.data[i]
Base.@propagate_inbounds Base.setindex!(c::CountingStorage, v, i::Int) = (c.data[i] = v)
Base.IndexStyle(::Type{<:CountingStorage}) = IndexLinear()

@testset "per-call floor: the destination bounds check runs once per macro block" begin
    Ma, Ka, Na = 40, 7, 30
    kernel = SIMDKernel(Val(8), Val(6), Float64)
    Amat, Bmat = randn(Ma, Ka), randn(Ka, Na)
    Cmat = zeros(Ma, Na)
    base = _pcf_plan(
        StridedView(Cmat), StridedView(Amat), (1, 2), StridedView(Bmat), (2, 3),
        (1, 3); kernel = kernel
    )

    # One (jc, pc, ic) block at this blocking, so the hoisted checks run once.
    @test base.blocking.mc >= Ma && base.blocking.nc >= Na && base.blocking.kc >= Ka
    m_slivers = cld(Ma, mr(kernel))
    n_slivers = cld(Na, nr(kernel))
    ntiles = m_slivers * n_slivers
    @test ntiles >= 20   # a per-tile check would be at least this many calls

    cstore = CountingStorage(zeros(Ma * Na))
    astore = CountingStorage(copy(vec(Amat)))
    bstore = CountingStorage(copy(vec(Bmat)))
    p = _pcf_Plan(
        base.kernel, base.mgroup, base.ngroup, base.kgroup, base.blocking,
        astore, 0, bstore, 0, cstore, 0,
        base.atransform, base.btransform, base.workspace,
    )
    astore.n = 0; bstore.n = 0; cstore.n = 0
    _pcf_exec(p, 1.0, 0.0)
    @test reshape(cstore.data, Ma, Na) ≈ Amat * Bmat

    # Exactly one `length` per operand per `execute!`: the three hoisted
    # `checked_span_bounds` calls. (No other code path on this plan asks the
    # storage for its length -- `_pack_a_contiguous_eligible` and
    # `_vector_store_eligible` both fail on the type, not the length.)
    @test cstore.n == 1
    @test astore.n == 1
    @test bstore.n == 1

    # The fully checked oracle still checks per tile, which is what makes it an
    # independent oracle: it asks far more often.
    fill!(cstore.data, 0.0)
    astore.n = 0; bstore.n = 0; cstore.n = 0
    _pcf_exec_tw(p, 1.0, 0.0)
    @test reshape(cstore.data, Ma, Na) ≈ Amat * Bmat
    @test cstore.n >= ntiles
end

@testset "per-call floor: a block that must be rejected is still rejected" begin
    Ma, Ka, Na = 40, 7, 30
    kernel = SIMDKernel(Val(8), Val(6), Float64)
    Amat, Bmat = randn(Ma, Ka), randn(Ka, Na)
    Cmat = zeros(Ma, Na)
    base = _pcf_plan(
        StridedView(Cmat), StridedView(Amat), (1, 2), StridedView(Bmat), (2, 3),
        (1, 3); kernel = kernel
    )

    function replan(;
            Cstorage = vec(Cmat), Astorage = vec(Amat), Bstorage = vec(Bmat),
            Cbase = 0, Abase = 0, Bbase = 0
        )
        return _pcf_Plan(
            base.kernel, base.mgroup, base.ngroup, base.kgroup, base.blocking,
            Astorage, Abase, Bstorage, Bbase, Cstorage, Cbase,
            base.atransform, base.btransform, base.workspace,
        )
    end

    # Destination one element short: the last micro-tile of the block would
    # write past the end. The once-per-block check must catch it, BEFORE
    # anything is written.
    short_C = zeros(Ma * Na - 1)
    @test_throws BoundsError _pcf_exec(replan(Cstorage = short_C), 1.0, 0.0)
    @test all(iszero, short_C)
    # ... and the fully checked oracle agrees, which is the point of keeping it.
    @test_throws BoundsError _pcf_exec_tw(replan(Cstorage = short_C), 1.0, 0.0)

    # A base offset that pushes the destination off the front.
    @test_throws BoundsError _pcf_exec(replan(Cbase = -1), 1.0, 0.0)

    # Source operands short: caught before any read, on A and on B separately.
    @test_throws BoundsError _pcf_exec(replan(Astorage = zeros(Ma * Ka - 1)), 1.0, 0.0)
    @test_throws BoundsError _pcf_exec(replan(Bstorage = zeros(Ka * Na - 1)), 1.0, 0.0)
    @test_throws BoundsError _pcf_exec(replan(Abase = -1), 1.0, 0.0)
    @test_throws BoundsError _pcf_exec(replan(Bbase = -1), 1.0, 0.0)

    # Exactly-sized storage is accepted (the check is not off-by-one tight).
    exact = zeros(Ma * Na)
    _pcf_exec(replan(Cstorage = exact), 1.0, 0.0)
    @test reshape(exact, Ma, Na) ≈ Amat * Bmat
end

@testset "per-call floor: rejection when the binding address is in an INTERIOR sliver" begin
    # The rejection test above uses a dense column-major GEMM, whose offsets
    # increase monotonically, so its out-of-bounds address is necessarily in
    # the LAST sliver of the block -- which a first-only or last-only range
    # accumulation would still catch. This fixture puts the binding address
    # strictly in the middle, and makes the slivers irregular while it is at
    # it.
    #
    # C[m,n1,n2] = A[m,k] * B[k,n1,n2], with C a REVERSED view along n2, so
    # the N composite's C map is (+M, -M*N1) over lengths (N1, N2): offsets
    # climb by M inside each n1 run of 13 and then fall by 13M at every run
    # boundary. At NR = 6 that makes slivers straddle the boundary (hence
    # irregular), and it puts the block MAXIMUM at logical coordinate
    # q = N1-1 = 12 -- sliver index 2 of 7, an interior one -- while the first
    # and last slivers top out far below it.
    M, K, N1, N2 = 20, 4, 13, 3
    kernel = SIMDKernel(Val(8), Val(6), Float64)
    Amat = randn(M, K)
    Barr = randn(K, N1, N2)
    Cfull = zeros(M, N1, N2)
    Cr = view(Cfull, :, :, N2:-1:1)
    Av, Bv, Cv = StridedView(Amat), StridedView(Barr), StridedView(Cr)

    base = _pcf_plan(Cv, Av, (1, 2), Bv, (2, 3, 4), (1, 3, 4); kernel = kernel)
    @test base.Astorage === parent(Av)                   # no M/N swap: M's run is 20 >= 8
    @test !first(QS.affine_ramp(base.ngroup))            # the buffer path, not the ramp path
    Qn = axis_length(base.ngroup)
    @test Qn == N1 * N2
    @test base.blocking.nc >= Qn                         # one jc block, so 7 slivers
    nsliv = cld(Qn, nr(kernel))
    @test nsliv == 7

    # Where the extreme actually sits, derived rather than asserted by hand.
    noffs = [offsets(base.ngroup, q)[2] for q in 0:(Qn - 1)]
    binding = argmax(noffs) - 1                          # zero-based logical coordinate
    @test binding == N1 - 1
    @test 0 < binding ÷ nr(kernel) < nsliv - 1           # a strictly interior sliver
    @test maximum(noffs[1:nr(kernel)]) < noffs[binding + 1]                 # not the first
    @test maximum(noffs[(1 + (nsliv - 1) * nr(kernel)):end]) < noffs[binding + 1]  # not the last

    # The whole, correctly sized destination is accepted and computes the
    # right answer -- so the fixture is a legitimate contraction, not one the
    # engine would reject anyway.
    _pcf_exec(base, 1.0, 0.0)
    ref = zeros(M, N1, N2)
    for m in 1:M, n1 in 1:N1, n2 in 1:N2
        ref[m, n1, n2] = sum(Amat[m, k] * Barr[k, n1, n2] for k in 1:K)
    end
    @test Cr ≈ ref

    # Now one element short. The only address that overflows is the one
    # attained at `binding`, inside sliver 2; a range accumulated from the
    # first or the last sliver alone would accept this plan and write out of
    # bounds.
    short_C = zeros(M * N1 * N2 - 1)
    pshort = _pcf_Plan(
        base.kernel, base.mgroup, base.ngroup, base.kgroup, base.blocking,
        base.Astorage, base.Abase, base.Bstorage, base.Bbase, short_C, base.Cbase,
        base.atransform, base.btransform, base.workspace,
    )
    @test_throws BoundsError _pcf_exec(pshort, 1.0, 0.0)
    @test all(iszero, short_C)                            # nothing written before the throw

    # That this fixture DISCRIMINATES -- i.e. that a first-sliver-only or
    # last-sliver-only aggregate range would have accepted the short destination
    # and written out of bounds -- is asserted directly rather than by mutating
    # the driver, because a driver with that bug really does perform the
    # out-of-bounds write (corrupting the heap). Feeding `checked_span_bounds`
    # the three candidate ranges settles the same question deterministically and
    # without executing anything.
    moffs = [offsets(base.mgroup, q)[2] for q in 0:(axis_length(base.mgroup) - 1)]
    mrange = (minimum(moffs), maximum(moffs))
    NR = nr(kernel)
    truerange = (minimum(noffs), maximum(noffs))
    firstonly = (minimum(noffs[1:NR]), maximum(noffs[1:NR]))
    lastonly = let tail = noffs[(1 + (nsliv - 1) * NR):end]
        (minimum(tail), maximum(tail))
    end
    shortlen = length(short_C)
    # The true range rejects ...
    @test_throws BoundsError QS.checked_span_bounds(base.Cbase, mrange, truerange, shortlen)
    # ... while either truncated range silently accepts. This is exactly the
    # "too-small aggregate range throws nothing" failure mode, and it is what
    # the test above (`_classify_slivers! accumulates the TRUE block range`)
    # is there to make impossible.
    @test QS.checked_span_bounds(base.Cbase, mrange, firstonly, shortlen) === nothing
    @test QS.checked_span_bounds(base.Cbase, mrange, lastonly, shortlen) === nothing

    # The fully checked oracle agrees that this plan must be rejected --
    # independently, and via a different mechanism (its own per-tile check).
    #
    # It does NOT leave the destination untouched, and that difference is
    # pinned here on purpose: `execute_tilewise!` validates each tile as it
    # reaches it, so it writes every tile that precedes the offending one,
    # whereas `execute!`'s once-per-block check fails before writing anything
    # in the block. That makes the failure mode STRICTER, not weaker -- but
    # only per block: with several macro blocks, earlier
    # blocks can still have been written before a later block's check throws,
    # so "atomic" is a claim about one block, not about `execute!`.
    fill!(short_C, 0.0)
    @test_throws BoundsError _pcf_exec_tw(pshort, 1.0, 0.0)
    @test any(!iszero, short_C)

    # The same shape with an interior-sliver MINIMUM: shifting the base down
    # by one makes the smallest address -1, and that minimum is attained in
    # the last n1 run, not the first sliver.
    plow = _pcf_Plan(
        base.kernel, base.mgroup, base.ngroup, base.kgroup, base.blocking,
        base.Astorage, base.Abase, base.Bstorage, base.Bbase,
        zeros(M * N1 * N2), base.Cbase - 1,
        base.atransform, base.btransform, base.workspace,
    )
    @test_throws BoundsError _pcf_exec(plow, 1.0, 0.0)
end

@testset "per-call floor: the unsafe_* packers keep every non-bounds check" begin
    kernel = SIMDKernel(Val(8), Val(6), Float64)
    src = SourceTile(collect(1.0:64.0), 0, AffineAxis(0, 1, 8), AffineAxis(0, 8, 4))
    packed = zeros(8 * 4)

    # Agreement with the checked entry point, value for value.
    ref = zeros(8 * 4)
    pack_a!(ref, src, kernel, identity)
    QS.unsafe_pack_a!(packed, src, kernel, identity)
    @test packed == ref

    # Capacity, extent and eltype checks are kept; only bounds checks are
    # skipped.
    @test_throws DimensionMismatch QS.unsafe_pack_a!(zeros(3), src, kernel, identity)
    @test_throws ArgumentError QS.unsafe_pack_a!(zeros(Float32, 64), src, kernel, identity)
    toowide = SourceTile(collect(1.0:200.0), 0, AffineAxis(0, 1, 9), AffineAxis(0, 16, 4))
    @test_throws ArgumentError QS.unsafe_pack_a!(zeros(200), toowide, kernel, identity)

    srcB = SourceTile(collect(1.0:64.0), 0, AffineAxis(0, 1, 4), AffineAxis(0, 4, 6))
    refB = zeros(6 * 4)
    packedB = zeros(6 * 4)
    pack_b!(refB, srcB, kernel, identity)
    QS.unsafe_pack_b!(packedB, srcB, kernel, identity)
    @test packedB == refB
    @test_throws DimensionMismatch QS.unsafe_pack_b!(zeros(3), srcB, kernel, identity)

    # `unsafe_execute_tile!` likewise keeps the extent/capacity prologue.
    dest = DestinationTile(zeros(8 * 6), 0, AffineAxis(0, 1, 8), AffineAxis(0, 8, 6))
    @test_throws DimensionMismatch QS.unsafe_execute_tile!(
        kernel, dest, zeros(3), zeros(6 * 4), 4, 1.0, 0.0
    )
    @test_throws ArgumentError QS.unsafe_execute_tile!(
        kernel, dest, zeros(8 * 4), zeros(6 * 4), -1, 1.0, 0.0
    )
    # And it computes the same tile as the checked one.
    d1 = DestinationTile(zeros(8 * 6), 0, AffineAxis(0, 1, 8), AffineAxis(0, 8, 6))
    d2 = DestinationTile(zeros(8 * 6), 0, AffineAxis(0, 1, 8), AffineAxis(0, 8, 6))
    execute_tile!(kernel, d1, ref, refB, 4, 1.0, 0.0)
    QS.unsafe_execute_tile!(kernel, d2, ref, refB, 4, 1.0, 0.0)
    @test d1.storage == d2.storage
end

# ===========================================================================
# 3. affine_ramp and the closed-form block description
# ===========================================================================

@testset "per-call floor: affine_ramp classifies exactly the rank-<=1 folds" begin
    ar = QS.affine_ramp

    # Rank zero: Q == 1, offsets always 0.
    @test ar(AxisGroup((), ((), ()))) == (true, (0, 0))
    # Rank one.
    @test ar(AxisGroup((7,), ((3,), (-2,)))) == (true, (3, -2))
    # Singleton dimensions never advance, whatever their stride claims.
    @test ar(AxisGroup((1, 7, 1), ((99, 3, -4), (5, -2, 8)))) == (true, (3, -2))
    # Foldable rank two on BOTH maps.
    @test ar(AxisGroup((4, 5), ((1, 4), (2, 8)))) == (true, (1, 2))
    # Foldable on map 1 only -> not a ramp.
    @test first(ar(AxisGroup((4, 5), ((1, 4), (2, 9))))) == false
    # Non-foldable (the ao2mo_2 N composite: a stride-16 label sits between).
    @test first(ar(AxisGroup((16, 16, 16), ((1, 256, 4096), (1, 256, 4096))))) == false
    # Three-deep fold uses the ACCUMULATED length, not the previous one.
    @test ar(AxisGroup((2, 3, 4), ((1, 2, 6), (5, 10, 30)))) == (true, (1, 5))
    @test first(ar(AxisGroup((2, 3, 4), ((1, 2, 4), (5, 10, 30))))) == false
    # Empty domain is vacuously a ramp.
    @test first(ar(AxisGroup((0, 3), ((1, 4), (1, 4))))) == true

    # The definition itself: for every ramp, offsets(g, q) == q .* steps.
    Random.seed!(5150)
    for trial in 1:300
        D = rand(1:3)
        lens = ntuple(_ -> rand(1:4), D)
        strd = ntuple(_ -> ntuple(_ -> rand(-6:6), D), 2)
        g = AxisGroup(lens, strd)
        (isramp, steps) = ar(g)
        Q = axis_length(g)
        if isramp
            for q in 0:(Q - 1)
                @test offsets(g, q) == (q * steps[1], q * steps[2])
            end
        else
            # Not a ramp: some q must disagree with EVERY candidate step pair,
            # which for a nonempty domain means disagreeing with offsets(g, 1).
            Q >= 2 || continue
            s = offsets(g, 1)
            @test any(q -> offsets(g, q) != (q * s[1], q * s[2]), 0:(Q - 1))
        end
        # `normalize_group` must agree about the rank.
        Q > 0 && @test isramp == (length(normalize_group(g).lengths) <= 1)
    end
end

@testset "per-call floor: _ramp_slivers! reproduces _classify_slivers! exactly" begin
    Random.seed!(31337)
    for trial in 1:300
        D = rand(1:3)
        lens = ntuple(_ -> rand(1:5), D)
        # Build a ramp by construction about half the time, random otherwise.
        strd = if rand() < 0.5
            s1 = rand(-5:5); s2 = rand(-5:5)
            acc = 1
            t1 = Int[]; t2 = Int[]
            for d in 1:D
                push!(t1, acc * s1); push!(t2, acc * s2)
                acc *= lens[d]
            end
            (Tuple(t1), Tuple(t2))
        else
            ntuple(_ -> ntuple(_ -> rand(-5:5), D), 2)
        end
        g = AxisGroup(lens, strd)
        Q = axis_length(g)
        Q == 0 && continue
        (isramp, steps) = QS.affine_ramp(g)
        isramp || continue

        reg = rand(1:4)
        first = rand(0:(Q - 1))
        blocklen = rand(1:(Q - first))
        nsliv = cld(blocklen, reg)

        buf1 = zeros(Int, blocklen); buf2 = zeros(Int, blocklen)
        d1 = Vector{BlockDescriptor}(undef, nsliv)
        d2 = Vector{BlockDescriptor}(undef, nsliv)
        fill_offsets!((buf1, buf2), g, first, blocklen)
        want = QS._classify_slivers!(d1, d2, buf1, buf2, blocklen, reg, nsliv)

        r1 = Vector{BlockDescriptor}(undef, nsliv)
        r2 = Vector{BlockDescriptor}(undef, nsliv)
        got = QS._ramp_slivers!(r1, r2, steps[1], steps[2], first, blocklen, reg, nsliv)

        # Descriptors identical field for field, not merely equivalent.
        for s in 1:nsliv
            @test r1[s].base == d1[s].base && r1[s].stride == d1[s].stride
            @test r1[s].count == d1[s].count && r1[s].regular == d1[s].regular
            @test r2[s].base == d2[s].base && r2[s].stride == d2[s].stride
            @test r2[s].count == d2[s].count && r2[s].regular == d2[s].regular
        end
        # ... and the block ranges the hoisted bounds check consumes agree.
        @test got == want
    end
end

@testset "per-call floor: ramp and buffer paths give the same contraction" begin
    # `plain` is a ramp on all three composites; `ao2mo` is a ramp on M and K
    # but NOT on N; `blocked` forces several (jc, pc, ic) blocks so the ramp
    # descriptors are exercised at a nonzero `first`.
    Random.seed!(424242)
    kernel = SIMDKernel(Val(8), Val(6), Float64)

    function check(Csz, Asz, Bsz, indA, indB, indC; kw...)
        A = randn(Asz); B = randn(Bsz); C = randn(Csz)
        C0 = copy(C)
        p = _pcf_plan(
            StridedView(C), StridedView(A), indA, StridedView(B), indB, indC; kw...
        )
        _pcf_exec(p, 2.0, -0.5)
        ref = copy(C0)
        p2 = _pcf_plan(
            StridedView(ref), StridedView(A), indA, StridedView(B), indB, indC; kw...
        )
        _pcf_exec_tw(p2, 2.0, -0.5)
        @test C ≈ ref
        return p
    end

    p = check((40, 30), (40, 7), (7, 30), (1, 2), (2, 3), (1, 3); kernel = kernel)
    @test first(QS.affine_ramp(p.mgroup)) && first(QS.affine_ramp(p.ngroup)) &&
        first(QS.affine_ramp(p.kgroup))

    # ao2mo_2 at dim 6: C[a,b,r,s] = A[q,b] * B[a,q,r,s].
    d = 6
    p = check(
        (d, d, d, d), (d, d), (d, d, d, d),
        (1, 2), (3, 1, 4, 5), (3, 2, 4, 5); kernel = kernel
    )
    @test first(QS.affine_ramp(p.mgroup))
    @test first(QS.affine_ramp(p.kgroup))
    @test !first(QS.affine_ramp(p.ngroup))   # the mixed path, both branches live

    # Several macro blocks: force tiny mc/kc/nc so `first` is nonzero.
    check(
        (40, 30), (40, 21), (21, 30), (1, 2), (2, 3), (1, 3);
        kernel = kernel, mc = 8, kc = 5, nc = 6
    )
    # Non-ramp composites on both sides (rank-4 operands, non-folding order).
    d = 5
    check(
        (d, d, d, d, d, d), (d, d, d, d), (d, d, d, d),
        (1, 2, 3, 4), (3, 5, 6, 7), (4, 6, 7, 1, 2, 5); kernel = kernel
    )
end

@testset "per-call floor: execute! still allocates nothing in steady state" begin
    Ma, Ka, Na = 40, 21, 30
    A = randn(Ma, Ka); B = randn(Ka, Na); C = zeros(Ma, Na)
    p = _pcf_plan(
        StridedView(C), StridedView(A), (1, 2), StridedView(B), (2, 3), (1, 3)
    )
    _pcf_exec(p, 1.0, 0.0)
    _pcf_exec(p, 1.0, 0.0)
    allocs = @allocated _pcf_exec(p, 1.0, 0.0)
    @test C ≈ A * B
    @test allocs == 0 skip = (VERSION < v"1.11")

    # Same on a scattered/permuted fixture, where every composite is a ramp on
    # one side only and the fallback path runs.
    Ap = permutedims(randn(Ka, Ma), (2, 1))
    Bn = view(randn(Ka, 2Na), :, (2Na):-1:(Na + 1))
    Cs = view(zeros(2Ma, Na), 1:Ma, :)
    ps = _pcf_plan(
        StridedView(Cs), StridedView(Ap), (1, 2), StridedView(Bn), (2, 3), (1, 3)
    )
    _pcf_exec(ps, 1.0, 0.0)
    _pcf_exec(ps, 1.0, 0.0)
    allocs_s = @allocated _pcf_exec(ps, 1.0, 0.0)
    @test Cs ≈ Ap * Bn
    @test allocs_s == 0 skip = (VERSION < v"1.11")
end
