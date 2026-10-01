# Planning helpers, the once-per-macro-block storage-bounds check, and the
# closed-form (`affine_ramp`) block description against the buffer path.

using StridedViews: StridedView, offset

const QS = QuasiStrided
const _pcf_plan = QuasiStrided.plan_contract
const _pcf_exec = QuasiStrided.execute!
const _pcf_exec_tw = QuasiStrided.execute_tilewise!
const _pcf_Plan = QuasiStrided.ContractPlan

@testset "per-call floor: _classify_labels" begin
    for (indA, indB, indC, want) in (
            ((1, 2), (2, 3), (1, 3), ((1,), (3,), (2,))),                          # GEMM
            ((1, 2), (2, 3), (3, 1), ((1,), (3,), (2,))),                          # transposed C
            ((1, 2, 3, 4), (3, 5, 6, 7), (4, 6, 7, 1, 2, 5), ((1, 2, 4), (5, 6, 7), (3,))),
            ((1, 2), (3, 1, 4, 5), (3, 2, 4, 5), ((2,), (3, 4, 5), (1,))),
            ((1, 2, 3, 4), (3, 4, 5, 6), (1, 2, 5, 6), ((1, 2), (5, 6), (3, 4))),
            ((1,), (1, 2), (2,), ((), (2,), (1,))),                                # no M
            ((1, 2), (3, 4), (1, 2, 3, 4), ((1, 2), (3, 4), ())),                  # outer product
        )
        got = QS._classify_labels(indA, indB, indC)
        @test got === want
        @test map(length, got) == QS._group_ranks(length(indA), length(indB), length(indC))
    end
    @test_throws ArgumentError QS._classify_labels((1, 1), (1, 2), (1, 2))
    @test_throws ArgumentError QS._classify_labels((1, 2), (2, 3), (1, 2, 3))  # all three
    @test_throws ArgumentError QS._classify_labels((1, 2), (3, 4), (1, 3))     # dangling in A
    @test_throws ArgumentError QS._classify_labels((1, 2), (3, 4), (1, 2, 3, 9))
end

@testset "per-call floor: AxisGroup from labels matches a direct construction" begin
    Random.seed!(1234)
    for trial in 1:60
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

        g = AxisGroup(Tuple(labels), (ind1, v1), (ind2, v2))
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
            AxisGroup, (typeof(Tuple(labels)), Tuple{typeof(ind1), typeof(v1)}, Tuple{typeof(ind2), typeof(v2)})
        ) == [AxisGroup{shared, 2}]
    end

    v1 = StridedView(randn(3, 4))
    v2 = StridedView(randn(5, 4))
    @test_throws DimensionMismatch AxisGroup((1,), ((1, 2), v1), ((1, 2), v2))

    # Rank zero: an outer product's K group.
    g0 = AxisGroup((), ((1, 2), v1), ((1, 2), v2))
    @test g0 isa AxisGroup{0, 2}
    @test axis_length(g0) == 1
end

@testset "per-call floor: plan_contract allocation stays under a ceiling" begin
    # A ceiling that a `Set`/`Vector`-based planner would exceed several-fold.
    A = randn(64, 64); B = randn(64, 64); C = zeros(64, 64)
    Av, Bv, Cv = StridedView(A), StridedView(B), StridedView(C)
    p = _pcf_plan(Cv, Av, (1, 2), Bv, (2, 3), (1, 3))
    ws = p.workspace
    f() = _pcf_plan(Cv, Av, (1, 2), Bv, (2, 3), (1, 3); workspace = ws)
    f()
    @test (@allocated f()) <= 3000
end

@testset "per-call floor: checked_span_bounds is equivalent to the per-tile check" begin
    # For a region split into slivers, checking the union range once accepts
    # exactly what checking each sliver accepts.
    Random.seed!(99)
    for trial in 1:150
        nrow = rand(1:9)
        ncol = rand(1:5)
        rowoffs = [rand(-20:20) for _ in 1:nrow]
        coloffs = [rand(-20:20) for _ in 1:ncol]
        base = rand(-5:30)
        len = rand(1:60)
        reg = rand(1:4)                      # sliver height
        nsliv = cld(nrow, reg)

        cols = view(coloffs, 1:ncol)
        persliver = true
        for s in 0:(nsliv - 1)
            first = s * reg
            cnt = min(reg, nrow - first)
            rows = view(rowoffs, (first + 1):(first + cnt))
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

    @test QS.checked_span_bounds(0, (0, -1), (0, 0), 1) === nothing
    @test QS.checked_span_bounds(0, (0, 0), (0, -1), 1) === nothing
    @test_throws BoundsError QS.checked_span_bounds(0, (0, 0), (0, 0), 0)
end

@testset "per-call floor: _classify_slivers! accumulates the TRUE block range" begin
    # The range handed to `checked_span_bounds` is the true min/max over the
    # whole block; a too-small range would silently under-validate.
    Random.seed!(20260922)
    for trial in 1:150
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
        @test all(s -> d1[s].count == min(reg, blocklen - (s - 1) * reg), 1:nsliv)
    end

    # Both extremes in an interior sliver.
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
    @test QS.descriptor_offset_range(d1[1], buf1, 0) == (1, 1)
    @test QS.descriptor_offset_range(d1[3], buf1, 12) == (1, 1)
end

@testset "per-call floor: descriptor_offset_range agrees with extrema" begin
    Random.seed!(7)
    for trial in 1:100
        n = rand(1:8)
        buf = [rand(-30:30) for _ in 1:n]
        rand() < 0.4 && (buf = [3 + 5 * (t - 1) for t in 1:n])   # force a regular run
        d = describe_block(buf, 0, n)
        GC.@preserve buf @test QS.descriptor_offset_range(d, buf, 0) == extrema(QS._axis_of(d, buf, 0))
    end
    @test QS.descriptor_offset_range(BlockDescriptor(0, 0, 0, true), Int[], 0) == (0, -1)
end

# Counts `length` calls: the hoisted bounds check asks once per operand per
# block, a per-tile check once per tile. Not a `DenseVector`, so the fast paths
# stay off.
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

    # One (N, K, M) block, so the hoisted checks run once.
    @test base.blocking.m_block >= Ma && base.blocking.n_block >= Na && base.blocking.k_block >= Ka
    m_tiles = cld(Ma, tile_size(kernel, 1))
    n_tiles = cld(Na, tile_size(kernel, 2))
    ntiles = m_tiles * n_tiles
    @test ntiles >= 20   # a per-tile check would be at least this many calls

    cstore = CountingStorage(zeros(Ma * Na))
    astore = CountingStorage(copy(vec(Amat)))
    bstore = CountingStorage(copy(vec(Bmat)))
    p = _pcf_Plan(
        base.kernel, base.mgroup, base.ngroup, base.kgroup, base.blocking,
        astore, 0, bstore, 0, cstore, 0,
        base.atransform, base.btransform, base.workspace, base.mpack, base.npack,
    )
    astore.n = 0; bstore.n = 0; cstore.n = 0
    _pcf_exec(p, 1.0, 0.0)
    @test reshape(cstore.data, Ma, Na) ≈ Amat * Bmat

    @test cstore.n == 1
    @test astore.n == 1
    @test bstore.n == 1

    # The oracle checks per tile.
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
            base.atransform, base.btransform, base.workspace, base.mpack, base.npack,
        )
    end

    # One element short: rejected before anything is written.
    short_C = zeros(Ma * Na - 1)
    @test_throws BoundsError _pcf_exec(replan(Cstorage = short_C), 1.0, 0.0)
    @test all(iszero, short_C)
    @test_throws BoundsError _pcf_exec_tw(replan(Cstorage = short_C), 1.0, 0.0)

    @test_throws BoundsError _pcf_exec(replan(Cbase = -1), 1.0, 0.0)

    @test_throws BoundsError _pcf_exec(replan(Astorage = zeros(Ma * Ka - 1)), 1.0, 0.0)
    @test_throws BoundsError _pcf_exec(replan(Bstorage = zeros(Ka * Na - 1)), 1.0, 0.0)
    @test_throws BoundsError _pcf_exec(replan(Abase = -1), 1.0, 0.0)
    @test_throws BoundsError _pcf_exec(replan(Bbase = -1), 1.0, 0.0)

    exact = zeros(Ma * Na)
    _pcf_exec(replan(Cstorage = exact), 1.0, 0.0)
    @test reshape(exact, Ma, Na) ≈ Amat * Bmat
end

@testset "per-call floor: rejection when the binding address is in an INTERIOR sliver" begin
    # C[m,n1,n2] = A[m,k] * B[k,n1,n2] with C reversed along n2: the N
    # offsets climb within each n1 run and fall at its boundary, so the block
    # maximum sits in an interior, irregular sliver (2 of 7).
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
    n_length = axis_length(base.ngroup)
    @test n_length == N1 * N2
    @test base.blocking.n_block >= n_length                         # one N block, so 7 slivers
    nsliv = cld(n_length, tile_size(kernel, 2))
    @test nsliv == 7

    noffs = [offsets(base.ngroup, q)[2] for q in 0:(n_length - 1)]
    binding = argmax(noffs) - 1                          # zero-based logical coordinate
    @test binding == N1 - 1
    @test 0 < binding ÷ tile_size(kernel, 2) < nsliv - 1           # a strictly interior sliver
    @test maximum(noffs[1:tile_size(kernel, 2)]) < noffs[binding + 1]                 # not the first
    @test maximum(noffs[(1 + (nsliv - 1) * tile_size(kernel, 2)):end]) < noffs[binding + 1]  # not the last

    _pcf_exec(base, 1.0, 0.0)
    ref = zeros(M, N1, N2)
    for m in 1:M, n1 in 1:N1, n2 in 1:N2
        ref[m, n1, n2] = sum(Amat[m, k] * Barr[k, n1, n2] for k in 1:K)
    end
    @test Cr ≈ ref

    # One element short: only the address at `binding` overflows.
    short_C = zeros(M * N1 * N2 - 1)
    pshort = _pcf_Plan(
        base.kernel, base.mgroup, base.ngroup, base.kgroup, base.blocking,
        base.Astorage, base.Abase, base.Bstorage, base.Bbase, short_C, base.Cbase,
        base.atransform, base.btransform, base.workspace, base.mpack, base.npack,
    )
    @test_throws BoundsError _pcf_exec(pshort, 1.0, 0.0)
    @test all(iszero, short_C)                            # nothing written before the throw

    # A first- or last-sliver-only range would accept it.
    moffs = [offsets(base.mgroup, q)[2] for q in 0:(axis_length(base.mgroup) - 1)]
    mrange = (minimum(moffs), maximum(moffs))
    NR = tile_size(kernel, 2)
    truerange = (minimum(noffs), maximum(noffs))
    firstonly = (minimum(noffs[1:NR]), maximum(noffs[1:NR]))
    lastonly = let tail = noffs[(1 + (nsliv - 1) * NR):end]
        (minimum(tail), maximum(tail))
    end
    shortlen = length(short_C)
    @test_throws BoundsError QS.checked_span_bounds(base.Cbase, mrange, truerange, shortlen)
    @test QS.checked_span_bounds(base.Cbase, mrange, firstonly, shortlen) === nothing
    @test QS.checked_span_bounds(base.Cbase, mrange, lastonly, shortlen) === nothing

    # The oracle rejects too, but only at the offending tile, after writing
    # the ones before it.
    fill!(short_C, 0.0)
    @test_throws BoundsError _pcf_exec_tw(pshort, 1.0, 0.0)
    @test any(!iszero, short_C)

    # An interior-sliver minimum: the base shifted down by one.
    plow = _pcf_Plan(
        base.kernel, base.mgroup, base.ngroup, base.kgroup, base.blocking,
        base.Astorage, base.Abase, base.Bstorage, base.Bbase,
        zeros(M * N1 * N2), base.Cbase - 1,
        base.atransform, base.btransform, base.workspace, base.mpack, base.npack,
    )
    @test_throws BoundsError _pcf_exec(plow, 1.0, 0.0)
end

@testset "per-call floor: the unsafe_* packers keep every non-bounds check" begin
    kernel = SIMDKernel(Val(8), Val(6), Float64)
    src = Tile(collect(1.0:64.0), 0, AffineAxis(0, 1, 8), AffineAxis(0, 8, 4))
    packed = zeros(8 * 4)

    ref = zeros(8 * 4)
    pack_a!(ref, src, kernel, identity)
    QS.unsafe_pack_a!(packed, src, kernel, identity)
    @test packed == ref

    # Only the bounds checks are skipped.
    @test_throws DimensionMismatch QS.unsafe_pack_a!(zeros(3), src, kernel, identity)
    @test_throws ArgumentError QS.unsafe_pack_a!(zeros(Float32, 64), src, kernel, identity)
    toowide = Tile(collect(1.0:200.0), 0, AffineAxis(0, 1, 9), AffineAxis(0, 16, 4))
    @test_throws ArgumentError QS.unsafe_pack_a!(zeros(200), toowide, kernel, identity)

    srcB = Tile(collect(1.0:64.0), 0, AffineAxis(0, 1, 4), AffineAxis(0, 4, 6))
    refB = zeros(6 * 4)
    packedB = zeros(6 * 4)
    pack_b!(refB, srcB, kernel, identity)
    QS.unsafe_pack_b!(packedB, srcB, kernel, identity)
    @test packedB == refB
    @test_throws DimensionMismatch QS.unsafe_pack_b!(zeros(3), srcB, kernel, identity)

    dest = Tile(zeros(8 * 6), 0, AffineAxis(0, 1, 8), AffineAxis(0, 8, 6))
    @test_throws DimensionMismatch QS.unsafe_execute_tile!(
        kernel, dest, zeros(3), zeros(6 * 4), 4, 1.0, 0.0
    )
    @test_throws ArgumentError QS.unsafe_execute_tile!(
        kernel, dest, zeros(8 * 4), zeros(6 * 4), -1, 1.0, 0.0
    )
    d1 = Tile(zeros(8 * 6), 0, AffineAxis(0, 1, 8), AffineAxis(0, 8, 6))
    d2 = Tile(zeros(8 * 6), 0, AffineAxis(0, 1, 8), AffineAxis(0, 8, 6))
    execute_tile!(kernel, d1, ref, refB, 4, 1.0, 0.0)
    QS.unsafe_execute_tile!(kernel, d2, ref, refB, 4, 1.0, 0.0)
    @test d1.storage == d2.storage
end

@testset "per-call floor: affine_ramp classifies exactly the rank-<=1 folds" begin
    ar = QS.affine_ramp

    @test ar(AxisGroup((), ((), ()))) == (true, (0, 0))
    @test ar(AxisGroup((7,), ((3,), (-2,)))) == (true, (3, -2))
    # Singleton dimensions never advance, whatever their stride claims.
    @test ar(AxisGroup((1, 7, 1), ((99, 3, -4), (5, -2, 8)))) == (true, (3, -2))
    # Foldable on both maps, or on one only.
    @test ar(AxisGroup((4, 5), ((1, 4), (2, 8)))) == (true, (1, 2))
    @test first(ar(AxisGroup((4, 5), ((1, 4), (2, 9))))) == false
    @test first(ar(AxisGroup((16, 16, 16), ((1, 256, 4096), (1, 256, 4096))))) == false
    # A three-deep fold uses the accumulated length.
    @test ar(AxisGroup((2, 3, 4), ((1, 2, 6), (5, 10, 30)))) == (true, (1, 5))
    @test first(ar(AxisGroup((2, 3, 4), ((1, 2, 4), (5, 10, 30))))) == false
    # Empty domain is vacuously a ramp.
    @test first(ar(AxisGroup((0, 3), ((1, 4), (1, 4))))) == true

    # For every ramp, offsets(g, q) == q .* steps.
    Random.seed!(5150)
    for trial in 1:100
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
            Q >= 2 || continue
            s = offsets(g, 1)
            @test any(q -> offsets(g, q) != (q * s[1], q * s[2]), 0:(Q - 1))
        end
    end
end

@testset "per-call floor: _ramp_slivers! reproduces _classify_slivers! exactly" begin
    Random.seed!(31337)
    for trial in 1:150
        D = rand(1:3)
        lens = ntuple(_ -> rand(1:5), D)
        # A ramp by construction half the time.
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

        for s in 1:nsliv
            @test r1[s].base == d1[s].base && r1[s].stride == d1[s].stride
            @test r1[s].count == d1[s].count && r1[s].regular == d1[s].regular
            @test r2[s].base == d2[s].base && r2[s].stride == d2[s].stride
            @test r2[s].count == d2[s].count && r2[s].regular == d2[s].regular
        end
        @test got == want
    end
end

@testset "per-call floor: ramp and buffer paths give the same contraction" begin
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

    # ao2mo_2: a ramp on M and K but not on N.
    d = 6
    p = check(
        (d, d, d, d), (d, d), (d, d, d, d),
        (1, 2), (3, 1, 4, 5), (3, 2, 4, 5); kernel = kernel
    )
    @test first(QS.affine_ramp(p.mgroup))
    @test first(QS.affine_ramp(p.kgroup))
    @test !first(QS.affine_ramp(p.ngroup))   # the mixed path, both branches live

    # Several macro blocks, so ramp descriptors start at a nonzero `first`.
    check(
        (40, 30), (40, 21), (21, 30), (1, 2), (2, 3), (1, 3);
        kernel = kernel, m_block = 8, k_block = 5, n_block = 6
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

    # Permuted A, reversed B, sliced C: the non-ramp path.
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

    # M, N and K composites ordered differently per operand: scattered tile axes.
    A4 = randn(5, 6, 7, 9); B4 = randn(7, 6, 11, 3); C4 = zeros(3, 9, 11, 5)
    p4 = _pcf_plan(
        StridedView(C4), StridedView(A4), (1, 2, 3, 4), StridedView(B4), (3, 2, 5, 6), (6, 4, 5, 1)
    )
    _pcf_exec(p4, 1.0, 0.5)
    _pcf_exec(p4, 1.0, 0.5)
    @test (@allocated _pcf_exec(p4, 1.0, 0.5)) == 0 skip = (VERSION < v"1.11")
end
