# AxisGroup indexing against an oracle built on `CartesianIndices` (first axis
# fastest), never on fill_offsets!/block_descriptors! themselves.

include("../helpers.jl")

using QuasiStrided: affine_ramp

function oracle_offsets(lengths::NTuple{D, Int}, strides::NTuple{P, NTuple{D, Int}}) where {D, P}
    return [ntuple(p -> sum((Tuple(ci)[d] - 1) * strides[p][d] for d in 1:D; init = 0), P) for ci in vec(CartesianIndices(lengths))]
end
oracle_map(lengths, strides, p) = [o[p] for o in oracle_offsets(lengths, strides)]
fields(d::BlockDescriptor) = (d.base, d.stride, d.count, d.regular)

@testset "AxisGroup: worked A/B/C example" begin
    M = AxisGroup((3, 2), ((1, 15), (1, 12)))  # A, C
    N = AxisGroup((4,), ((5,), (3,)))          # B, C
    K = AxisGroup((5,), ((3,), (1,)))          # A, B
    for (g, e1, e2) in (
            (M, [0, 1, 2, 15, 16, 17], [0, 1, 2, 12, 13, 14]),
            (N, [0, 5, 10, 15], [0, 3, 6, 9]),
            (K, [0, 3, 6, 9, 12], [0, 1, 2, 3, 4]),
        )
        Q = axis_length(g)
        @test [offsets(g, q) for q in 0:(Q - 1)] == collect(zip(e1, e2))
        b1, b2 = zeros(Int, Q), zeros(Int, Q)
        fill_offsets!((b1, b2), g, 0, Q)
        @test (b1, b2) == (e1, e2)
    end
    # M intervals -> descriptors (regular runs, a boundary-crossing irregular one, empty).
    for (first, count, expA, expC) in (
            (0, 3, (0, 1, 3, true), (0, 1, 3, true)),
            (3, 3, (15, 1, 3, true), (12, 1, 3, true)),
            (0, 4, (0, 0, 4, false), (0, 0, 4, false)),
            (2, 2, (2, 13, 2, true), (2, 10, 2, true)),
            (6, 0, (0, 0, 0, true), (0, 0, 0, true)),
        )
        dA, dC = block_descriptors!((zeros(Int, 4), zeros(Int, 4)), M, first, count)
        @test (fields(dA), fields(dC)) == (expA, expC)
    end
end

@testset "AxisGroup: randomized offsets, intervals, descriptors, ramps" begin
    rng = Random.MersenneTwister(0xA51CE_1)
    for trial in 1:200
        D, P = rand(rng, 0:3), rand(rng, 1:3)
        lengths = ntuple(_ -> rand(rng, 0:4), D)
        strides = ntuple(_ -> ntuple(_ -> rand(rng, -8:8), D), P)
        g = AxisGroup(lengths, strides)
        full = oracle_offsets(lengths, strides)
        Q = axis_length(g)
        @test Q == length(full)
        isramp, steps = affine_ramp(g)
        if Q > 0
            step1 = Q > 1 ? full[2] : ntuple(_ -> 0, P)
            @test isramp == (full == [step1 .* q for q in 0:(Q - 1)])
            isramp && @test steps == step1
        end
        for _ in 1:4
            first = rand(rng, 0:Q)
            count = rand(rng, 0:(Q - first))
            bufs = ntuple(_ -> fill(-1, count + 2), P)   # the suffix must stay untouched
            descs = block_descriptors!(bufs, g, first, count)
            for p in 1:P
                expected = [full[i + 1][p] for i in first:(first + count - 1)]
                @test bufs[p] == [expected; -1; -1]
                d = descs[p]
                affine = count <= 1 || expected == [expected[1] + t * (expected[2] - expected[1]) for t in 0:(count - 1)]
                @test d.regular == affine && d.count == count
                affine && count >= 1 && @test [d.base + t * d.stride for t in 0:(count - 1)] == expected
            end
        end
    end
end

@testset "AxisGroup: zero-length and singleton dimensions" begin
    g = AxisGroup((3, 0, 5), ((1, 10, 100),))
    @test axis_length(g) == 0
    @test_throws BoundsError offsets(g, 0)
    # A zero length makes the domain empty even when the other lengths overflow.
    @test axis_length(AxisGroup((typemax(Int) ÷ 2, typemax(Int) ÷ 2, 0), ((1, 1, 1),))) == 0
    g0 = AxisGroup((), ((), (), ()))
    @test axis_length(g0) == 1 && offsets(g0, 0) == (0, 0, 0)
    # Singleton strides never contribute, even extreme ones.
    g1 = AxisGroup((1,), ((typemin(Int),),))
    @test offsets(g1, 0) == (0,)
end

@testset "AxisGroup: constructor validation and overflow policy" begin
    @test_throws ArgumentError AxisGroup((-1,), ((1,),))
    @test_throws ArgumentError AxisGroup((3, -2), ((1, 1),))
    @test_throws ArgumentError AxisGroup((3,), ())             # P == 0
    @test_throws OverflowError AxisGroup((typemax(Int) ÷ 2 + 2, typemax(Int) ÷ 2 + 2), ((1, 1),))
    @test_throws OverflowError AxisGroup((typemax(Int),), ((2,),))
    @test_throws OverflowError AxisGroup((2,), ((typemin(Int),),))   # abs(typemin) must not wrap
    @test axis_length(AxisGroup((0,), ((typemax(Int),),))) == 0
    @test axis_length(AxisGroup((2,), ((typemax(Int) ÷ 2,),))) == 2   # right at the boundary
end

@testset "fill_offsets!: invalid requests leave buffers untouched" begin
    g = AxisGroup((3, 2), ((1, 10), (2, 20)))
    Q = axis_length(g)
    for (first, count, err) in ((0, -1, ArgumentError), (Q, 1, BoundsError), (-1, 1, BoundsError), (1, Q, BoundsError))
        a, b = fill(-7, Q), fill(-7, Q)
        @test_throws err fill_offsets!((a, b), g, first, count)
        @test a == b == fill(-7, Q)
    end
    short, ok = fill(-7, Q - 1), fill(-7, Q)
    @test_throws DimensionMismatch fill_offsets!((short, ok), g, 0, Q)
    @test ok == fill(-7, Q)
    fill_offsets!((ok, short), g, Q, 0)                            # empty interval at Q
    @test ok == fill(-7, Q)
end

@testset "describe_block: edge cases, overflow, invalid arguments" begin
    @test fields(describe_block(Int[], 0)) == (0, 0, 0, true)
    @test fields(describe_block([42], 1)) == (42, 0, 1, true)
    @test fields(describe_block([5, 8, 11, 14], 4)) == (5, 3, 4, true)
    @test fields(describe_block([0, 5, 1, 6], 4)) == (0, 0, 4, false)
    @test fields(describe_block([typemin(Int), typemax(Int)], 2)) == (typemin(Int), 0, 2, false)
    @test fields(describe_block([1, 2, 42, 3], 2, 1)) == (42, 0, 1, true)
    @test fields(describe_block(collect(0:2:20), 3, 4)) == (6, 2, 4, true)
    @test fields(describe_block(collect(1:10), 10, 0)) == (0, 0, 0, true)
    @test_throws ArgumentError describe_block([1, 2, 3], -1)
    @test_throws DimensionMismatch describe_block([1, 2], 3)
    @test_throws ArgumentError describe_block([1, 2, 3], -1, 1)
    @test_throws DimensionMismatch describe_block([1, 2, 3], 2, 2)
end

# AxisGroups built from StridedViews, cross-checked against StridedView's own
# indexing.
@testset "AxisGroup over a StridedView reproduces its indexing ($name)" for (name, v) in (
        ("permuted", permutedims(StridedView(reshape(collect(1.0:30.0), 3, 5, 2)), (3, 1, 2))),
        ("sliced", view(StridedView(reshape(collect(1.0:60.0), 3, 5, 4)), 2:3, 1:3, 4:2:4)),
        ("permuted and sliced", permutedims(view(StridedView(reshape(collect(1.0:120.0), 4, 5, 6)), 2:4, 2:5, 1:2:5), (3, 1, 2))),
        ("negative stride", StridedView(collect(1.0:12.0), (4, 3), (-1, 3), 3)),
        ("zero stride", StridedView(collect(1.0:5.0), (5, 3), (1, 0), 0)),
    )
    g = AxisGroup(size(v), (Base.strides(v),))
    @test axis_length(g) == length(v)
    for (q, ci) in enumerate(CartesianIndices(size(v)))
        (o,) = offsets(g, q - 1)
        @test parent(v)[offset(v) + o + 1] == v[ci]
    end
end

@testset "AxisGroup from labels matches a direct construction" begin
    A, B, C = StridedView(randn(3, 5, 2)), StridedView(randn(5, 4)), StridedView(zeros(3, 4, 2))
    indA, indB, indC = (1, 2, 3), (2, 4), (1, 4, 3)   # A[a,k,b] B[k,n] C[a,n,b]
    M = AxisGroup((1, 3), (indA, A), (indC, C))
    @test (M.lengths, M.strides) == ((3, 2), ((1, 15), (1, 12)))
    K = AxisGroup((2,), (indA, A), (indB, B))
    @test (K.lengths, K.strides) == ((5,), ((3,), (1,)))
    @test_throws DimensionMismatch AxisGroup((2,), (indA, A), (indB, StridedView(randn(6, 4))))

    Random.seed!(1234)
    for trial in 1:20
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

@testset "affine_ramp classifies exactly the rank-<=1 folds" begin
    ar = QuasiStrided.affine_ramp

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
end
