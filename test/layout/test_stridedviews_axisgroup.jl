# AxisGroups built from StridedViews, cross-checked against StridedView's own
# indexing.

using StridedViews: StridedView, offset

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

@testset "AxisGroup from labels: worked example" begin
    A, B, C = StridedView(randn(3, 5, 2)), StridedView(randn(5, 4)), StridedView(zeros(3, 4, 2))
    indA, indB, indC = (1, 2, 3), (2, 4), (1, 4, 3)   # A[a,k,b] B[k,n] C[a,n,b]
    M = AxisGroup((1, 3), (indA, A), (indC, C))
    @test (M.lengths, M.strides) == ((3, 2), ((1, 15), (1, 12)))
    K = AxisGroup((2,), (indA, A), (indB, B))
    @test (K.lengths, K.strides) == ((5,), ((3,), (1,)))
    @test_throws DimensionMismatch AxisGroup((2,), (indA, A), (indB, StridedView(randn(6, 4))))
end
