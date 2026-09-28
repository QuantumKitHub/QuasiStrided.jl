# The two-map AxisGroup for one of M/N/K from label positions in a pair of
# operands: (v1, v2) is (A, C), (B, C) or (A, B). `labels` is an `NTuple{D,Int}`
# so the group is a concrete `AxisGroup{D,2}`; a runtime-length list would be
# heap-boxed and allocate per group.
@inline function _build_pair_group(
        labels::NTuple{D, Int},
        ind1::NTuple{N1, Int}, v1::StridedView,
        ind2::NTuple{N2, Int}, v2::StridedView
    ) where {D, N1, N2}
    # `Base.strides` on a `StridedView` rebuilds a tuple: call it once.
    st1 = Base.strides(v1)
    st2 = Base.strides(v2)
    pos1 = ntuple(d -> findfirst(==(@inbounds labels[d]), ind1)::Int, Val(D))
    pos2 = ntuple(d -> findfirst(==(@inbounds labels[d]), ind2)::Int, Val(D))
    lens = ntuple(Val(D)) do d
        l1 = size(v1, pos1[d])
        l2 = size(v2, pos2[d])
        l1 == l2 || _throw_label_length((@inbounds labels[d]), l1, l2)
        l1
    end
    s1 = ntuple(d -> st1[pos1[d]], Val(D))
    s2 = ntuple(d -> st2[pos2[d]], Val(D))
    return AxisGroup(lens, (s1, s2))
end
