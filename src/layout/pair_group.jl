# Two-map AxisGroups built from label positions in a pair of operands.

# Build the two-map AxisGroup for one of M/N/K: (v1,v2) is (A,C)/(B,C)/(A,B).
# Raises DimensionMismatch on a matched-label length mismatch.
#
# `labels` arrives as an `NTuple{D,Int}` whose length IS the composite's rank
# (`_classify_labels` in src/planning/labels.jl sizes it from the label
# tuples' lengths, see `_group_ranks` there), so `D` is a type parameter here
# and the group comes out as a concrete `AxisGroup{D,2}` with statically sized
# tuples throughout. A runtime-length list would instead be inferred as
# `Tuple{Vararg{Int}}`, heap-boxed, and read back through a dynamic
# `getindex`, allocating per group even at `D == 1`.
@inline function _build_pair_group(
        labels::NTuple{D, Int},
        ind1::NTuple{N1, Int}, v1::StridedView,
        ind2::NTuple{N2, Int}, v2::StridedView
    ) where {D, N1, N2}
    # Hoisted out of the per-dimension closures: `Base.strides` on a
    # `StridedView` rebuilds a tuple, so call it once, not once per `d`.
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
