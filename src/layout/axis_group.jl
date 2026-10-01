# A grouped tensor axis: `D` dims enumerated by one zero-based logical
# coordinate, fastest dim first, with `P` per-operand stride maps. Pure layout:
# no base offset, storage or eltype.

struct AxisGroup{D, P}
    lengths::NTuple{D, Int}
    strides::NTuple{P, NTuple{D, Int}}

    function AxisGroup{D, P}(
            lengths::NTuple{D, Int},
            strides::NTuple{P, NTuple{D, Int}}
        ) where {D, P}
        P >= 1 || throw(ArgumentError("AxisGroup requires at least one map (P >= 1), got P = $P"))
        for (d, L) in enumerate(lengths)
            L >= 0 || throw(ArgumentError("AxisGroup lengths must be nonnegative, got lengths[$d] = $L"))
        end
        validate_axis_group_bounds(lengths, strides)
        return new{D, P}(lengths, strides)
    end
end

AxisGroup(lengths::NTuple{D, Int}, strides::NTuple{P, NTuple{D, Int}}) where {D, P} =
    AxisGroup{D, P}(lengths, strides)

# The checks that make all later offset arithmetic safe in plain `Int`: the
# cardinality, and every map's offset range `sum((L[d]-1)*abs(S[d]))`.
function validate_axis_group_bounds(
        lengths::NTuple{D, Int},
        strides::NTuple{P, NTuple{D, Int}}
    ) where {D, P}
    any(iszero, lengths) && return nothing
    foldl(checked_mul, lengths; init = 1)
    for S in strides
        acc = 0
        for d in 1:D
            lengths[d] > 1 || continue
            acc = checked_add(acc, checked_mul(lengths[d] - 1, checked_abs(S[d])))
        end
    end
    return nothing
end

"""
    AxisGroup(labels, (ind1, v1), (ind2, v2))

The two-map group of `labels` in operands `v1` and `v2`, whose axes carry the
labels `ind1` and `ind2`. Throws a `DimensionMismatch` if a label's axis
lengths differ.
"""
@inline function AxisGroup(
        labels::NTuple{D, Int},
        (ind1, v1)::Tuple{NTuple{N1, Int}, StridedView},
        (ind2, v2)::Tuple{NTuple{N2, Int}, StridedView}
    ) where {D, N1, N2}
    # `Base.strides` on a `StridedView` rebuilds a tuple: call it once.
    st1 = Base.strides(v1)
    st2 = Base.strides(v2)
    pos1 = ntuple(d -> findfirst(==(@inbounds labels[d]), ind1)::Int, Val(D))
    pos2 = ntuple(d -> findfirst(==(@inbounds labels[d]), ind2)::Int, Val(D))
    lens = ntuple(Val(D)) do d
        l1 = size(v1, pos1[d])
        l2 = size(v2, pos2[d])
        l1 == l2 || throw_label_length((@inbounds labels[d]), l1, l2)
        l1
    end
    s1 = ntuple(d -> st1[pos1[d]], Val(D))
    s2 = ntuple(d -> st2[pos2[d]], Val(D))
    return AxisGroup(lens, (s1, s2))
end

@noinline throw_label_length(label::Int, l1::Int, l2::Int) = throw(
    DimensionMismatch("label $label has mismatched axis length: $l1 vs $l2")
)

axis_length(g::AxisGroup) = prod(g.lengths)

# Not inline lambdas: a closure over loop-reassigned variables is boxed and
# allocates every call.
@inline function add_offsets(offs::NTuple{P, Int}, g::AxisGroup{D, P}, d::Int, x::Int) where {D, P}
    return ntuple(p -> offs[p] + x * g.strides[p][d], Val(P))
end

@inline function sub_reset_offsets(offs::NTuple{P, Int}, g::AxisGroup{D, P}, d::Int, Ld::Int) where {D, P}
    return ntuple(p -> offs[p] - (Ld - 1) * g.strides[p][d], Val(P))
end

function fill_offsets!(
        buffers::NTuple{P, Vector{Int}}, g::AxisGroup{D, P},
        first::Int, count::Int
    ) where {D, P}
    Q = axis_length(g)

    count >= 0 || throw(ArgumentError("count must be nonnegative, got $count"))
    (0 <= first <= Q) ||
        throw(BoundsError("AxisGroup interval start $first out of range [0, $Q]", first))
    count <= Q - first ||
        throw(BoundsError("AxisGroup interval (first=$first, count=$count) exceeds domain size $Q", first))

    for p in 1:P
        length(buffers[p]) >= count ||
            throw(DimensionMismatch("buffer $p has length $(length(buffers[p])), need at least $count"))
    end

    count == 0 && return buffers

    # Decode `first` once, then step with reset-before-carry increments.
    r = first
    x = ntuple(_ -> 0, Val(D))
    offs = ntuple(_ -> 0, Val(P))
    for d in 1:D
        L = g.lengths[d]
        xd = r % L
        r = r ÷ L
        x = Base.setindex(x, xd, d)
        if xd != 0
            offs = add_offsets(offs, g, d, xd)
        end
    end

    t = 0
    @inbounds while true
        for p in 1:P
            buffers[p][t + 1] = offs[p]
        end
        t += 1
        t == count && break

        d = 1
        while d <= D
            L = g.lengths[d]
            xd = x[d]
            if xd < L - 1
                x = Base.setindex(x, xd + 1, d)
                offs = add_offsets(offs, g, d, 1)
                break
            else
                x = Base.setindex(x, 0, d)
                offs = sub_reset_offsets(offs, g, d, L)
                d += 1
            end
        end
    end

    return buffers
end

# The step of map `p` if it is a single ramp, `offsets(g, q)[p] == q * step`
# (0 for an empty group), else `nothing`.
function map_ramp_step(g::AxisGroup{D}, p::Int) where {D}
    step = 0
    run = 1
    started = false
    for d in 1:D
        L = g.lengths[d]
        L == 0 && return 0
        L == 1 && continue
        S = g.strides[p][d]
        if !started
            step = S
            run = L
            started = true
        else
            Int128(run) * Int128(step) == Int128(S) || return nothing
            run *= L
        end
    end
    return step
end

# Whether every map is a single ramp, so that the offset buffer can be replaced
# by arithmetic, and the steps (meaningless when not).
function affine_ramp(g::AxisGroup{D, P}) where {D, P}
    steps = ntuple(_ -> 0, Val(P))
    for p in 1:P
        step = map_ramp_step(g, p)
        step === nothing && return (false, ntuple(_ -> 0, Val(P)))
        steps = Base.setindex(steps, step, p)
    end
    return (true, steps)
end
