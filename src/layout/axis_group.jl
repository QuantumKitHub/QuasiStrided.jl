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
        _validate_axis_group_bounds(lengths, strides)
        return new{D, P}(lengths, strides)
    end
end

AxisGroup(lengths::NTuple{D, Int}, strides::NTuple{P, NTuple{D, Int}}) where {D, P} =
    AxisGroup{D, P}(lengths, strides)

# Validation-time only; Int128 so nothing (incl. abs(typemin(Int))) can wrap.
function _checked_axis_length(lengths::NTuple{D, Int}) where {D}
    any(==(0), lengths) && return 0
    q = one(Int128)
    for L in lengths
        q *= Int128(L)
        q > Int128(typemax(Int)) &&
            throw(OverflowError("AxisGroup cardinality (product of lengths) exceeds typemax(Int)"))
    end
    return Int(q)
end

function _validate_axis_group_bounds(
        lengths::NTuple{D, Int},
        strides::NTuple{P, NTuple{D, Int}}
    ) where {D, P}
    Q = _checked_axis_length(lengths)
    Q == 0 && return nothing
    for (p, S) in enumerate(strides)
        acc = zero(Int128)
        for d in 1:D
            acc += Int128(lengths[d] - 1) * abs(Int128(S[d]))
            acc > Int128(typemax(Int)) &&
                throw(
                OverflowError(
                    "AxisGroup map $p: sum((L[d]-1)*abs(S[d])) exceeds typemax(Int); " *
                        "this layout is not representable under the conservative offset-range bound"
                )
            )
        end
    end
    return nothing
end

axis_length(g::AxisGroup) = _unchecked_axis_length(g.lengths)

@inline function _unchecked_axis_length(lengths::NTuple{D, Int}) where {D}
    q = 1
    for L in lengths
        q *= L
    end
    return q
end

# Type-stable, stack-allocated "replace element i of an NTuple{N,Int}".
@inline _tupleset(t::NTuple{N, Int}, i::Int, v::Int) where {N} =
    ntuple(j -> ifelse(j == i, v, t[j]), Val(N))

# Not inline lambdas: a closure over loop-reassigned variables is boxed and
# allocates every call.
@inline function _add_offsets(offs::NTuple{P, Int}, g::AxisGroup{D, P}, d::Int, x::Int) where {D, P}
    return ntuple(p -> offs[p] + x * g.strides[p][d], Val(P))
end

@inline function _sub_reset_offsets(offs::NTuple{P, Int}, g::AxisGroup{D, P}, d::Int, Ld::Int) where {D, P}
    return ntuple(p -> offs[p] - (Ld - 1) * g.strides[p][d], Val(P))
end

function offsets(g::AxisGroup{D, P}, q::Int) where {D, P}
    Q = axis_length(g)
    (0 <= q < Q) || throw(BoundsError(g, q))
    r = q
    offs = ntuple(_ -> 0, Val(P))
    for d in 1:D
        L = g.lengths[d]
        x = r % L
        r = r ÷ L
        if x != 0
            offs = _add_offsets(offs, g, d, x)
        end
    end
    return offs
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
    for i in 1:P, j in (i + 1):P
        buffers[i] === buffers[j] &&
            throw(ArgumentError("buffers must be distinct Vector{Int} objects (buffers $i and $j alias)"))
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
        x = _tupleset(x, d, xd)
        if xd != 0
            offs = _add_offsets(offs, g, d, xd)
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
                x = _tupleset(x, d, xd + 1)
                offs = _add_offsets(offs, g, d, 1)
                break
            else
                x = _tupleset(x, d, 0)
                offs = _sub_reset_offsets(offs, g, d, L)
                d += 1
            end
        end
    end

    return buffers
end

# Whether every map is a single ramp `offsets(g, q)[p] == q * steps[p]` (so the
# offset buffer can be replaced by arithmetic). `steps` is meaningless when not.
function affine_ramp(g::AxisGroup{D, P}) where {D, P}
    zerosteps = ntuple(_ -> 0, Val(P))
    steps = zerosteps
    run = 1
    started = false
    for d in 1:D
        L = g.lengths[d]
        L == 0 && return (true, zerosteps)  # empty domain: vacuously a ramp.
        L == 1 && continue                  # singleton: coordinate never advances.
        Sd = ntuple(p -> g.strides[p][d], Val(P))
        if !started
            steps = Sd
            run = L
            started = true
        else
            for p in 1:P
                Int128(run) * Int128(steps[p]) == Int128(Sd[p]) ||
                    return (false, zerosteps)
            end
            run *= L  # a sub-product of the validated cardinality
        end
    end
    return (true, steps)
end

# An offset interval of one map: `regular` iff `buffer[t+1] == base + t*stride`
# for all `t < count`; otherwise read the buffer (valid until it is refilled).
struct BlockDescriptor
    base::Int
    stride::Int
    count::Int
    regular::Bool
end

# Classifies `buffer[first+1 : first+count]`; an overflowing difference is irregular.
function describe_block(buffer::Vector{Int}, first::Int, count::Int)
    first >= 0 || throw(ArgumentError("first must be nonnegative, got $first"))
    count >= 0 || throw(ArgumentError("count must be nonnegative, got $count"))
    first + count <= length(buffer) ||
        throw(
        DimensionMismatch(
            "buffer length $(length(buffer)) is less than first+count = $(first + count)"
        )
    )

    count == 0 && return BlockDescriptor(0, 0, 0, true)

    @inbounds base = buffer[first + 1]
    count == 1 && return BlockDescriptor(base, 0, 1, true)

    @inbounds stride, overflowed = Base.Checked.sub_with_overflow(buffer[first + 2], buffer[first + 1])
    overflowed && return BlockDescriptor(base, 0, count, false)

    @inbounds for t in 2:(count - 1)
        diff, ovf = Base.Checked.sub_with_overflow(buffer[first + t + 1], buffer[first + t])
        (ovf || diff != stride) && return BlockDescriptor(base, 0, count, false)
    end

    return BlockDescriptor(base, stride, count, true)
end

describe_block(buffer::Vector{Int}, count::Int) = describe_block(buffer, 0, count)

function block_descriptors!(
        buffers::NTuple{P, Vector{Int}}, g::AxisGroup{D, P},
        first::Int, count::Int
    ) where {D, P}
    fill_offsets!(buffers, g, first, count)
    return ntuple(p -> describe_block(buffers[p], count), Val(P))
end

# Same offset sequence with singleton dims dropped and adjacent dims folded
# where `next_stride[p] == length * stride[p]` for every map. Never reorders.
function normalize_group(g::AxisGroup{D, P}) where {D, P}
    axis_length(g) == 0 && return g

    dims = Tuple{Int, NTuple{P, Int}}[]
    for d in 1:D
        L = g.lengths[d]
        if L != 1
            push!(dims, (L, ntuple(p -> g.strides[p][d], P)))
        end
    end

    if isempty(dims)
        emptylengths = NTuple{0, Int}()
        emptystrides = ntuple(_ -> NTuple{0, Int}(), P)
        return AxisGroup(emptylengths, emptystrides)
    end

    folded = Tuple{Int, NTuple{P, Int}}[]
    curL, curS = dims[1]
    for i in 2:length(dims)
        nextL, nextS = dims[i]
        foldable = true
        for p in 1:P
            if Int128(curL) * Int128(curS[p]) != Int128(nextS[p])
                foldable = false
                break
            end
        end
        if foldable
            curL = curL * nextL
        else
            push!(folded, (curL, curS))
            curL, curS = nextL, nextS
        end
    end
    push!(folded, (curL, curS))

    newD = length(folded)
    newlengths = ntuple(i -> folded[i][1], newD)
    newstrides = ntuple(p -> ntuple(i -> folded[i][2][p], newD), P)

    return AxisGroup(newlengths, newstrides)
end
