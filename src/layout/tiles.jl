# Tile axes and tiles. Addresses are zero-based (storage index = address + 1),
# coordinates one-based; a tile's `base` is the address its offsets are
# relative to. A tile axis is the vector of offsets of its rows or columns: an
# `AffineAxis`, a `ScatterAxis`, or any `AbstractVector{Int}`.

# Offsets `base + (t - 1) * stride`. Not a `StepRange`: the stride may be zero.
struct AffineAxis <: AbstractVector{Int}
    base::Int
    stride::Int
    count::Int

    function AffineAxis(base::Int, stride::Int, count::Int)
        count >= 0 || throw(ArgumentError("AffineAxis count must be nonnegative, got $count"))
        return new(base, stride, count)
    end
end

Base.size(ax::AffineAxis) = (ax.count,)
Base.IndexStyle(::Type{AffineAxis}) = IndexLinear()

@inline function Base.getindex(ax::AffineAxis, t::Int)
    @boundscheck checkbounds(ax, t)
    return ax.base + (t - 1) * ax.stride
end

function Base.extrema(ax::AffineAxis)
    isempty(ax) && throw(ArgumentError("extrema of an empty AffineAxis"))
    last = Base.Checked.checked_add(ax.base, Base.Checked.checked_mul(ax.count - 1, ax.stride))
    return minmax(ax.base, last)
end

# Offsets read from a borrowed buffer, valid while its owner is
# `GC.@preserve`d. A pointer rather than a `view`, so that it is `isbits` and a
# `Union{AffineAxis, ScatterAxis}` is never heap-boxed.
struct ScatterAxis <: AbstractVector{Int}
    offsets::Ptr{Int}
    count::Int
end

Base.size(ax::ScatterAxis) = (ax.count,)
Base.IndexStyle(::Type{ScatterAxis}) = IndexLinear()

@inline function Base.getindex(ax::ScatterAxis, t::Int)
    @boundscheck checkbounds(ax, t)
    return unsafe_load(ax.offsets, t)
end

# An offset interval of one map: `regular` iff `buffer[t+1] == base + t*stride`
# for all `t < count`; otherwise read the buffer (valid until it is refilled).
struct BlockDescriptor
    base::Int
    stride::Int
    count::Int
    regular::Bool
end

# Classifies `buffer[first+1 : first+count]`. A difference that overflows is
# irregular: a regular block's offset range is computed as `base + t*stride`.
function describe_block(buffer::Vector{Int}, first::Int, count::Int)
    first >= 0 || throw(ArgumentError("first must be nonnegative, got $first"))
    count >= 0 || throw(ArgumentError("count must be nonnegative, got $count"))
    count <= length(buffer) - first ||
        throw(DimensionMismatch("buffer length $(length(buffer)) is less than first + count = $first + $count"))

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

# Logical `(i, j)` addresses `base + rows[i] + cols[j]`.
struct Tile{S, R <: AbstractVector{Int}, C <: AbstractVector{Int}}
    storage::S
    base::Int
    rows::R
    cols::C
end

Base.size(tile::Tile) = (length(tile.rows), length(tile.cols))

Base.@propagate_inbounds Base.getindex(tile::Tile, i::Int, j::Int) =
    tile.storage[tile.base + tile.rows[i] + tile.cols[j] + 1]

Base.@propagate_inbounds function Base.setindex!(tile::Tile, v, i::Int, j::Int)
    tile.storage[tile.base + tile.rows[i] + tile.cols[j] + 1] = v
    return tile
end

# Bounds checks run once per tile (or per macro block) before the unchecked
# hot paths. Offset ranges are `(lo, hi)`, `(0, -1)` when empty.

# `extrema` of the axis `d` describes, without materializing it.
function descriptor_offset_range(d::BlockDescriptor, buffer::Vector{Int}, first::Int)
    d.count == 0 && return (0, -1)
    d.regular && return extrema(AffineAxis(d.base, d.stride, d.count))
    return extrema(@view buffer[(first + 1):(first + d.count)])
end

# Exact, not conservative, for any rectangular (row-set x column-set) region:
# `rlo + clo` and `rhi + chi` are both realized addresses. This is what lets a
# whole macro block be checked once instead of per sliver.
function checked_span_bounds(
        base::Int, rows::Tuple{Int, Int}, cols::Tuple{Int, Int}, storage_length::Int
    )
    (rlo, rhi) = rows
    (clo, chi) = cols
    (rhi < rlo || chi < clo) && return nothing
    lo128 = Int128(base) + Int128(rlo) + Int128(clo)
    hi128 = Int128(base) + Int128(rhi) + Int128(chi)
    (lo128 >= 0 && hi128 <= Int128(storage_length - 1)) ||
        throw(BoundsError("tile addresses [$lo128, $hi128] exceed storage bounds [0, $(storage_length - 1)]", base))
    return nothing
end

function checked_tile_storage_bounds(
        base::Int, rows::AbstractVector{Int}, cols::AbstractVector{Int}, storage_length::Int
    )
    (isempty(rows) || isempty(cols)) && return nothing
    return checked_span_bounds(base, extrema(rows), extrema(cols), storage_length)
end

checked_tile_storage_bounds(tile::Tile) =
    checked_tile_storage_bounds(tile.base, tile.rows, tile.cols, length(tile.storage))

# Whether an axis steps through storage one element at a time.
is_unit_stride(ax::AffineAxis) = ax.stride == 1
is_unit_stride(::AbstractVector{Int}) = false
