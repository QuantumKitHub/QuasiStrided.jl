# Tile axes and tiles. Coordinates and addresses are zero-based (storage index
# = address + 1); a tile's `base` is the address of its logical origin.

# `t -> base + t*stride`.
struct AffineAxis
    base::Int
    stride::Int
    count::Int

    function AffineAxis(base::Int, stride::Int, count::Int)
        count >= 0 || throw(ArgumentError("AffineAxis count must be nonnegative, got $count"))
        return new(base, stride, count)
    end
end

# `t -> offsets[t+1]`, borrowing `offsets`.
struct ScatterAxis{V <: AbstractVector{Int}}
    offsets::V
    count::Int

    function ScatterAxis(offsets::V, count::Int) where {V <: AbstractVector{Int}}
        count >= 0 || throw(ArgumentError("ScatterAxis count must be nonnegative, got $count"))
        count <= length(offsets) ||
            throw(
            DimensionMismatch(
                "ScatterAxis: offsets has length $(length(offsets)), " *
                    "need at least count = $count"
            )
        )
        return new{V}(offsets, count)
    end
end

# `ScatterAxis` over a borrowed raw pointer, so that it is `isbits` and
# `Union{AffineAxis, PtrScatterAxis}` needs no heap box in the driver.
struct PtrScatterAxis
    offsets::Ptr{Int}
    count::Int

    function PtrScatterAxis(offsets::Ptr{Int}, count::Int)
        count >= 0 ||
            throw(ArgumentError("PtrScatterAxis count must be nonnegative, got $count"))
        return new(offsets, count)
    end
end

const Axis = Union{AffineAxis, ScatterAxis, PtrScatterAxis}

axis_length(ax::AffineAxis) = ax.count
axis_length(ax::ScatterAxis) = ax.count
axis_length(ax::PtrScatterAxis) = ax.count

@inline axis_offset(ax::AffineAxis, t::Int) = ax.base + t * ax.stride
@inline axis_offset(ax::ScatterAxis, t::Int) = @inbounds ax.offsets[t + 1]
@inline axis_offset(ax::PtrScatterAxis, t::Int) =
    unsafe_load(ax.offsets + sizeof(Int) * t)

function axis_from_descriptor(descriptor::BlockDescriptor, buffer::Vector{Int}, first::Int)
    if descriptor.regular
        return AffineAxis(descriptor.base, descriptor.stride, descriptor.count)
    else
        return ScatterAxis(view(buffer, (first + 1):(first + descriptor.count)), descriptor.count)
    end
end

axis_from_descriptor(descriptor::BlockDescriptor, buffer::Vector{Int}) =
    axis_from_descriptor(descriptor, buffer, 0)

# Logical `(i, j)` addresses `base + row_offset(i) + col_offset(j)`.
struct QSTile{S, R <: Axis, C <: Axis}
    storage::S
    base::Int
    rows::R
    cols::C
end

const SourceTile = QSTile
const DestinationTile = QSTile

nrows(tile::QSTile) = axis_length(tile.rows)
ncols(tile::QSTile) = axis_length(tile.cols)

@inline function tile_offset(tile::QSTile, i::Int, j::Int)
    return tile.base + axis_offset(tile.rows, i) + axis_offset(tile.cols, j)
end

@inline function tile_load(tile::QSTile, i::Int, j::Int)
    return @inbounds tile.storage[tile_offset(tile, i, j) + 1]
end

@inline function tile_store!(tile::QSTile, i::Int, j::Int, v)
    @inbounds tile.storage[tile_offset(tile, i, j) + 1] = v
    return tile
end

# Bounds checks run once per tile (or per macro block) before the unchecked
# hot paths. Offset ranges are `(lo, hi)`, `(0, -1)` when empty.

function axis_offset_range(ax::AffineAxis)
    ax.count == 0 && return (0, -1)
    lo128 = Int128(ax.base)
    hi128 = Int128(ax.base) + Int128(ax.count - 1) * Int128(ax.stride)
    lo128, hi128 = minmax(lo128, hi128)
    (typemin(Int) <= lo128 && hi128 <= typemax(Int)) ||
        throw(OverflowError("axis_offset_range: affine axis range not representable as Int"))
    return (Int(lo128), Int(hi128))
end

function axis_offset_range(ax::ScatterAxis)
    ax.count == 0 && return (0, -1)
    prefix = view(ax.offsets, 1:ax.count)
    return (Int(minimum(prefix)), Int(maximum(prefix)))
end

function axis_offset_range(ax::PtrScatterAxis)
    ax.count == 0 && return (0, -1)
    lo = hi = unsafe_load(ax.offsets)
    for t in 1:(ax.count - 1)
        v = axis_offset(ax, t)
        lo, hi = min(lo, v), max(hi, v)
    end
    return (lo, hi)
end

# `axis_offset_range` of the axis `d` describes, without materializing it.
function descriptor_offset_range(d::BlockDescriptor, buffer::Vector{Int}, first::Int)
    d.count == 0 && return (0, -1)
    d.regular && return axis_offset_range(AffineAxis(d.base, d.stride, d.count))
    lo = hi = buffer[first + 1]
    for t in 1:(d.count - 1)
        v = buffer[first + t + 1]
        lo = min(lo, v)
        hi = max(hi, v)
    end
    return (lo, hi)
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

function checked_tile_storage_bounds(base::Int, rows::Axis, cols::Axis, storage_length::Int)
    (axis_length(rows) == 0 || axis_length(cols) == 0) && return nothing
    return checked_span_bounds(
        base, axis_offset_range(rows), axis_offset_range(cols), storage_length
    )
end

checked_tile_storage_bounds(tile::QSTile) =
    checked_tile_storage_bounds(tile.base, tile.rows, tile.cols, length(tile.storage))

# Whether an axis steps through storage one element at a time. Deliberately no
# fallback method: an unknown axis type must be a MethodError, not `false`.
_unit_stride_rows(ax::AffineAxis) = ax.stride == 1
_unit_stride_rows(::ScatterAxis) = false
_unit_stride_rows(::PtrScatterAxis) = false
