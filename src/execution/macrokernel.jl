# Macro-kernel helpers for the five-loop nest: function barriers over the
# tile axis types, packed-sliver addressing, and sliver classification.

# The axis `d` describes. GUARDRAIL: a `Union{AffineAxis, ScatterAxis}` of
# `isbits` types, never boxed; each consumer below binds each axis type as its
# own parameter, so it builds a concretely typed `Tile`.
@inline _axis_of(d::BlockDescriptor, buffer::Vector{Int}, first::Int) =
    d.regular ? AffineAxis(d.base, d.stride, d.count) : ScatterAxis(pointer(buffer, first + 1), d.count)

# The same with the axis type fixed by the path: `Val(true)` for a ramp map,
# whose descriptors are always regular (checked).
@inline _axis_of(d::BlockDescriptor, buffer::Vector{Int}, first::Int, ::Val{false}) =
    _axis_of(d, buffer, first)
@inline function _axis_of(d::BlockDescriptor, ::Vector{Int}, ::Int, ::Val{true})
    d.regular || _throw_irregular_ramp_descriptor()
    return AffineAxis(d.base, d.stride, d.count)
end
@noinline _throw_irregular_ramp_descriptor() =
    throw(AssertionError("an affine-ramp map produced an irregular block descriptor"))

# B passes its axes swapped (its `transpose`). GUARDRAIL: `transform` needs its
# own bound type parameter, or it costs a dynamic dispatch per call.
Base.@propagate_inbounds function _pack_sliver!(
        packed::PK, storage::S, base::Int,
        lanes::R, steps::C, spec, transform::TF
    ) where {PK, S, R <: AbstractVector{Int}, C <: AbstractVector{Int}, TF}
    pack!(packed, Tile(storage, base, lanes, steps), spec, transform)
    return nothing
end

Base.@propagate_inbounds function _execute_micro_tile!(
        kernel, storage::S, base::Int, rows::R, cols::C,
        packed_a::PA, packed_b::PB, k_block_length::Int, alpha, beta
    ) where {PA, PB, S, R <: AbstractVector{Int}, C <: AbstractVector{Int}}
    destination = Tile(storage, base, rows, cols)
    execute_tile!(kernel, destination, packed_a, packed_b, k_block_length, alpha, beta)
    return nothing
end

@inline function _scale_micro_tile!(
        storage::S, base::Int, rows::R, cols::C, beta
    ) where {S, R <: AbstractVector{Int}, C <: AbstractVector{Int}}
    destination = Tile(storage, base, rows, cols)
    scale_tile!(destination, beta)
    return nothing
end

# Sliver `tile_index` of a packed panel at the current block's depth
# `k_block_length`, shared by the packing and the consuming step so the two
# cannot disagree. GUARDRAIL: `width` counts reals per K step
# (`sliver_width`), not the tile size; they differ for complex kernels.
@inline function _sliver_panel(buffer, width::Int, k_block_length::Int, tile_index::Int)
    stride = width * k_block_length
    return packed_panel(buffer, tile_index * stride + 1, stride)
end

# Classify each register sliver of a just-filled macro block, and return the
# two maps' offset ranges over the whole block (the slivers partition it) for
# the hoisted bounds checks.
@inline function _classify_slivers!(g::GroupBuffers, block_length::Int, tile::Int, tile_count::Int)
    desc1, desc2 = g.descriptors
    buf1, buf2 = g.offsets
    lo1 = typemax(Int); hi1 = typemin(Int)
    lo2 = typemax(Int); hi2 = typemin(Int)
    for tile_index in 0:(tile_count - 1)
        tile_start = tile_index * tile
        tile_length = min(tile, block_length - tile_start)
        d1 = describe_block(buf1, tile_start, tile_length)
        d2 = describe_block(buf2, tile_start, tile_length)
        desc1[tile_index + 1] = d1
        desc2[tile_index + 1] = d2
        (l1, h1) = descriptor_offset_range(d1, buf1, tile_start)
        if h1 >= l1
            lo1 = min(lo1, l1); hi1 = max(hi1, h1)
        end
        (l2, h2) = descriptor_offset_range(d2, buf2, tile_start)
        if h2 >= l2
            lo2 = min(lo2, l2); hi2 = max(hi2, h2)
        end
    end
    # `hi < lo` means empty to `checked_span_bounds`.
    return ((lo1, hi1), (lo2, hi2))
end

# Closed-form block descriptors for an affine-ramp composite (offset of `q`
# is `q * step`), with no offset buffer or scan. `==` to what `describe_block`
# gives for the materialized interval, including its `stride == 0` for a
# count-1 block.
@inline _ramp_descriptor(step::Int, first::Int, count::Int) =
    count == 0 ? BlockDescriptor(0, 0, 0, true) :
    count == 1 ? BlockDescriptor(first * step, 0, 1, true) :
    BlockDescriptor(first * step, step, count, true)

# Offset range of `[first, first+count)`; `(0, -1)` if empty. Cannot overflow:
# `AxisGroup` validated every in-domain offset at construction.
@inline _ramp_offset_range(step::Int, first::Int, count::Int) =
    count == 0 ? (0, -1) : minmax(first * step, (first + count - 1) * step)

# `_classify_slivers!`'s closed-form twin.
@inline function _ramp_slivers!(
        g::GroupBuffers, step1::Int, step2::Int, first::Int,
        block_length::Int, tile::Int, tile_count::Int
    )
    desc1, desc2 = g.descriptors
    for tile_index in 0:(tile_count - 1)
        tile_start = tile_index * tile
        tile_length = min(tile, block_length - tile_start)
        q0 = first + tile_start
        desc1[tile_index + 1] = _ramp_descriptor(step1, q0, tile_length)
        desc2[tile_index + 1] = _ramp_descriptor(step2, q0, tile_length)
    end
    return (
        _ramp_offset_range(step1, first, block_length),
        _ramp_offset_range(step2, first, block_length),
    )
end
