# The packer: copies one sliver, up to `L` lanes along `tile.rows` by the K
# steps along `tile.cols`, into a contiguous panel in the spec's format. A is
# packed as its tile, B as `transpose` of its tile.

operand_name(i::Int) = i == 1 ? "A" : "B"

# A runtime check rather than dispatch, so a mismatch is an ArgumentError.
@inline function check_packed_eltype(panel, spec::SliverSpec{I, L, F, T}) where {I, L, F, T}
    eltype(panel) === real(T) || throw_packed_eltype(eltype(panel), T)
    return nothing
end

# Out of line so the packer carries no string formatting (and no GC frame).
@noinline throw_packed_eltype(got::Type, ::Type{T}) where {T} = throw(
    ArgumentError(
        real(T) === T ?
            "packed buffer eltype $got does not match kernel scalar type $T" :
            "packed buffer eltype $got does not match kernel real type $(real(T)) (scalar type $T)"
    )
)
@noinline throw_pack_extent(i::Int, got::Int, limit::Int) = throw(
    ArgumentError(
        "pack!: $(operand_name(i)) sliver has $got lanes, need 0 <= $got <= tile_size(kernel, $i) = $limit"
    )
)
@noinline throw_pack_short(i::Int, got::Int, need::Int, k_block_length::Int) = throw(
    DimensionMismatch(
        "pack!: packed $(operand_name(i)) buffer has length $got, need at least " *
            "packed_$(i == 1 ? "a" : "b")_length(kernel, k_block_length=$k_block_length) = $need"
    )
)

# `transform` is applied to each loaded (complex) element before it is split
# into the packed format; padding lanes are literal zeros and never read the
# source or call `transform`. All validation happens before any write.
pack!(panel::V, tile::Tile, spec::SliverSpec, transform::F) where {V, F} =
    pack_tile!(panel, tile, spec, transform, Val(true))

# The packer writes only `PackedPanel`s; a `DenseVector` is borrowed as one.
function pack!(panel::V, tile::Tile, spec::SliverSpec, transform::F) where {V <: DenseVector, F}
    GC.@preserve panel pack!(packed_panel(panel, 1, length(panel)), tile, spec, transform)
    return panel
end

# Skips only `checked_tile_storage_bounds(tile)`: the caller (`_execute_nest!`)
# has already validated the whole macro block the sliver belongs to. `@inline`
# because out of line each call marshals the `Tile` through the stack.
@inline unsafe_pack!(panel::V, tile::Tile, spec::SliverSpec, transform::F) where {V, F} =
    pack_tile!(panel, tile, spec, transform, Val(false))

@inline function pack_tile!(
        panel::V, tile::Tile, spec::SliverSpec{I, L}, transform::F, ::Val{BOUNDS}
    ) where {V, I, L, F, BOUNDS}
    check_packed_eltype(panel, spec)
    lanes, k_block_length = size(tile)
    (0 <= lanes <= L) || throw_pack_extent(I, lanes, L)
    needed = packed_length(spec, k_block_length)
    length(panel) >= needed || throw_pack_short(I, length(panel), needed, k_block_length)
    # `<= 0`, not `== 0` (counts are never negative): LLVM may then assume
    # `k_block_length >= 1` in the loops below.
    k_block_length <= 0 && return panel
    BOUNDS && checked_tile_storage_bounds(tile)
    if real_contiguous_eligible(tile, spec, transform, lanes)
        return pack_real_contiguous!(panel, tile, spec, k_block_length)
    elseif complex_contiguous_eligible(tile, spec, transform, lanes)
        return pack_complex_contiguous!(panel, tile, spec, k_block_length, transform)
    end
    return pack_scalar!(panel, tile, spec, transform, lanes, k_block_length)
end

# What a packed element converts to: the real operand of a mixed-domain kernel
# packs `real(T)`.
element_type(::SliverSpec{I, L, RealFormat, T}) where {I, L, T} = real(T)
element_type(::SliverSpec{I, L, F, T}) where {I, L, F, T} = T

# A full sliver gets a constant-trip inner loop that LLVM fully unrolls; a tail
# writes its valid lanes and its padding as two loops. A per-lane
# `t < lanes ? load : zero` would compile to a branch around the load and keep
# the loop scalar.
@inline function pack_scalar!(
        panel::V, tile::Tile, spec::SliverSpec{I, L}, transform::F, lanes::Int, k_block_length::Int
    ) where {V, I, L, F}
    T = element_type(spec)
    if lanes == L
        @inbounds for p in 1:k_block_length
            for t in 1:L
                emit_value!(panel, spec, 0, t, p, convert(T, transform(tile[t, p]))::T)
            end
        end
    else
        @inbounds for p in 1:k_block_length
            for t in 1:lanes
                emit_value!(panel, spec, 0, t, p, convert(T, transform(tile[t, p]))::T)
            end
            for t in (lanes + 1):L
                emit_padding!(panel, spec, 0, t, p)
            end
        end
    end
    return panel
end

# Stores of lane `t` at K step `p`, `base` reals into `panel`. `real`/`imag`
# on the loaded element are the only accessors: a `Tile` may address scattered
# storage, so the source is never `reinterpret`ed.
@inline emit_value!(panel, spec::SliverSpec, base::Int, t::Int, p::Int, z) =
    emit!(panel, spec, base, t, p, real(z), imag(z), -imag(z))
# Literal zeros: the value path would write 1e's `-im` as `-0.0`.
@inline function emit_padding!(panel, spec::SliverSpec, base::Int, t::Int, p::Int)
    z = zero(realtype(spec))
    return emit!(panel, spec, base, t, p, z, z, z)
end

@inline function emit!(panel, spec::SliverSpec{I, L, RealFormat}, base, t, p, re, im, neg_im) where {I, L}
    panel_store!(panel, base + panel_offset(spec, t, p), re)
    return nothing
end

@inline function emit!(panel, spec::SliverSpec{I, L, PlanarFormat}, base, t, p, re, im, neg_im) where {I, L}
    panel_store!(panel, base + panel_offset(spec, t, p), re)
    panel_store!(panel, base + panel_offset(spec, t, p, 1), im)
    return nothing
end

@inline function emit!(panel, spec::SliverSpec{I, L, InterleavedFormat}, base, t, p, re, im, neg_im) where {I, L}
    panel_store!(panel, base + panel_offset(spec, 2t - 1, p), re)
    panel_store!(panel, base + panel_offset(spec, 2t, p), im)
    return nothing
end

@inline function emit!(panel, spec::SliverSpec{I, L, OneEFormat}, base, t, p, re, im, neg_im) where {I, L}
    panel_store!(panel, base + panel_offset(spec, 2t - 1, p), re)
    panel_store!(panel, base + panel_offset(spec, 2t, p), im)
    panel_store!(panel, base + panel_offset(spec, 2t - 1, p, 2), neg_im)
    panel_store!(panel, base + panel_offset(spec, 2t, p, 2), re)
    return nothing
end
