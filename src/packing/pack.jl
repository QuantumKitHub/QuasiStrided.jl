# Packers: copy one sliver of A (up to MR logical rows) or B (up to NR logical
# columns) over a K range into a contiguous panel in the descriptor's format.

# A runtime check rather than dispatch, so a mismatch is an ArgumentError.
@inline function _check_packed_eltype(packed, kernel::Descriptor{MR, NR, T2}) where {MR, NR, T2}
    R = realtype(kernel)
    eltype(packed) === R && return nothing
    msg = R === T2 ?
        "packed buffer eltype $(eltype(packed)) does not match kernel scalar type $T2" :
        "packed buffer eltype $(eltype(packed)) does not match kernel real type $R " *
        "(scalar type $T2)"
    throw(ArgumentError(msg))
end

# Out of line so the packers carry no string formatting (and no GC frame).
@noinline _throw_pack_extent(which::Symbol, name::Symbol, got::Int, limit::Int) = throw(
    ArgumentError(
        "$(which)!: source $(name) count $got must satisfy 0 <= $got <= $(which === :pack_a ? "mr" : "nr")(kernel)=$limit"
    )
)
@noinline _throw_pack_short(which::Symbol, got::Int, need::Int, kc::Int) = throw(
    DimensionMismatch(
        "$(which)!: packed buffer has length $got, " *
            "need at least packed_$(which === :pack_a ? "a" : "b")_length(kernel, kc=$kc) = $need"
    )
)

# `transform` is applied to each loaded (complex) element before it is split
# into the packed format; padding lanes are literal zeros and never read the
# source or call `transform`. All validation happens before any write.
function pack_a!(
        packed::V, source::QSTile, kernel::Descriptor{MR, NR, T2, FA, FB},
        transform::F
    ) where {V, MR, NR, T2, FA, FB, F}
    return _pack_a!(packed, source, kernel, transform, Val(true))
end

function pack_b!(
        packed::V, source::QSTile, kernel::Descriptor{MR, NR, T2, FA, FB},
        transform::F
    ) where {V, MR, NR, T2, FA, FB, F}
    return _pack_b!(packed, source, kernel, transform, Val(true))
end

# Skips only `checked_tile_storage_bounds(source)`: the caller (`_execute_nest!`)
# has already validated the whole macro block the sliver belongs to. `@inline`
# because out of line each call marshals the `QSTile` through the stack.
@inline function unsafe_pack_a!(
        packed::V, source::QSTile, kernel::Descriptor{MR, NR, T2, FA, FB},
        transform::F
    ) where {V, MR, NR, T2, FA, FB, F}
    return _pack_a!(packed, source, kernel, transform, Val(false))
end

@inline function unsafe_pack_b!(
        packed::V, source::QSTile, kernel::Descriptor{MR, NR, T2, FA, FB},
        transform::F
    ) where {V, MR, NR, T2, FA, FB, F}
    return _pack_b!(packed, source, kernel, transform, Val(false))
end

@inline function _pack_a!(
        packed::V, source::QSTile, kernel::Descriptor{MR, NR, T2, FA, FB},
        transform::F, ::Val{BOUNDS}
    ) where {V, MR, NR, T2, FA, FB, F, BOUNDS}
    _check_packed_eltype(packed, kernel)
    m = nrows(source)
    kc = ncols(source)
    (0 <= m <= MR) || _throw_pack_extent(:pack_a, :row, m, MR)
    needed = packed_a_length(kernel, kc)
    length(packed) >= needed || _throw_pack_short(:pack_a, length(packed), needed, kc)
    # `<= 0`, not `== 0` (counts are never negative): LLVM may then assume
    # `kc >= 1` in the loops below.
    kc <= 0 && return packed
    BOUNDS && checked_tile_storage_bounds(source)
    return _pack_a_sliver!(FA(), packed, source, kernel, transform, m, kc)
end

@inline function _pack_b!(
        packed::V, source::QSTile, kernel::Descriptor{MR, NR, T2, FA, FB},
        transform::F, ::Val{BOUNDS}
    ) where {V, MR, NR, T2, FA, FB, F, BOUNDS}
    _check_packed_eltype(packed, kernel)
    kc = nrows(source)
    n = ncols(source)
    (0 <= n <= NR) || _throw_pack_extent(:pack_b, :column, n, NR)
    needed = packed_b_length(kernel, kc)
    length(packed) >= needed || _throw_pack_short(:pack_b, length(packed), needed, kc)
    kc <= 0 && return packed
    BOUNDS && checked_tile_storage_bounds(source)
    return _pack_b_sliver!(FB(), packed, source, kernel, transform, n, kc)
end

# A's packed index runs along `source.rows`, B's along `source.cols`; each
# contiguous fast path needs that lane axis to be unit-stride.
@inline function _pack_a_sliver!(
        format::FMT, packed::V, source::QSTile, kernel::Descriptor{MR, NR, T2},
        transform::F, m::Int, kc::Int
    ) where {FMT, V, MR, NR, T2, F}
    if format isa RealFormat
        if _pack_a_contiguous_eligible(packed, source, transform, m, Val(MR), real(T2))
            rowbase = source.base + source.rows.base
            return _pack_a_contiguous!(packed, source.storage, rowbase, source.cols, Val(MR), kc)
        end
    elseif _pack_complex_contiguous_eligible(
            packed, source.storage, source.rows, transform, format, m, Val(MR), T2
        )
        elembase = source.base + source.rows.base
        return _pack_complex_contiguous!(
            format, packed, source.storage, elembase, source.cols, Val(MR), kc, transform
        )
    end
    load = (i, p) -> tile_load(source, i, p)
    plane_offset = (plane, i, p) -> packed_a_plane_offset(kernel, plane, i, p)
    return _pack_panel!(packed, _element_type(format, T2), format, Val(MR), kc, m, transform, load, plane_offset)
end

@inline function _pack_b_sliver!(
        format::FMT, packed::V, source::QSTile, kernel::Descriptor{MR, NR, T2},
        transform::F, n::Int, kc::Int
    ) where {FMT, V, MR, NR, T2, F}
    if _pack_complex_contiguous_eligible(
            packed, source.storage, source.cols, transform, format, n, Val(NR), T2
        )
        elembase = source.base + source.cols.base
        return _pack_complex_contiguous!(
            format, packed, source.storage, elembase, source.rows, Val(NR), kc, transform
        )
    end
    load = (j, p) -> tile_load(source, p, j)
    plane_offset = (plane, j, p) -> packed_b_plane_offset(kernel, plane, j, p)
    return _pack_panel!(packed, _element_type(format, T2), format, Val(NR), kc, n, transform, load, plane_offset)
end

# What a packed element converts to: the real operand of a mixed-domain kernel
# packs `real(T)`.
@inline _element_type(::RealFormat, ::Type{T}) where {T} = real(T)
@inline _element_type(::PackFormat, ::Type{T}) where {T} = T

# One sliver over `kc` K steps; `PD` (MR or NR) is a compile-time constant. A
# full sliver gets a constant-trip inner loop that LLVM fully unrolls; a tail
# writes its valid lanes and its padding as two loops. A per-lane
# `t < valid ? load : zero` would compile to a branch around the load and keep
# the loop scalar.
@inline function _pack_panel!(
        packed::V, ::Type{T}, format::FMT, ::Val{PD}, kc::Int, valid::Int,
        transform::F, load::L, plane_offset::P
    ) where {V, T, FMT <: PackFormat, PD, F, L, P}
    if valid == PD
        @inbounds for p in 0:(kc - 1)
            for t in 0:(PD - 1)
                z = convert(T, transform(load(t, p)))::T
                _pack_emit!(packed, format, plane_offset, t, p, z)
            end
        end
    else
        @inbounds for p in 0:(kc - 1)
            for t in 0:(valid - 1)
                z = convert(T, transform(load(t, p)))::T
                _pack_emit!(packed, format, plane_offset, t, p, z)
            end
            for t in valid:(PD - 1)
                _pack_emit_zero!(packed, format, plane_offset, t, p, real(T))
            end
        end
    end
    return packed
end

# Per-lane stores of one element. `real`/`imag` on the loaded element are the
# only accessors: a `QSTile` may address scattered storage, so the source is
# never `reinterpret`ed.
@inline function _pack_emit!(
        packed::V, ::RealFormat, plane_offset::P, t::Int, p::Int, z::T
    ) where {V, P, T}
    panel_store!(packed, plane_offset(0, t, p), z)
    return nothing
end

@inline function _pack_emit!(
        packed::V, ::PlanarFormat, plane_offset::P, t::Int, p::Int, z::T
    ) where {V, P, T}
    panel_store!(packed, plane_offset(0, t, p), real(z))
    panel_store!(packed, plane_offset(1, t, p), imag(z))
    return nothing
end

@inline function _pack_emit!(
        packed::V, ::InterleavedFormat, plane_offset::P, t::Int, p::Int, z::T
    ) where {V, P, T}
    panel_store!(packed, plane_offset(0, 2 * t, p), real(z))
    panel_store!(packed, plane_offset(0, 2 * t + 1, p), imag(z))
    return nothing
end

@inline function _pack_emit!(
        packed::V, ::OneEFormat, plane_offset::P, t::Int, p::Int, z::T
    ) where {V, P, T}
    re = real(z)
    im = imag(z)
    panel_store!(packed, plane_offset(0, 2 * t, p), re)
    panel_store!(packed, plane_offset(0, 2 * t + 1, p), im)
    panel_store!(packed, plane_offset(2, 2 * t, p), -im)
    panel_store!(packed, plane_offset(2, 2 * t + 1, p), re)
    return nothing
end

# Padding stores literal zeros; `_pack_emit!(.., zero(T))` would write 1e's
# `-im` as `-0.0`.
@inline function _pack_emit_zero!(
        packed::V, ::RealFormat, plane_offset::P, t::Int, p::Int, ::Type{R}
    ) where {V, P, R}
    panel_store!(packed, plane_offset(0, t, p), zero(R))
    return nothing
end

@inline function _pack_emit_zero!(
        packed::V, ::PlanarFormat, plane_offset::P, t::Int, p::Int, ::Type{R}
    ) where {V, P, R}
    panel_store!(packed, plane_offset(0, t, p), zero(R))
    panel_store!(packed, plane_offset(1, t, p), zero(R))
    return nothing
end

@inline function _pack_emit_zero!(
        packed::V, ::InterleavedFormat, plane_offset::P, t::Int, p::Int, ::Type{R}
    ) where {V, P, R}
    panel_store!(packed, plane_offset(0, 2 * t, p), zero(R))
    panel_store!(packed, plane_offset(0, 2 * t + 1, p), zero(R))
    return nothing
end

@inline function _pack_emit_zero!(
        packed::V, ::OneEFormat, plane_offset::P, t::Int, p::Int, ::Type{R}
    ) where {V, P, R}
    panel_store!(packed, plane_offset(0, 2 * t, p), zero(R))
    panel_store!(packed, plane_offset(0, 2 * t + 1, p), zero(R))
    panel_store!(packed, plane_offset(2, 2 * t, p), zero(R))
    panel_store!(packed, plane_offset(2, 2 * t + 1, p), zero(R))
    return nothing
end
