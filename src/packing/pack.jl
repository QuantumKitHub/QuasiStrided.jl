# Packers: copy one sliver of A (up to MR logical rows) or B (up to NR logical
# columns) over a K range into a contiguous panel in the descriptor's packed
# format. `pack_a!`/`pack_b!` validate and dispatch on the operand's
# `PackFormat`; each format has its own leaf loop, and a contiguous fast path
# (src/packing/pack_contiguous.jl) for the one sliver shape it can serve.

# `Vec`/`vload`/`vstore` already arrive via src/packing/panel.jl's `using`; the
# complex fast path additionally needs the compile-time-index shuffle (SIMD.jl
# v3), which is one LLVM `shufflevector` instruction and never a runtime gather.
using SIMD: shufflevector

# Explicit runtime check (not dispatch) so a mismatch raises ArgumentError. The
# packed buffer holds `realtype(kernel)`, which is *not* `scalartype(kernel)`
# once the element type is complex.
@inline function _check_packed_eltype(packed, kernel::Descriptor{MR, NR, T2}) where {MR, NR, T2}
    R = realtype(kernel)
    eltype(packed) === R && return nothing
    msg = R === T2 ?
        "packed buffer eltype $(eltype(packed)) does not match kernel scalar type $T2" :
        "packed buffer eltype $(eltype(packed)) does not match kernel real type $R " *
        "(scalar type $T2)"
    throw(ArgumentError(msg))
end

# ----------------------------------------------------------------------------
# Shared argument validation for all four pack_a!/pack_b! methods
# ----------------------------------------------------------------------------
# Extent bound, nonnegative `kc`, packed capacity, then the
# one-time storage-bounds check before any `@inbounds` loop. Returns
# `(valid, kc)`; `kc == 0` is the no-op the caller returns from. One bound type
# parameter per argument, as at `pack_a!` in src/microkernels/interface.jl.
#
# `packed_a_length`/`packed_b_length` count ELEMENTS for a real descriptor and
# REALS for a complex one, at the logical `kc` in both cases, so the same check
# serves both without knowing which it has.

# `BOUNDS` is a compile-time flag, not a runtime one: at `Val(true)` the body
# below runs every check, and at `Val(false)` the `checked_tile_storage_bounds`
# call is folded away entirely. Only
# `unsafe_pack_a!`/`unsafe_pack_b!` ever pass `Val(false)`, and only from a
# caller that has already validated the WHOLE macro block this sliver belongs
# to (src/execution/execute.jl, `_execute_nest!`). Every other check -- extents, the
# packed-buffer capacity, the `kc == 0` no-op -- is kept in both modes.
@inline _check_pack_a(packed::V, source::QSTile, kernel::K) where {V, K} =
    _check_pack_a(packed, source, kernel, Val(true))

# The throws live behind `@noinline` helpers so that the packers -- entered
# once per sliver -- carry no inline string formatting (and hence no GC frame);
# see `_throw_tile_extent` in src/microkernels/interface.jl for the reason.
@noinline _throw_pack_extent(which::Symbol, name::Symbol, got::Int, limit::Int) = throw(
    ArgumentError(
        "$(which)!: source $(name) count $got must satisfy 0 <= $got <= $(which === :pack_a ? "mr" : "nr")(kernel)=$limit"
    )
)
@noinline _throw_pack_kc(which::Symbol, kc::Int) =
    throw(ArgumentError("$(which)!: source K extent (kc) must be nonnegative, got $kc"))
@noinline _throw_pack_short(which::Symbol, got::Int, need::Int, kc::Int) = throw(
    DimensionMismatch(
        "$(which)!: packed buffer has length $got, " *
            "need at least packed_$(which === :pack_a ? "a" : "b")_length(kernel, kc=$kc) = $need"
    )
)

@inline function _check_pack_a(
        packed::V, source::QSTile, kernel::K, ::Val{BOUNDS}
    ) where {V, K, BOUNDS}
    MR = mr(kernel)
    m = nrows(source)
    kc = ncols(source)
    (0 <= m <= MR) || _throw_pack_extent(:pack_a, :row, m, MR)
    kc >= 0 || _throw_pack_kc(:pack_a, kc)
    needed = packed_a_length(kernel, kc)
    length(packed) >= needed || _throw_pack_short(:pack_a, length(packed), needed, kc)
    kc == 0 && return (m, 0)
    BOUNDS && checked_tile_storage_bounds(source)
    return (m, kc)
end

@inline _check_pack_b(packed::V, source::QSTile, kernel::K) where {V, K} =
    _check_pack_b(packed, source, kernel, Val(true))

@inline function _check_pack_b(
        packed::V, source::QSTile, kernel::K, ::Val{BOUNDS}
    ) where {V, K, BOUNDS}
    NR = nr(kernel)
    kc = nrows(source)
    n = ncols(source)
    kc >= 0 || _throw_pack_kc(:pack_b, kc)
    (0 <= n <= NR) || _throw_pack_extent(:pack_b, :column, n, NR)
    needed = packed_b_length(kernel, kc)
    length(packed) >= needed || _throw_pack_short(:pack_b, length(packed), needed, kc)
    kc == 0 && return (n, 0)
    BOUNDS && checked_tile_storage_bounds(source)
    return (n, kc)
end

# ----------------------------------------------------------------------------
# Entry points
# ----------------------------------------------------------------------------

"""
    pack_a!(packed, source::QSTile, kernel::Descriptor{MR,NR,T,FA,FB}, transform) -> packed

Pack an A source tile into `packed` (a buffer of `realtype(kernel)`: a
`Vector`, a `SubArray` sliver of a macro panel, or a [`PackedPanel`](@ref)) in
the physical format `FA`. `source` has `0 <= m <= mr(kernel)` **logical** rows
and `kc = ncols(source)` columns; `packed` needs `length >=
packed_a_length(kernel, kc)`, which counts reals at logical `kc`.

Row `i < m` commits `convert(T, transform(A[i,p]))` -- for a complex `T`,
`transform` is applied to the complex element and the result is then split
into the packed format, never applied per real half. Padding rows (`i >= m`)
write literal zeros into every real of the lane without reading `source` or
calling `transform`. `kc == 0` is a no-op. All validation happens before any
write. Never allocates.

For a real kernel the layout is `packed_a_offset(kernel, i, p) == i +
mr(kernel)*p`; see [`PlanarFormat`](@ref) and [`OneEFormat`](@ref) for the
complex ones. See [`unsafe_pack_a!`](@ref) for the sibling entry point that
skips the storage-bounds half of the validation.
"""
function pack_a!(
        packed::V, source::QSTile, kernel::Descriptor{MR, NR, T2, FA, FB},
        transform::F
    ) where {V, MR, NR, T2, FA, FB, F}
    return _pack_a!(packed, source, kernel, transform, Val(true))
end

"""
    unsafe_pack_a!(packed, source::QSTile, kernel, transform) -> packed

[`pack_a!`](@ref) **without** the `checked_tile_storage_bounds(source)` call.

PRECONDITION, which the caller must have established: every address `source`
can read -- `source.base + row_offset(i) + col_offset(p)` for `0 <= i <
nrows(source)`, `0 <= p < ncols(source)` -- lies in
`0:length(source.storage)-1`. Violating it is an out-of-bounds read through an
`@inbounds`/pointer path, not an exception.

Every other check `pack_a!` makes is still made here: the row-extent bound,
`kc >= 0`, the packed-buffer capacity, and the packed eltype. Only the
address-range check moves, and it moves to the caller.

The one caller in this package is `_execute_nest!`
(src/execution/execute.jl), which validates the union of an entire macro
block's slivers in a single [`checked_span_bounds`](@ref) call before packing
any of them -- an exactly equivalent test, because the block's slivers
partition its offset buffer and all of them share the same K axis, so the
block's offset range is the union of the slivers' and the check only ever
looks at range extremes.

`@inline` (as is `unsafe_pack_b!`): the driver calls these once per sliver,
and out of line each call marshals the `QSTile` through the stack.
"""
@inline function unsafe_pack_a!(
        packed::V, source::QSTile, kernel::Descriptor{MR, NR, T2, FA, FB},
        transform::F
    ) where {V, MR, NR, T2, FA, FB, F}
    return _pack_a!(packed, source, kernel, transform, Val(false))
end

"""
    pack_b!(packed, source::QSTile, kernel::Descriptor{MR,NR,T,FA,FB}, transform) -> packed

Pack a B source tile into `packed` in the physical format `FB`. `source` has
`kc = nrows(source)` rows and `0 <= n <= nr(kernel)` **logical** columns;
`packed` needs `length >= packed_b_length(kernel, kc)` reals. For a real
kernel the layout is `packed_b_offset(kernel, j, p) == j + nr(kernel)*p` (not
column-major). Same `transform`, padding, validation and allocation contract
as [`pack_a!`](@ref).
"""
function pack_b!(
        packed::V, source::QSTile, kernel::Descriptor{MR, NR, T2, FA, FB},
        transform::F
    ) where {V, MR, NR, T2, FA, FB, F}
    return _pack_b!(packed, source, kernel, transform, Val(true))
end

"""
    unsafe_pack_b!(packed, source::QSTile, kernel, transform) -> packed

[`pack_b!`](@ref) without the `checked_tile_storage_bounds(source)` call; the
B-side counterpart of [`unsafe_pack_a!`](@ref), with the same precondition
(the caller has validated every address `source` can read) and the same single
caller in this package.
"""
@inline function unsafe_pack_b!(
        packed::V, source::QSTile, kernel::Descriptor{MR, NR, T2, FA, FB},
        transform::F
    ) where {V, MR, NR, T2, FA, FB, F}
    return _pack_b!(packed, source, kernel, transform, Val(false))
end

@inline function _pack_a!(
        packed::V, source::QSTile, kernel::Descriptor{MR, NR, T2, FA, FB},
        transform::F, bounds::Val{BOUNDS}
    ) where {V, MR, NR, T2, FA, FB, F, BOUNDS}
    _check_packed_eltype(packed, kernel)
    m, kc = _check_pack_a(packed, source, kernel, bounds)
    kc == 0 && return packed
    return _pack_a_sliver!(FA(), packed, source, kernel, transform, m, kc)
end

@inline function _pack_b!(
        packed::V, source::QSTile, kernel::Descriptor{MR, NR, T2, FA, FB},
        transform::F, bounds::Val{BOUNDS}
    ) where {V, MR, NR, T2, FA, FB, F, BOUNDS}
    _check_packed_eltype(packed, kernel)
    n, kc = _check_pack_b(packed, source, kernel, bounds)
    kc == 0 && return packed
    return _pack_b_sliver!(FB(), packed, source, kernel, transform, n, kc)
end

# ----------------------------------------------------------------------------
# The leaf loop, shared by every format
# ----------------------------------------------------------------------------

# One sliver over `kc` K steps. `load`/`plane_offset` close over the
# operand-specific index mapping, and `format` dispatches the per-lane store
# through `_pack_emit!`/`_pack_emit_zero!`. `kc == 0` is handled by the caller.
#
# `PD` (the physical dim: MR or NR) is a compile-time constant. A full sliver
# (`valid == PD`) gets a constant-trip-count inner loop that LLVM fully
# unrolls; a tail sliver writes its valid lanes and then its zero padding as
# two separate loops. Either way no loop body holds a conditional load: a
# per-element `t < valid ? load : zero` compiles to a branch around the load,
# which blocks if-conversion and keeps the whole loop scalar. The per-K-step
# store order (0, 1, ..., PD-1) and the padding contract (padding lanes never
# read `source` and never call `transform`) hold for every format.
#
# `transform` applies to the loaded (complex, for a complex format) element
# BEFORE `_pack_emit!` splits it into planes; padding lanes go through
# `_pack_emit_zero!` instead, which writes literal zeros without calling
# `transform` (a `-0.0` hazard for `OneEFormat`'s `-im` plane otherwise).
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

# RealFormat: one real per element, no plane split. `plane_offset` is still
# called with a leading `plane` argument (always `0` here) so that every
# format's call sites share the same closure shape.
@inline function _pack_emit!(
        packed::V, ::RealFormat, plane_offset::P, t::Int, p::Int, z::T
    ) where {V, P, T}
    panel_store!(packed, plane_offset(0, t, p), z)
    return nothing
end

@inline function _pack_emit_zero!(
        packed::V, ::RealFormat, plane_offset::P, t::Int, p::Int, ::Type{R}
    ) where {V, P, R}
    panel_store!(packed, plane_offset(0, t, p), zero(R))
    return nothing
end

# The real sliver packers. A has a contiguous fast path; B does not.
@inline function _pack_a_sliver!(
        ::RealFormat, packed::V, source::QSTile, kernel::Descriptor{MR, NR, T2},
        transform::F, m::Int, kc::Int
    ) where {V, MR, NR, T2, F}
    if _pack_a_contiguous_eligible(packed, source, transform, m, Val(MR), T2)
        rowbase = source.base + source.rows.base
        return _pack_a_contiguous!(packed, source.storage, rowbase, source.cols, Val(MR), kc)
    end

    load = (i, p) -> tile_load(source, i, p)
    plane_offset = (plane, i, p) -> packed_a_offset(kernel, i, p)
    _pack_panel!(packed, T2, RealFormat(), Val(MR), kc, m, transform, load, plane_offset)
    return packed
end

@inline function _pack_b_sliver!(
        ::RealFormat, packed::V, source::QSTile, kernel::Descriptor{MR, NR, T2},
        transform::F, n::Int, kc::Int
    ) where {V, MR, NR, T2, F}
    load = (j, p) -> tile_load(source, p, j)  # source.rows=K, source.cols=N
    plane_offset = (plane, j, p) -> packed_b_offset(kernel, j, p)
    _pack_panel!(packed, T2, RealFormat(), Val(NR), kc, n, transform, load, plane_offset)
    return packed
end


# ===========================================================================
# Complex packing
#
# Everything below writes `real(T)` into the packed buffer. The `transform`
# contract, which holds for every format:
#
#   `transform` is applied to each loaded *source element* before it is
#   committed, in whatever physical format the buffer uses. A packer that
#   splits an element into planes must produce a result identical to applying
#   `transform` to the complex value and THEN splitting -- never per real half,
#   where `conj` would be a silent no-op. Padding lanes are never read and
#   never call `transform`.
#
# `real(z)`/`imag(z)` on the loaded element are the only accessors used; the
# source is never `reinterpret`ed, because a `QSTile` addresses arbitrary
# strided (possibly scattered) storage for which that would be unsound.
# ===========================================================================

# --- emit: one logical K step, one lane, one format ------------------------
#
# `plane_offset(plane, index, p)` is `packed_a_plane_offset`/
# `packed_b_plane_offset` for the operand being packed, i.e. exactly
# `p * per_k + plane * vr + index`.

# PlanarFormat ("1r"): `o[t] = re; o[vr + t] = im`.
@inline function _pack_emit!(
        packed::V, ::PlanarFormat, plane_offset::P, t::Int, p::Int, z::T
    ) where {V, P, T}
    panel_store!(packed, plane_offset(0, t, p), real(z))
    panel_store!(packed, plane_offset(1, t, p), imag(z))
    return nothing
end

# OneEFormat ("1e"): the real 2x2 block [[re, -im], [im, re]], stored as two
# real K steps of `2vr`:
#
#     o[2t] = re;        o[2t + 1] = im
#     o[2vr + 2t] = -im; o[2vr + 2t + 1] = re
#
# This does NOT fit the two-plane shape the plane-offset helpers describe: its
# four reals are two *real* K steps of doubled width, not four planes of `vr`.
# The linear formula still covers it exactly, so addressing through it keeps
# the layout written down once: `plane_offset(0, ...)` is the first real K step
# and `plane_offset(2, ...)` the second, since `plane * vr` at `plane = 2` is
# precisely the `2vr` stride between them. The `index` argument therefore runs
# over reals (`0:2vr-1`), not over logical rows -- the one place in this file
# where it does.
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

# InterleavedFormat: `o[2t] = re; o[2t + 1] = im` -- exactly 1e's first real K
# step and nothing else, addressed the same way (`index` over reals).
@inline function _pack_emit!(
        packed::V, ::InterleavedFormat, plane_offset::P, t::Int, p::Int, z::T
    ) where {V, P, T}
    panel_store!(packed, plane_offset(0, 2 * t, p), real(z))
    panel_store!(packed, plane_offset(0, 2 * t + 1, p), imag(z))
    return nothing
end

# Padding lanes: literal zero into every real of the lane, without reading
# `source` and without calling `transform`. Written separately rather than as
# `_pack_emit!(..., zero(T))` because 1e's `-im` of a zero is `-0.0`, and the
# contract says padding writes a literal zero.
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


# The complex sliver packers, for any complex format.
@inline function _pack_a_sliver!(
        format::FMT, packed::V, source::QSTile, kernel::Descriptor{MR, NR, T2},
        transform::F, m::Int, kc::Int
    ) where {FMT <: Union{PlanarFormat, InterleavedFormat, OneEFormat}, V, MR, NR, T2, F}
    # A's packed index runs along `source.rows` (the MR logical rows).
    if _pack_complex_contiguous_eligible(
            packed, source.storage, source.rows, transform, format, m, Val(MR), T2
        )
        elembase = source.base + source.rows.base
        return _pack_complex_contiguous!(
            format, packed, source.storage, elembase, source.cols, Val(MR), kc, transform
        )
    end

    load = (i, p) -> tile_load(source, i, p)
    plane_offset = (plane, i, p) -> packed_a_plane_offset(kernel, plane, i, p)
    _pack_panel!(packed, T2, format, Val(MR), kc, m, transform, load, plane_offset)
    return packed
end

@inline function _pack_b_sliver!(
        format::FMT, packed::V, source::QSTile, kernel::Descriptor{MR, NR, T2},
        transform::F, n::Int, kc::Int
    ) where {FMT <: Union{PlanarFormat, InterleavedFormat, OneEFormat}, V, MR, NR, T2, F}
    # B's packed index runs along `source.cols` (the NR logical columns), so
    # the unit-stride requirement is on the N axis, not the K axis -- the
    # mirror image of A's. `FB` is `PlanarFormat` under both shipped complex
    # methods, so 1m's B panel uses the same predicate and the same packer as
    # `PlanarMethod`.
    if _pack_complex_contiguous_eligible(
            packed, source.storage, source.cols, transform, format, n, Val(NR), T2
        )
        elembase = source.base + source.cols.base
        return _pack_complex_contiguous!(
            format, packed, source.storage, elembase, source.rows, Val(NR), kc, transform
        )
    end

    load = (j, p) -> tile_load(source, p, j)  # source.rows=K, source.cols=N
    plane_offset = (plane, j, p) -> packed_b_plane_offset(kernel, plane, j, p)
    _pack_panel!(packed, T2, format, Val(NR), kc, n, transform, load, plane_offset)
    return packed
end
