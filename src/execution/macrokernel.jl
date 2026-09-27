# Macro-kernel helpers for the five-loop nest: function barriers over the
# tile axis types, packed-sliver addressing, and sliver classification.

# GUARDRAIL, load-bearing: `_axis_of` returns a `Union{AffineAxis,PtrScatterAxis}`, and each
# consumer below is a `where {R<:Axis, C<:Axis}` barrier method that Julia
# specializes per concrete (R,C), so no partially-applied -- heap-boxed --
# `QSTile` is ever built. **Do not** collapse these helpers into their call
# sites, and do not let a union cross any other boundary.
# Both arms are isbits, so this Union itself needs no heap box (see
# PtrScatterAxis).
@inline function _axis_of(d::BlockDescriptor, buffer::Vector{Int}, first::Int)
    return d.regular ? AffineAxis(d.base, d.stride, d.count) :
        PtrScatterAxis(pointer(buffer, first + 1), d.count)
end

# The same, with the sliver's axis type fixed statically by the execution
# path (`_NestPath`, src/execution/barrier.jl): `Val(true)` where the
# composite's map is an affine ramp, which makes every block descriptor
# `describe_block` or `_ramp_descriptor` produces from it regular, so the axis
# is always an `AffineAxis` and no `PtrScatterAxis` arm is compiled
# downstream. The `regular` test stays as a cheap guard on that invariant.
@inline _axis_of(d::BlockDescriptor, buffer::Vector{Int}, first::Int, ::Val{false}) =
    _axis_of(d, buffer, first)
@inline function _axis_of(d::BlockDescriptor, ::Vector{Int}, ::Int, ::Val{true})
    d.regular || _throw_irregular_ramp_descriptor()
    return AffineAxis(d.base, d.stride, d.count)
end
@noinline _throw_irregular_ramp_descriptor() =
    throw(AssertionError("an affine-ramp map produced an irregular block descriptor"))

# `pack!` is pack_a!/pack_b! -- or their `unsafe_pack_a!`/`unsafe_pack_b!`
# siblings (src/packing/pack.jl) -- as a plain function, specialized on, never a
# closure; A and B differ only in which of rows/cols is the k axis, which the
# caller has already resolved. `transform` is the plan's per-operand
# `identity`/`conj` singleton.
#
# This helper is only as safe as the `pack!` it is handed: with an `unsafe_*`
# packer it performs no storage-bounds check, which is why the call sites spell
# that name out rather than hiding it behind a flag.
#
# GUARDRAIL: every argument here has its OWN bound type parameter, `transform`
# included. Leaving `TF` unbound costs a dynamic dispatch (~80 B/call), for the
# reason spelled out at `pack_a!` in src/microkernels/interface.jl.
# And all THREE call sites -- `_execute_nest!`'s two and `execute_tilewise!`'s
# one -- must pass the matching operand's transform: missing the third makes
# the in-tree ORACLE silently wrong for conjugated inputs.
@inline function _pack_sliver!(
        pack!::PF, packed::PK, storage::S, base::Int,
        rows::R, cols::C, kernel, transform::TF
    ) where {PF, PK, S, R <: Axis, C <: Axis, TF}
    pack!(packed, SourceTile(storage, base, rows, cols), kernel, transform)
    return nothing
end

@inline function _execute_micro_tile!(
        kernel, storage::S, base::Int, rows::R, cols::C,
        packed_a::PA, packed_b::PB, kc_len::Int, alpha, beta
    ) where {PA, PB, S, R <: Axis, C <: Axis}
    destination = DestinationTile(storage, base, rows, cols)
    execute_tile!(kernel, destination, packed_a, packed_b, kc_len, alpha, beta)
    return nothing
end

# Same guardrail barrier as `_execute_micro_tile!`, over `unsafe_execute_tile!`
# (src/microkernels/interface.jl) instead of `execute_tile!`: the destination's storage-bounds
# check has already been made ONCE for the whole (ic, jc) macro block this tile
# belongs to. `unsafe_` is in the name at every call site precisely because the
# precondition lives at the caller.
@inline function unsafe_execute_micro_tile!(
        kernel, storage::S, base::Int, rows::R, cols::C,
        packed_a::PA, packed_b::PB, kc_len::Int, alpha, beta
    ) where {PA, PB, S, R <: Axis, C <: Axis}
    destination = DestinationTile(storage, base, rows, cols)
    unsafe_execute_tile!(kernel, destination, packed_a, packed_b, kc_len, alpha, beta)
    return nothing
end

@inline function _scale_micro_tile!(
        storage::S, base::Int, rows::R, cols::C, beta
    ) where {S, R <: Axis, C <: Axis}
    destination = DestinationTile(storage, base, rows, cols)
    scale_tile!(destination, beta)
    return nothing
end

# Sliver `s` of a shared packed panel, as a borrowed pointer, at the CURRENT
# block's depth `kc_len` (< the buffer's per-sliver capacity on a tail K block,
# since the buffer is sized for kc_eff). Used by both the packing and the
# consuming step of a (jc,pc,ic) iteration, so the two cannot disagree.
#
# GUARDRAIL: `reg_tile` here is a count of REALS per logical K step --
# `packed_a_per_k`/`packed_b_per_k`, NOT `mr`/`nr`. They coincide for every
# real kernel (pinned in test/planning/test_kernel_selection.jl), so this is the identity on the
# real path; for a complex kernel only the packed count addresses the panel
# correctly.
@inline function _sliver_panel(buffer, reg_tile::Int, kc_len::Int, s::Int)
    stride = reg_tile * kc_len
    return packed_panel(buffer, s * stride + 1, stride)
end

# Classify each register sliver of a just-filled macro block. Shared by the
# N side (jc: B/C) and the M side (ic: A/C), which are structurally identical.
#
# Also returns the two maps' BLOCK offset ranges, `((lo1, hi1), (lo2, hi2))`,
# accumulated from the sliver descriptors as they are produced rather than in a
# second pass -- `O(1)` per regular sliver, and for an irregular one exactly
# the scan a per-sliver `checked_tile_storage_bounds` would do.
# Because the slivers partition `buf[1:blocklen]`, this union IS the range of
# the whole block, which is what `_execute_nest!`'s hoisted
# `checked_span_bounds` calls need.
@inline function _classify_slivers!(
        desc1::Vector{BlockDescriptor}, desc2::Vector{BlockDescriptor},
        buf1::Vector{Int}, buf2::Vector{Int},
        blocklen::Int, reg_tile::Int, nslivers::Int
    )
    lo1 = typemax(Int); hi1 = typemin(Int)
    lo2 = typemax(Int); hi2 = typemin(Int)
    for s in 0:(nslivers - 1)
        sfirst = s * reg_tile
        scount = min(reg_tile, blocklen - sfirst)
        d1 = describe_block(buf1, sfirst, scount)
        d2 = describe_block(buf2, sfirst, scount)
        desc1[s + 1] = d1
        desc2[s + 1] = d2
        (l1, h1) = descriptor_offset_range(d1, buf1, sfirst)
        if h1 >= l1
            lo1 = min(lo1, l1); hi1 = max(hi1, h1)
        end
        (l2, h2) = descriptor_offset_range(d2, buf2, sfirst)
        if h2 >= l2
            lo2 = min(lo2, l2); hi2 = max(hi2, h2)
        end
    end
    # `hi < lo` is `checked_span_bounds`'s "empty, always passes" convention,
    # which is what these initial values mean when no sliver contributed.
    return ((lo1, hi1), (lo2, hi2))
end

# ----------------------------------------------------------------------------
# Closed-form block description for an affine-ramp composite. When
# `affine_ramp(g)` holds, logical
# coordinate `q` maps to offset `q * step[p]` for every map `p`, so a block's
# whole sliver structure follows from arithmetic and neither the offset buffer
# nor `describe_block`'s scan is needed. `fill_offsets!` + `_classify_slivers!`
# stay as the fallback for every composite that is not provably a ramp.
# ----------------------------------------------------------------------------

# Exactly what `describe_block` classifies a materialized ramp interval as,
# INCLUDING its `stride == 0` convention for a count-1 block (which is
# behaviourally irrelevant -- an `AffineAxis` of count 1 never multiplies by
# its stride -- but keeping it identical means the descriptors these two paths
# produce are `==`, not merely equivalent, which a test can assert).
@inline _ramp_descriptor(step::Int, first::Int, count::Int) =
    count == 0 ? BlockDescriptor(0, 0, 0, true) :
    count == 1 ? BlockDescriptor(first * step, 0, 1, true) :
    BlockDescriptor(first * step, step, count, true)

# Offset range of `[first, first+count)` under a ramp, in `axis_offset_range`'s
# `(lo, hi)` / `(0, -1)`-if-empty convention. `first * step` and
# `(first+count-1) * step` are offsets of coordinates inside the group's
# domain, so they are covered by `AxisGroup`'s construction-time excursion
# validation and cannot overflow.
@inline _ramp_offset_range(step::Int, first::Int, count::Int) =
    count == 0 ? (0, -1) : minmax(first * step, (first + count - 1) * step)

# `_classify_slivers!`'s closed-form twin: same descriptors, same returned
# block ranges, no buffer touched.
@inline function _ramp_slivers!(
        desc1::Vector{BlockDescriptor}, desc2::Vector{BlockDescriptor},
        step1::Int, step2::Int, first::Int,
        blocklen::Int, reg_tile::Int, nslivers::Int
    )
    for s in 0:(nslivers - 1)
        sfirst = s * reg_tile
        scount = min(reg_tile, blocklen - sfirst)
        q0 = first + sfirst
        desc1[s + 1] = _ramp_descriptor(step1, q0, scount)
        desc2[s + 1] = _ramp_descriptor(step2, q0, scount)
    end
    return (
        _ramp_offset_range(step1, first, blocklen),
        _ramp_offset_range(step2, first, blocklen),
    )
end
