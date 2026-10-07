# The five-loop nest and its helpers: function barriers over the tile axis
# types, block packing, packed-sliver addressing and sliver classification.

# The nest over the N coordinates `n_range`, in N blocks from its start. The
# caller preserves `plan.workspace`: packed panels and scatter axes borrow
# pointers into it.
function nest!(
        plan::ContractPlan{T}, alphaT::T, betaT::T, path::NestPath{UNPACKED_B, AFF, SPLIT},
        n_range::UnitRange{Int}
    ) where {T, UNPACKED_B, AFF, SPLIT}
    kernel = plan.kernel
    ws = plan.workspace
    m_tile, n_tile = tile_size(kernel)
    (; m_block, k_block, n_block) = plan.blocking
    m_length = axis_length(plan.mgroup)
    k_length = axis_length(plan.kgroup)

    split_a, split_b = SPLIT
    mgroup = split_a ? split_group(plan.mgroup, plan.mpack) : plan.mgroup
    ngroup = split_b ? split_group(plan.ngroup, plan.npack) : plan.ngroup

    # Ramp composites get closed-form block descriptors, no offset buffers; the
    # block pack reads the offsets. A group is a ramp when both its maps are,
    # which the path's flags record, except N's B map under unpacked B.
    aff_mA, aff_mC, aff_nB, aff_nC, aff_kA, aff_kB = AFF
    m_ramp = aff_mA && aff_mC
    n_ramp = (UNPACKED_B ? is_ramp_map(ngroup, 1) : aff_nB) && aff_nC
    k_ramp = aff_kA && aff_kB
    m_step = m_ramp ? affine_ramp(mgroup)[2] : (0, 0)
    n_step = n_ramp ? affine_ramp(ngroup)[2] : (0, 0)
    k_step = k_ramp ? affine_ramp(plan.kgroup)[2] : (0, 0)

    lenA = length(plan.Astorage)
    lenB = length(plan.Bstorage)
    lenC = length(plan.Cstorage)

    # --- loop over N blocks ---
    n_block_start = first(n_range)
    while n_block_start <= last(n_range)
        n_block_length = min(n_block, last(n_range) + 1 - n_block_start)
        n_tiles = cld(n_block_length, n_tile)
        (rng_nB, rng_nC) = if n_ramp
            ramp_slivers!(
                ws.n, n_step[1], n_step[2], n_block_start, n_block_length, n_tile, n_tiles
            )
        else
            fill_offsets!(ws.n.offsets, ngroup, n_block_start, n_block_length)
            classify_slivers!(ws.n, n_block_length, n_tile, n_tiles)
        end

        # --- loop over K blocks ---
        k_block_start = 0
        first_k_block = true
        while k_block_start < k_length
            k_block_length = min(k_block, k_length - k_block_start)
            dK_A, dK_B, rng_kA, rng_kB = if k_ramp
                (
                    ramp_descriptor(k_step[1], k_block_start, k_block_length),
                    ramp_descriptor(k_step[2], k_block_start, k_block_length),
                    ramp_offset_range(k_step[1], k_block_start, k_block_length),
                    ramp_offset_range(k_step[2], k_block_start, k_block_length),
                )
            else
                fill_offsets!(ws.k, plan.kgroup, k_block_start, k_block_length)
                dA = describe_block(ws.k[1], 0, k_block_length)
                dB = describe_block(ws.k[2], 0, k_block_length)
                (
                    dA, dB,
                    descriptor_offset_range(dA, ws.k[1], 0),
                    descriptor_offset_range(dB, ws.k[2], 0),
                )
            end

            colsA_k = axis_of(dK_A, ws.k[1], 0, Val(aff_kA))
            rowsB_k = axis_of(dK_B, ws.k[2], 0, Val(aff_kB))

            # Hoisted bounds checks (B here, A and C per M block): each
            # rectangle is exactly the union of the per-sliver/per-tile
            # regions, and the storage check compares only range extremes, so
            # the `@inbounds` pack and tile calls below touch nothing unchecked.
            checked_span_bounds(plan.Bbase, rng_kB, rng_nB, lenB)

            beta_eff = first_k_block ? betaT : one(T)

            UNPACKED_B || pack_block!(plan, path, Val(2), rowsB_k, n_block_length, k_block_length, n_tiles)

            # --- loop over M blocks ---
            m_block_start = 0
            while m_block_start < m_length
                m_block_length = min(m_block, m_length - m_block_start)
                m_tiles = cld(m_block_length, m_tile)
                (rng_mA, rng_mC) = if m_ramp
                    ramp_slivers!(
                        ws.m, m_step[1], m_step[2], m_block_start, m_block_length, m_tile, m_tiles
                    )
                else
                    fill_offsets!(ws.m.offsets, mgroup, m_block_start, m_block_length)
                    classify_slivers!(ws.m, m_block_length, m_tile, m_tiles)
                end

                checked_span_bounds(plan.Abase, rng_mA, rng_kA, lenA)
                checked_span_bounds(plan.Cbase, rng_mC, rng_nC, lenC)

                pack_block!(plan, path, Val(1), colsA_k, m_block_length, k_block_length, m_tiles)

                # --- loops over N tiles and M tiles ---
                micro_tiles!(
                    plan, path, UNPACKED_B ? rowsB_k : nothing, m_tiles, n_tiles, k_block_length,
                    alphaT, beta_eff, Val(aff_mC), Val(aff_nC)
                )

                m_block_start += m_block_length
            end

            first_k_block = false
            k_block_start += k_block_length
        end

        n_block_start += n_block_length
    end
    return nothing
end

# Pack one macro block of operand `I` (1: A over M, 2: B over N): line by line
# when the path splits it, else sliver by sliver. Inside the caller's bounds
# checks.
@inline function pack_block!(
        plan::ContractPlan, ::NestPath{UNPACKED_B, AFF, SPLIT}, side::Val{I}, kaxis::KA,
        block_length::Int, k_block_length::Int, tiles::Int
    ) where {UNPACKED_B, AFF, SPLIT, I, KA <: AbstractVector{Int}}
    kernel = plan.kernel
    spec = sliver_spec(kernel, I)
    width = sliver_width(kernel, I)
    tile = tile_size(kernel, I)
    storage, base, transform, buffer, g, split = packed_operand(plan, side)
    if SPLIT[I]
        pack_block_by_lines!(
            packed_panel(buffer, 1, width * k_block_length * tiles), spec,
            storage, base, g.offsets[1], kaxis, transform, block_length, k_block_length, split
        )
    else
        for tile_index in 0:(tiles - 1)
            tile_start = tile_index * tile
            panel = sliver_panel(buffer, width, k_block_length, tile_index)
            lanes = axis_of(g.descriptors[1][tile_index + 1], g.offsets[1], tile_start, Val(AFF[2I - 1]))
            @inbounds pack_sliver!(panel, storage, base, lanes, kaxis, spec, transform)
        end
    end
    return nothing
end

# Operand `I`'s storage, base and transform, packed buffer, free-group buffers
# and split.
@inline packed_operand(plan::ContractPlan, ::Val{1}) =
    (plan.Astorage, plan.Abase, plan.atransform, plan.workspace.packed_a, plan.workspace.m, plan.mpack)
@inline packed_operand(plan::ContractPlan, ::Val{2}) =
    (plan.Bstorage, plan.Bbase, plan.btransform, plan.workspace.packed_b, plan.workspace.n, plan.npack)

# The tile loops over one (N, K, M) block. `@noinline`: one call per block, and
# the inlined microkernel is most of a nest's size; compile cost grows faster
# than linearly with function size. `rowsB_k`: B's K axis when B is read in
# place (a barrier over its type, so each view is concretely typed), else
# `nothing`, so that the packed-B loop does not specialise on it.
@noinline function micro_tiles!(
        plan::ContractPlan, path::NestPath, rowsB_k::KB, m_tiles::Int, n_tiles::Int, k_block_length::Int,
        alphaT, beta_eff, aff_mC::Val{MC}, aff_nC::Val{NC}
    ) where {KB, MC, NC}
    kernel = plan.kernel
    ws = plan.workspace
    m_tile, n_tile = tile_size(kernel)
    # GUARDRAIL: reals per sliver per K step address the packed panels;
    # `m_tile`/`n_tile` count register-tile rows. They differ for complex
    # kernels.
    a_sliver_width = sliver_width(kernel, 1)
    for n_tile_index in 0:(n_tiles - 1)
        n_tile_start = n_tile_index * n_tile
        bsliver = b_sliver(plan, path, rowsB_k, k_block_length, n_tile_index, n_tile_start)
        colsC = axis_of(ws.n.descriptors[2][n_tile_index + 1], ws.n.offsets[2], n_tile_start, aff_nC)
        for m_tile_index in 0:(m_tiles - 1)
            m_tile_start = m_tile_index * m_tile
            apanel = sliver_panel(ws.packed_a, a_sliver_width, k_block_length, m_tile_index)
            rowsC = axis_of(ws.m.descriptors[2][m_tile_index + 1], ws.m.offsets[2], m_tile_start, aff_mC)
            # Inside the caller's C check.
            @inbounds execute_micro_tile!(
                kernel, plan.Cstorage, plan.Cbase, rowsC, colsC,
                apanel, bsliver, k_block_length, alphaT, beta_eff
            )
        end
    end
    return nothing
end

# The B sliver of N tile `n_tile_index`: its packed panel, or B in place.
@inline b_sliver(plan::ContractPlan, ::NestPath{false}, ::Nothing, k_block_length::Int, n_tile_index::Int, ::Int) =
    sliver_panel(plan.workspace.packed_b, sliver_width(plan.kernel, 2), k_block_length, n_tile_index)
@inline function b_sliver(
        plan::ContractPlan, ::NestPath{true}, rowsB_k::KB, ::Int, n_tile_index::Int, n_tile_start::Int
    ) where {KB <: AbstractVector{Int}}
    n = plan.workspace.n
    return unpacked_b_view(
        plan.kernel, plan.Bstorage, plan.Bbase, n.descriptors[1][n_tile_index + 1], n.offsets[1],
        n_tile_start, rowsB_k, plan.btransform
    )
end

# The axis `d` describes. GUARDRAIL: a `Union{AffineAxis, ScatterAxis}` of
# `isbits` types, never boxed; each consumer below binds each axis type as its
# own parameter, so it builds a concretely typed `Tile`.
@inline axis_of(d::BlockDescriptor, buffer::Vector{Int}, first::Int) =
    d.regular ? AffineAxis(d.base, d.stride, d.count) : ScatterAxis(pointer(buffer, first + 1), d.count)

# The same with the axis type fixed by the path: `Val(true)` for a ramp map,
# whose descriptors are always regular (checked).
@inline axis_of(d::BlockDescriptor, buffer::Vector{Int}, first::Int, ::Val{false}) =
    axis_of(d, buffer, first)
@inline function axis_of(d::BlockDescriptor, ::Vector{Int}, ::Int, ::Val{true})
    d.regular || throw_irregular_ramp_descriptor()
    return AffineAxis(d.base, d.stride, d.count)
end
@noinline throw_irregular_ramp_descriptor() =
    throw(AssertionError("an affine-ramp map produced an irregular block descriptor"))

# B passes its axes swapped (its `transpose`). GUARDRAIL: `transform` needs its
# own bound type parameter, or it costs a dynamic dispatch per call.
Base.@propagate_inbounds function pack_sliver!(
        packed::PK, storage::S, base::Int,
        lanes::R, steps::C, spec, transform::TF
    ) where {PK, S, R <: AbstractVector{Int}, C <: AbstractVector{Int}, TF}
    pack!(packed, Tile(storage, base, lanes, steps), spec, transform)
    return nothing
end

Base.@propagate_inbounds function execute_micro_tile!(
        kernel, storage::S, base::Int, rows::R, cols::C,
        packed_a::PA, packed_b::PB, k_block_length::Int, alpha, beta
    ) where {PA, PB, S, R <: AbstractVector{Int}, C <: AbstractVector{Int}}
    destination = Tile(storage, base, rows, cols)
    execute_tile!(kernel, destination, packed_a, packed_b, k_block_length, alpha, beta)
    return nothing
end

# Sliver `tile_index` of a packed panel at the current block's depth
# `k_block_length`, shared by the packing and the consuming step so the two
# cannot disagree. GUARDRAIL: `width` counts reals per K step
# (`sliver_width`), not the tile size; they differ for complex kernels.
@inline function sliver_panel(buffer, width::Int, k_block_length::Int, tile_index::Int)
    stride = width * k_block_length
    return packed_panel(buffer, tile_index * stride + 1, stride)
end

# Classify each register sliver of a just-filled macro block, and return the
# two maps' offset ranges over the whole block (the slivers partition it) for
# the hoisted bounds checks.
@inline function classify_slivers!(g::GroupBuffers, block_length::Int, tile::Int, tile_count::Int)
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
@inline ramp_descriptor(step::Int, first::Int, count::Int) =
    count == 0 ? BlockDescriptor(0, 0, 0, true) :
    count == 1 ? BlockDescriptor(first * step, 0, 1, true) :
    BlockDescriptor(first * step, step, count, true)

# Offset range of `[first, first+count)`; `(0, -1)` if empty. Cannot overflow:
# `AxisGroup` validated every in-domain offset at construction.
@inline ramp_offset_range(step::Int, first::Int, count::Int) =
    count == 0 ? (0, -1) : minmax(first * step, (first + count - 1) * step)

# `classify_slivers!`'s closed-form twin.
@inline function ramp_slivers!(
        g::GroupBuffers, step1::Int, step2::Int, first::Int,
        block_length::Int, tile::Int, tile_count::Int
    )
    desc1, desc2 = g.descriptors
    for tile_index in 0:(tile_count - 1)
        tile_start = tile_index * tile
        tile_length = min(tile, block_length - tile_start)
        q0 = first + tile_start
        desc1[tile_index + 1] = ramp_descriptor(step1, q0, tile_length)
        desc2[tile_index + 1] = ramp_descriptor(step2, q0, tile_length)
    end
    return (
        ramp_offset_range(step1, first, block_length),
        ramp_offset_range(step2, first, block_length),
    )
end
