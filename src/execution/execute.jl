# `C *= beta` tile by tile, never reading A or B: the
# `k_length == 0 || alpha == 0` pass of `execute!`. Borrows the first tile of
# the M/N offset buffers.
function _scale_all_of_C!(plan, betaT::T, m_tile::Int, n_tile::Int, m_length::Int, n_length::Int) where {T}
    ws = plan.workspace
    m_bufs = ws.m.offsets
    n_bufs = ws.n.offsets
    m_tile_start = 0
    while m_tile_start < m_length
        m_tile_length = min(m_tile, m_length - m_tile_start)
        (_, dM_C) = block_descriptors!(m_bufs, plan.mgroup, m_tile_start, m_tile_length)
        n_tile_start = 0
        while n_tile_start < n_length
            n_tile_length = min(n_tile, n_length - n_tile_start)
            (_, dN_C) = block_descriptors!(n_bufs, plan.ngroup, n_tile_start, n_tile_length)
            rowsC = _axis_of(dM_C, m_bufs[2], 0)
            colsC = _axis_of(dN_C, n_bufs[2], 0)
            GC.@preserve ws _scale_micro_tile!(plan.Cstorage, plan.Cbase, rowsC, colsC, betaT)
            n_tile_start += n_tile_length
        end
        m_tile_start += m_tile_length
    end
    return nothing
end

"""
    execute!(plan::ContractPlan, alpha::Number, beta::Number)

Compute `C = alpha * A * B + beta * C` for the contraction `plan` describes,
and return `plan.Cstorage`. Runs a BLIS five-loop nest over `plan.blocking`,
or a dedicated dot/outer-product path for degenerate extents. `beta` applies
exactly once per element and `beta == 0` never reads `C`; empty K or
`alpha == 0` never reads `A`/`B`. Operand bounds are checked once per macro
block. Allocation-free.
"""
function execute!(plan::ContractPlan{T}, alpha::Number, beta::Number) where {T}
    alphaT = convert(T, alpha)
    betaT = convert(T, beta)
    execute_short_circuit!(plan, alphaT, betaT) && return plan.Cstorage
    execute_path!(plan, alphaT, betaT, plan.path)
    return plan.Cstorage
end

# The short-circuits every path shares; `true` when the call is finished.
@inline function execute_short_circuit!(plan::ContractPlan{T}, alphaT::T, betaT::T) where {T}
    m_length = axis_length(plan.mgroup)
    n_length = axis_length(plan.ngroup)
    (m_length == 0 || n_length == 0) && return true
    if axis_length(plan.kgroup) == 0 || iszero(alphaT)
        _scale_all_of_C!(plan, betaT, tile_size(plan.kernel)..., m_length, n_length)
        return true
    end
    return false
end

execute_path!(plan::ContractPlan, alphaT, betaT, ::DotPath{MATB, W}) where {MATB, W} =
    (_execute_dot!(plan, alphaT, betaT, MATB, Val(W)); nothing)

execute_path!(plan::ContractPlan, alphaT, betaT, ::OuterPath{W}) where {W} = (
    _execute_outer!(
        plan, alphaT, betaT, axis_length(plan.mgroup), axis_length(plan.ngroup), Val(W)
    ); nothing
)

function execute_path!(
        plan::ContractPlan{T}, alphaT::T, betaT::T, path::NestPath
    ) where {T}
    kernel = plan.kernel
    ws = plan.workspace
    # Packed panels and scatter axes borrow pointers into `ws`.
    GC.@preserve ws begin
        _execute_nest!(
            plan, ws, kernel, tile_size(kernel)...,
            axis_length(plan.mgroup), axis_length(plan.ngroup), axis_length(plan.kgroup),
            plan.blocking.m_block, plan.blocking.k_block, plan.blocking.n_block, alphaT, betaT, path, nothing
        )
    end
    return nothing
end

# The nest over `pplan`, a copy of `plan` whose C is the workspace panel with
# dense M/N maps, one N block at a time: `panel_enter!` loads the block of
# C into the panel and `panel_exit!` rounds it back.
function execute_path!(
        plan::ContractPlan{T}, alphaT::T, betaT::T, ::PanelPath{P}
    ) where {T, P}
    kernel = plan.kernel
    ws = plan.workspace
    m_length = axis_length(plan.mgroup)
    pplan = ContractPlan(
        plan; mgroup = dense_second_map(plan.mgroup, 1), ngroup = dense_second_map(plan.ngroup, m_length),
        Cstorage = ws.c_panel, Cbase = 0
    )
    GC.@preserve ws begin
        _execute_nest!(
            pplan, ws, kernel, tile_size(kernel)...,
            m_length, axis_length(plan.ngroup), axis_length(plan.kgroup),
            plan.blocking.m_block, plan.blocking.k_block, plan.blocking.n_block, alphaT, betaT, P(), plan
        )
    end
    return nothing
end

# `g` with its second map replaced by the column-major one scaled by `step`.
@inline function dense_second_map(g::AxisGroup{D, 2}, step::Int) where {D}
    dense = ntuple(d -> step * prod(ntuple(i -> i < d ? g.lengths[i] : 1, Val(D))), Val(D))
    return AxisGroup(g.lengths, (g.strides[1], dense))
end

# `target`: `nothing`, or the plan whose C the panel stands in for.
@inline panel_enter!(::Nothing, plan, n_block_start, n_block_length, betaT) = plan
@inline panel_exit!(::Nothing, n_block_start, n_block_length) = nothing

function panel_enter!(target::ContractPlan, plan::ContractPlan, n_block_start::Int, n_block_length::Int, betaT)
    iszero(betaT) || panel_copy!(target, n_block_start, n_block_length, true)
    return ContractPlan(plan; Cbase = -n_block_start * axis_length(plan.mgroup))
end

panel_exit!(target::ContractPlan, n_block_start::Int, n_block_length::Int) = panel_copy!(target, n_block_start, n_block_length, false)

# Columns `n_block_start .+ (0:n_block_length-1)` of `target`'s C into
# (`load`) or out of the panel, converting to the destination's eltype. Borrows
# the M/N offset buffers, which the nest refills before reading them again.
function panel_copy!(target::ContractPlan, n_block_start::Int, n_block_length::Int, load::Bool)
    ws = target.workspace
    panel = ws.c_panel
    C = target.Cstorage
    m_length = axis_length(target.mgroup)
    m_block = target.blocking.m_block
    n_bufC = ws.n.offsets[2]
    fill_offsets!(ws.n.offsets, target.ngroup, n_block_start, n_block_length)
    m_block_start = 0
    while m_block_start < m_length
        m_block_length = min(m_block, m_length - m_block_start)
        m_bufC = ws.m.offsets[2]
        fill_offsets!(ws.m.offsets, target.mgroup, m_block_start, m_block_length)
        for j in 1:n_block_length, i in 1:m_block_length
            c = target.Cbase + m_bufC[i] + n_bufC[j] + 1
            q = m_block_start + i + (j - 1) * m_length
            if load
                panel[q] = C[c]
            else
                C[c] = panel[q]
            end
        end
        m_block_start += m_block_length
    end
    return nothing
end

function _execute_nest!(
        plan::ContractPlan{T}, ws, kernel::K, m_tile::Int, n_tile::Int,
        m_length::Int, n_length::Int, k_length::Int, m_block::Int, k_block::Int, n_block::Int,
        alphaT::T, betaT::T, ::NestPath{UNPACKED_B, AFF, SPLIT}, target
    ) where {T, K, UNPACKED_B, AFF, SPLIT}
    # GUARDRAIL: reals per sliver per K step address the packed panels;
    # `m_tile`/`n_tile` count register-tile rows. They differ for complex
    # kernels.
    a_sliver_width, b_sliver_width = sliver_width(kernel)

    atransform = plan.atransform
    btransform = plan.btransform

    split_a, split_b = SPLIT
    mgroup = split_a ? split_group(plan.mgroup, plan.mpack) : plan.mgroup
    ngroup = split_b ? split_group(plan.ngroup, plan.npack) : plan.ngroup

    # Ramp composites get closed-form block descriptors, no offset buffers; the
    # block pack reads the offsets.
    (m_ramp, m_step) = affine_ramp(mgroup)
    (n_ramp, n_step) = affine_ramp(ngroup)
    m_ramp &= !split_a
    n_ramp &= !split_b
    (k_ramp, k_step) = affine_ramp(plan.kgroup)

    lenA = length(plan.Astorage)
    lenB = length(plan.Bstorage)
    lenC = length(plan.Cstorage)

    aff_mA, aff_mC, aff_nB, aff_nC, aff_kA, aff_kB = map(Val, AFF)

    # --- loop over N blocks ---
    n_block_start = 0
    while n_block_start < n_length
        n_block_length = min(n_block, n_length - n_block_start)
        cplan = panel_enter!(target, plan, n_block_start, n_block_length, betaT)
        n_tiles = cld(n_block_length, n_tile)
        (rng_nB, rng_nC) = if n_ramp
            _ramp_slivers!(
                ws.n, n_step[1], n_step[2], n_block_start, n_block_length, n_tile, n_tiles
            )
        else
            fill_offsets!(ws.n.offsets, ngroup, n_block_start, n_block_length)
            _classify_slivers!(ws.n, n_block_length, n_tile, n_tiles)
        end

        # --- loop over K blocks ---
        k_block_start = 0
        firstpanel = true
        while k_block_start < k_length
            k_block_length = min(k_block, k_length - k_block_start)
            dK_A, dK_B, rng_kA, rng_kB = if k_ramp
                (
                    _ramp_descriptor(k_step[1], k_block_start, k_block_length),
                    _ramp_descriptor(k_step[2], k_block_start, k_block_length),
                    _ramp_offset_range(k_step[1], k_block_start, k_block_length),
                    _ramp_offset_range(k_step[2], k_block_start, k_block_length),
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

            colsA_k = _axis_of(dK_A, ws.k[1], 0, aff_kA)
            rowsB_k = _axis_of(dK_B, ws.k[2], 0, aff_kB)

            # Hoisted bounds checks (B here, A and C per M block): each
            # rectangle is exactly the union of the per-sliver/per-tile
            # regions, and the storage check compares only range extremes, so
            # the `@inbounds` pack and tile calls below touch nothing unchecked.
            checked_span_bounds(plan.Bbase, rng_kB, rng_nB, lenB)

            beta_eff = firstpanel ? betaT : one(T)

            if split_b
                pack_block_by_lines!(
                    packed_panel(ws.packed_b, 1, b_sliver_width * k_block_length * n_tiles), sliver_spec(kernel, 2),
                    plan.Bstorage, plan.Bbase, ws.n.offsets[1], rowsB_k, btransform, n_block_length, k_block_length,
                    plan.npack
                )
            elseif !UNPACKED_B
                for n_tile_index in 0:(n_tiles - 1)
                    n_tile_start = n_tile_index * n_tile
                    bpanel = _sliver_panel(ws.packed_b, b_sliver_width, k_block_length, n_tile_index)
                    colsB = _axis_of(ws.n.descriptors[1][n_tile_index + 1], ws.n.offsets[1], n_tile_start, aff_nB)
                    @inbounds _pack_sliver!(
                        bpanel, plan.Bstorage, plan.Bbase, colsB, rowsB_k,
                        sliver_spec(kernel, 2), btransform
                    )
                end
            end

            # --- loop over M blocks ---
            m_block_start = 0
            while m_block_start < m_length
                m_block_length = min(m_block, m_length - m_block_start)
                m_tiles = cld(m_block_length, m_tile)
                (rng_mA, rng_mC) = if m_ramp
                    _ramp_slivers!(
                        ws.m, m_step[1], m_step[2], m_block_start, m_block_length, m_tile, m_tiles
                    )
                else
                    fill_offsets!(ws.m.offsets, mgroup, m_block_start, m_block_length)
                    _classify_slivers!(ws.m, m_block_length, m_tile, m_tiles)
                end

                checked_span_bounds(plan.Abase, rng_mA, rng_kA, lenA)
                checked_span_bounds(cplan.Cbase, rng_mC, rng_nC, lenC)

                if split_a
                    pack_block_by_lines!(
                        packed_panel(ws.packed_a, 1, a_sliver_width * k_block_length * m_tiles), sliver_spec(kernel, 1),
                        plan.Astorage, plan.Abase, ws.m.offsets[1], colsA_k, atransform, m_block_length, k_block_length,
                        plan.mpack
                    )
                else
                    for m_tile_index in 0:(m_tiles - 1)
                        m_tile_start = m_tile_index * m_tile
                        apanel = _sliver_panel(ws.packed_a, a_sliver_width, k_block_length, m_tile_index)
                        rowsA = _axis_of(ws.m.descriptors[1][m_tile_index + 1], ws.m.offsets[1], m_tile_start, aff_mA)
                        @inbounds _pack_sliver!(
                            apanel, plan.Astorage, plan.Abase, rowsA, colsA_k,
                            sliver_spec(kernel, 1), atransform
                        )
                    end
                end

                # --- loops over N tiles and M tiles ---
                if UNPACKED_B
                    _micro_tiles_unpacked_b!(
                        kernel, cplan, ws, rowsB_k, m_tiles, n_tiles,
                        m_tile, n_tile, a_sliver_width, k_block_length, alphaT, beta_eff, aff_mC, aff_nC
                    )
                else
                    _micro_tiles_packed_b!(
                        kernel, cplan, ws, m_tiles, n_tiles,
                        m_tile, n_tile, a_sliver_width, b_sliver_width, k_block_length, alphaT, beta_eff, aff_mC, aff_nC
                    )
                end

                m_block_start += m_block_length
            end

            firstpanel = false
            k_block_start += k_block_length
        end

        panel_exit!(target, n_block_start, n_block_length)
        n_block_start += n_block_length
    end

    return plan.Cstorage
end

# The tile loops over one (N, K, M) block. `@noinline`: one call per block, and
# the inlined microkernel is most of a nest's size; compile cost grows faster
# than linearly with function size.
@noinline function _micro_tiles_packed_b!(
        kernel::K, plan::ContractPlan, ws, m_tiles::Int, n_tiles::Int,
        m_tile::Int, n_tile::Int, a_sliver_width::Int, b_sliver_width::Int, k_block_length::Int, alphaT, beta_eff,
        aff_mC::Val{MC}, aff_nC::Val{NC}
    ) where {K, MC, NC}
    for n_tile_index in 0:(n_tiles - 1)
        n_tile_start = n_tile_index * n_tile
        bpanel = _sliver_panel(ws.packed_b, b_sliver_width, k_block_length, n_tile_index)
        colsC = _axis_of(ws.n.descriptors[2][n_tile_index + 1], ws.n.offsets[2], n_tile_start, aff_nC)
        for m_tile_index in 0:(m_tiles - 1)
            m_tile_start = m_tile_index * m_tile
            apanel = _sliver_panel(ws.packed_a, a_sliver_width, k_block_length, m_tile_index)
            rowsC = _axis_of(ws.m.descriptors[2][m_tile_index + 1], ws.m.offsets[2], m_tile_start, aff_mC)
            # Inside the caller's C check.
            @inbounds _execute_micro_tile!(
                kernel, plan.Cstorage, plan.Cbase, rowsC, colsC,
                apanel, bpanel, k_block_length, alphaT, beta_eff
            )
        end
    end
    return nothing
end

"""
    contract!(C::StridedView, alpha::Number,
              A::StridedView, indA::NTuple{NA,Int},
              B::StridedView, indB::NTuple{NB,Int},
              beta::Number,
              indC::NTuple{NC,Int}; accumulator = nothing) where {NA,NB,NC}

Compute `C[indC] = alpha * sum_K A[indA] * B[indB] + beta * C[indC]`, with one
`Int` label per axis: a label in `indA` and `indB` but not `indC` is contracted
(K), and a label in `indC` and exactly one of `indA`/`indB` is free (M or N).
Any other label pattern, or a label repeated within one tuple, throws an
`ArgumentError`; matched labels of unequal axis length throw a
`DimensionMismatch`. Eltypes and `accumulator` are as in
[`plan_contract`](@ref). Equivalent to
`execute!(plan_contract(C, A, indA, B, indB, indC; accumulator), alpha, beta)`
— use those directly to reuse a plan across calls. Returns `C`.
"""
function contract!(
        C::StridedView, alpha::Number,
        A::StridedView, indA::NTuple{NA, Int},
        B::StridedView, indB::NTuple{NB, Int},
        beta::Number,
        indC::NTuple{NC, Int};
        accumulator::Union{Nothing, Type{Float32}, Type{Float64}} = nothing
    ) where {NA, NB, NC}
    plan = plan_contract(C, A, indA, B, indB, indC; accumulator)
    execute!(plan, alpha, beta)
    return C
end
