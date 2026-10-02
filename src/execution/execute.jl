# `C *= beta` tile by tile, never reading A or B: the
# `k_length == 0 || alpha == 0` pass of `execute!` and `execute_tilewise!`. Uses
# the tile-sized `tile_*` buffers, which is why those exist even under
# `oracle = false`.
function _scale_all_of_C!(plan, betaT::T, m_tile::Int, n_tile::Int, m_length::Int, n_length::Int) where {T}
    ws = plan.workspace
    m_bufs = (ws.tile_m_buf_A, ws.tile_m_buf_C)
    n_bufs = (ws.tile_n_buf_B, ws.tile_n_buf_C)
    m_tile_start = 0
    while m_tile_start < m_length
        m_tile_length = min(m_tile, m_length - m_tile_start)
        (_, dM_C) = block_descriptors!(m_bufs, plan.mgroup, m_tile_start, m_tile_length)
        n_tile_start = 0
        while n_tile_start < n_length
            n_tile_length = min(n_tile, n_length - n_tile_start)
            (_, dN_C) = block_descriptors!(n_bufs, plan.ngroup, n_tile_start, n_tile_length)
            rowsC = _axis_of(dM_C, ws.tile_m_buf_C, 0)
            colsC = _axis_of(dN_C, ws.tile_n_buf_C, 0)
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
    _execute_short_circuit!(plan, alphaT, betaT) && return plan.Cstorage
    _execute_across_barrier!(plan, alphaT, betaT, _select_path(plan))
    return plan.Cstorage
end

# The short-circuits every path shares; `true` when the call is finished.
@inline function _execute_short_circuit!(plan::ContractPlan{T}, alphaT::T, betaT::T) where {T}
    m_length = axis_length(plan.mgroup)
    n_length = axis_length(plan.ngroup)
    (m_length == 0 || n_length == 0) && return true
    if axis_length(plan.kgroup) == 0 || iszero(alphaT)
        _scale_all_of_C!(plan, betaT, tile_size(plan.kernel)..., m_length, n_length)
        return true
    end
    return false
end

# The path `execute!` runs past its short-circuits: the dot path, else the
# outer-product path, else the nest (B packed or read in place).
@inline _select_path(plan::ContractPlan{T}) where {T} = _select_path(
    T, _unpacked_b_kernel_eligible(plan.kernel),
    plan.Astorage, plan.Bstorage, plan.Cstorage, plan.mgroup, plan.ngroup, plan.kgroup, plan,
    plan.blocking.k_block, _is_split(plan.mpack), _is_split(plan.npack)
)

# On the plan's parts, so `plan_contract` can predict the path before the
# plan exists (`_path_hint`). `unpack_ok`: the kernel admits unpacked B.
# `capacity`: the plan whose workspace must hold the dot path's vector, or
# `nothing` to assume it does. `k_block`: the requested K block, `nothing` for
# the default. `split_a`/`split_b`: the plan packs A/B line by line.
@inline function _select_path(
        ::Type{T}, unpack_ok::Bool, Astorage, Bstorage, Cstorage,
        mgroup::AxisGroup, ngroup::AxisGroup, kgroup::AxisGroup, capacity, k_block,
        split_a::Bool = false, split_b::Bool = false
    ) where {T}
    m_length = axis_length(mgroup)
    n_length = axis_length(ngroup)
    k_length = axis_length(kgroup)
    panel = _c_panel_needed(T, Cstorage, k_length, k_block)
    if (m_length == 1 || n_length == 1) && !panel && _dot_applicable(T, Astorage, Bstorage, kgroup, m_length, n_length, k_length) &&
            _dot_capacity_ok(capacity)
        W = _dot_lanewidth(T)
        return m_length == 1 ? _lane_path(_DotPath{true}, W) : _lane_path(_DotPath{false}, W)
    end
    k_length == 1 && _outer_applicable(T, Astorage, Cstorage, mgroup, m_length) &&
        return _lane_path(_OuterPath, _dot_lanewidth(T))
    return _nest_path(
        unpack_ok && _unpacked_b_rule(mgroup, kgroup), mgroup, ngroup, kgroup, split_a, split_b, panel
    )
end

# Whether the partial sums between K blocks must live in a compute-type panel
# rather than in a C of narrower eltype. Static `false` unless C is narrower,
# so the default blocking is looked up only then.
@inline function _c_panel_needed(::Type{T}, Cstorage, k_length::Int, k_block) where {T}
    sizeof(real(eltype(Cstorage))) < sizeof(real(T)) || return false
    return k_length > (k_block === nothing ? _resolved_defaults(T).real_row.k_block : k_block)
end

# Run `path` behind a dynamic call. The plan crosses in the workspace slot
# with its storages stripped; they cross as arguments.
@inline function _execute_across_barrier!(plan::ContractPlan{T}, alphaT::T, betaT::T, path) where {T}
    core = _strip_storage(plan)
    slot = _barrier_slot!(plan.workspace, Tuple{typeof(core), T, T})
    slot[] = (core, alphaT, betaT)
    Base.inferencebarrier(_execute_resolved!)(
        path, slot, plan.Astorage, plan.Bstorage, plan.Cstorage
    )
    return nothing
end

function _execute_resolved!(
        path::P, slot::Base.RefValue{Tuple{R, T, T}}, Astorage::SA, Bstorage::SB, Cstorage::SC
    ) where {P, R <: ContractPlan, T, SA, SB, SC}
    core, alphaT, betaT = slot[]
    _execute_path!(_with_storage(core, Astorage, Bstorage, Cstorage), alphaT, betaT, path)
    return nothing
end

@inline _strip_storage(p::ContractPlan) = ContractPlan(
    p.kernel, p.mgroup, p.ngroup, p.kgroup, p.blocking,
    nothing, p.Abase, nothing, p.Bbase, nothing, p.Cbase,
    p.atransform, p.btransform, p.workspace, p.mpack, p.npack
)
@inline _with_storage(p::ContractPlan, Astorage, Bstorage, Cstorage) = ContractPlan(
    p.kernel, p.mgroup, p.ngroup, p.kgroup, p.blocking,
    Astorage, p.Abase, Bstorage, p.Bbase, Cstorage, p.Cbase,
    p.atransform, p.btransform, p.workspace, p.mpack, p.npack
)

# The continuation the TensorOperations adapter hands `_planned`: `execute!`
# inside the planning barrier. A struct, not a closure, so the scalars are
# concretely typed.
struct _Execute{T}
    alpha::T
    beta::T
end
(e::_Execute)(plan::ContractPlan) = (execute!(plan, e.alpha, e.beta); nothing)

# The path predicted before the kernel barrier, so its callee is specialised
# on the path as well as the kernel.
@inline _path_hint(::_Execute, req::_PlanRequest{T}, unpack_ok::Bool) where {T} = _select_path(
    T, unpack_ok,
    req.Astorage, req.Bstorage, req.Cstorage, req.mgroup, req.ngroup, req.kgroup, nothing,
    req.k_block
)

@inline _continue(e::_Execute{T}, plan::ContractPlan{T}, hint) where {T} =
    (_execute_hinted!(plan, e.alpha, e.beta, hint); nothing)
@inline _continue(e::_Execute{T}, plan::ContractPlan{T}, ::Nothing) where {T} =
    (execute!(plan, e.alpha, e.beta); nothing)

# Runs the predicted path statically when it provably equals the plan's own
# `_select_path` (always, for an automatically chosen kernel); otherwise
# falls back to `execute!`'s barrier. The prediction assumes no split, so a
# split plan crosses the barrier to its own nest path.
@inline function _execute_hinted!(plan::ContractPlan{T}, alphaT::T, betaT::T, hint) where {T}
    _execute_short_circuit!(plan, alphaT, betaT) && return nothing
    if !_hint_holds(plan, hint)
        _execute_across_barrier!(plan, alphaT, betaT, _select_path(plan))
    elseif _splits(plan, hint)
        _execute_across_barrier!(plan, alphaT, betaT, _split_path(plan, hint))
    else
        _execute_path!(plan, alphaT, betaT, hint)
    end
    return nothing
end

@inline _splits(plan::ContractPlan, hint) = false
@inline _splits(plan::ContractPlan, ::Union{_NestPath, _PanelPath}) = _is_split(plan.mpack) || _is_split(plan.npack)
@inline _split_path(plan::ContractPlan, ::_NestPath{U}) where {U} = _nest_path(
    U, plan.mgroup, plan.ngroup, plan.kgroup, _is_split(plan.mpack), _is_split(plan.npack)
)
@inline _split_path(plan::ContractPlan, ::_PanelPath{<:_NestPath{U}}) where {U} = _nest_path(
    U, plan.mgroup, plan.ngroup, plan.kgroup, _is_split(plan.mpack), _is_split(plan.npack), true
)

# The prediction differs from `_select_path(plan)` in three inputs only: the
# kernel's unpacked-B eligibility (predicted from its method; folds), the
# dot path's workspace capacity (assumed) and the panel decision (made at the
# default `k_block`; folds to `false` unless C is narrower than `T`).
@inline _hint_holds(plan::ContractPlan{T}, path::Union{_NestPath, _PanelPath}) where {T} =
    _c_panel_needed(T, plan.Cstorage, axis_length(plan.kgroup), plan.blocking.k_block) === (path isa _PanelPath) &&
    _unpacked_b_method_eligible(KernelMethod(plan.kernel)) === _unpacked_b_kernel_eligible(plan.kernel)
@inline _hint_holds(plan::ContractPlan, ::_DotPath) = _dot_capacity_ok(plan)
@inline _hint_holds(plan::ContractPlan, ::_OuterPath) = true

_execute_path!(plan::ContractPlan, alphaT, betaT, ::_DotPath{MATB, W}) where {MATB, W} =
    (_execute_dot!(plan, alphaT, betaT, MATB, Val(W)); nothing)

_execute_path!(plan::ContractPlan, alphaT, betaT, ::_OuterPath{W}) where {W} = (
    _execute_outer!(
        plan, alphaT, betaT, axis_length(plan.mgroup), axis_length(plan.ngroup), Val(W)
    ); nothing
)

function _execute_path!(
        plan::ContractPlan{T}, alphaT::T, betaT::T, path::_NestPath
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
# dense M/N maps, one N block at a time: `_panel_enter!` loads the block of
# C into the panel and `_panel_exit!` rounds it back.
function _execute_path!(
        plan::ContractPlan{T}, alphaT::T, betaT::T, ::_PanelPath{P}
    ) where {T, P}
    kernel = plan.kernel
    ws = plan.workspace
    m_length = axis_length(plan.mgroup)
    pplan = ContractPlan(
        kernel, _dense_second_map(plan.mgroup, 1), _dense_second_map(plan.ngroup, m_length),
        plan.kgroup, plan.blocking, plan.Astorage, plan.Abase, plan.Bstorage, plan.Bbase,
        ws.c_panel, 0, plan.atransform, plan.btransform, ws, plan.mpack, plan.npack
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
@inline function _dense_second_map(g::AxisGroup{D, 2}, step::Int) where {D}
    dense = ntuple(d -> step * prod(ntuple(i -> i < d ? g.lengths[i] : 1, Val(D))), Val(D))
    return AxisGroup(g.lengths, (g.strides[1], dense))
end

# `target`: `nothing`, or the plan whose C the panel stands in for.
@inline _panel_enter!(::Nothing, plan, n_block_start, n_block_length, betaT) = plan
@inline _panel_exit!(::Nothing, n_block_start, n_block_length) = nothing

function _panel_enter!(target::ContractPlan, plan::ContractPlan, n_block_start::Int, n_block_length::Int, betaT)
    iszero(betaT) || _panel_copy!(target, n_block_start, n_block_length, true)
    return ContractPlan(
        plan.kernel, plan.mgroup, plan.ngroup, plan.kgroup, plan.blocking,
        plan.Astorage, plan.Abase, plan.Bstorage, plan.Bbase, plan.Cstorage,
        -n_block_start * axis_length(plan.mgroup), plan.atransform, plan.btransform, plan.workspace,
        plan.mpack, plan.npack
    )
end

_panel_exit!(target::ContractPlan, n_block_start::Int, n_block_length::Int) = _panel_copy!(target, n_block_start, n_block_length, false)

# Columns `n_block_start .+ (0:n_block_length-1)` of `target`'s C into
# (`load`) or out of the panel, converting to the destination's eltype. Borrows
# the M/N offset buffers, which the nest refills before reading them again.
function _panel_copy!(target::ContractPlan, n_block_start::Int, n_block_length::Int, load::Bool)
    ws = target.workspace
    panel = ws.c_panel
    C = target.Cstorage
    m_length = axis_length(target.mgroup)
    m_block = target.blocking.m_block
    fill_offsets!((ws.n_buf_B, ws.n_buf_C), target.ngroup, n_block_start, n_block_length)
    m_block_start = 0
    while m_block_start < m_length
        m_block_length = min(m_block, m_length - m_block_start)
        fill_offsets!((ws.m_buf_A, ws.m_buf_C), target.mgroup, m_block_start, m_block_length)
        for j in 1:n_block_length, i in 1:m_block_length
            c = target.Cbase + ws.m_buf_C[i] + ws.n_buf_C[j] + 1
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
        alphaT::T, betaT::T, ::_NestPath{UNPACKED_B, AFF, SPLIT}, target
    ) where {T, K, UNPACKED_B, AFF, SPLIT}
    # GUARDRAIL: reals per sliver per K step address the packed panels;
    # `m_tile`/`n_tile` count register-tile rows. They differ for complex
    # kernels.
    a_sliver_width, b_sliver_width = sliver_width(kernel)

    atransform = plan.atransform
    btransform = plan.btransform

    split_a, split_b = SPLIT
    mgroup = split_a ? _split_group(plan.mgroup, plan.mpack) : plan.mgroup
    ngroup = split_b ? _split_group(plan.ngroup, plan.npack) : plan.ngroup

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
        cplan = _panel_enter!(target, plan, n_block_start, n_block_length, betaT)
        n_tiles = cld(n_block_length, n_tile)
        (rng_nB, rng_nC) = if n_ramp
            _ramp_slivers!(
                ws.n_desc_B, ws.n_desc_C, n_step[1], n_step[2], n_block_start,
                n_block_length, n_tile, n_tiles
            )
        else
            fill_offsets!((ws.n_buf_B, ws.n_buf_C), ngroup, n_block_start, n_block_length)
            _classify_slivers!(
                ws.n_desc_B, ws.n_desc_C, ws.n_buf_B, ws.n_buf_C,
                n_block_length, n_tile, n_tiles
            )
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
                fill_offsets!((ws.k_buf_A, ws.k_buf_B), plan.kgroup, k_block_start, k_block_length)
                dA = describe_block(ws.k_buf_A, 0, k_block_length)
                dB = describe_block(ws.k_buf_B, 0, k_block_length)
                (
                    dA, dB,
                    descriptor_offset_range(dA, ws.k_buf_A, 0),
                    descriptor_offset_range(dB, ws.k_buf_B, 0),
                )
            end

            colsA_k = _axis_of(dK_A, ws.k_buf_A, 0, aff_kA)
            rowsB_k = _axis_of(dK_B, ws.k_buf_B, 0, aff_kB)

            # Hoisted bounds checks (B here, A and C per M block): each
            # rectangle is exactly the union of the per-sliver/per-tile
            # regions, and the storage check compares only range extremes, so
            # the `@inbounds` pack and tile calls below touch nothing unchecked.
            checked_span_bounds(plan.Bbase, rng_kB, rng_nB, lenB)

            beta_eff = firstpanel ? betaT : one(T)

            if split_b
                _pack_block_transposed!(
                    packed_panel(ws.packed_b, 1, b_sliver_width * k_block_length * n_tiles), sliver_spec(kernel, 2),
                    plan.Bstorage, plan.Bbase, ws.n_buf_B, rowsB_k, btransform, n_block_length, k_block_length,
                    plan.npack
                )
            elseif !UNPACKED_B
                for n_tile_index in 0:(n_tiles - 1)
                    n_tile_start = n_tile_index * n_tile
                    bpanel = _sliver_panel(ws.packed_b, b_sliver_width, k_block_length, n_tile_index)
                    colsB = _axis_of(ws.n_desc_B[n_tile_index + 1], ws.n_buf_B, n_tile_start, aff_nB)
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
                        ws.m_desc_A, ws.m_desc_C, m_step[1], m_step[2], m_block_start,
                        m_block_length, m_tile, m_tiles
                    )
                else
                    fill_offsets!((ws.m_buf_A, ws.m_buf_C), mgroup, m_block_start, m_block_length)
                    _classify_slivers!(
                        ws.m_desc_A, ws.m_desc_C, ws.m_buf_A, ws.m_buf_C,
                        m_block_length, m_tile, m_tiles
                    )
                end

                checked_span_bounds(plan.Abase, rng_mA, rng_kA, lenA)
                checked_span_bounds(cplan.Cbase, rng_mC, rng_nC, lenC)

                if split_a
                    _pack_block_transposed!(
                        packed_panel(ws.packed_a, 1, a_sliver_width * k_block_length * m_tiles), sliver_spec(kernel, 1),
                        plan.Astorage, plan.Abase, ws.m_buf_A, colsA_k, atransform, m_block_length, k_block_length,
                        plan.mpack
                    )
                else
                    for m_tile_index in 0:(m_tiles - 1)
                        m_tile_start = m_tile_index * m_tile
                        apanel = _sliver_panel(ws.packed_a, a_sliver_width, k_block_length, m_tile_index)
                        rowsA = _axis_of(ws.m_desc_A[m_tile_index + 1], ws.m_buf_A, m_tile_start, aff_mA)
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

        _panel_exit!(target, n_block_start, n_block_length)
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
        colsC = _axis_of(ws.n_desc_C[n_tile_index + 1], ws.n_buf_C, n_tile_start, aff_nC)
        for m_tile_index in 0:(m_tiles - 1)
            m_tile_start = m_tile_index * m_tile
            apanel = _sliver_panel(ws.packed_a, a_sliver_width, k_block_length, m_tile_index)
            rowsC = _axis_of(ws.m_desc_C[m_tile_index + 1], ws.m_buf_C, m_tile_start, aff_mC)
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
