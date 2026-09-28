# `C *= beta` tile by tile, never reading A or B: the `Qk == 0 || alpha == 0`
# pass of `execute!` and `execute_tilewise!`. Uses the MR/NR-sized `tile_*`
# buffers, which is why those exist even under `oracle = false`.
function _scale_all_of_C!(plan, betaT::T, MRk::Int, NRk::Int, Qm::Int, Qn::Int) where {T}
    ws = plan.workspace
    m_bufs = (ws.tile_m_buf_A, ws.tile_m_buf_C)
    n_bufs = (ws.tile_n_buf_B, ws.tile_n_buf_C)
    mfirst = 0
    while mfirst < Qm
        mcount = min(MRk, Qm - mfirst)
        (_, dM_C) = block_descriptors!(m_bufs, plan.mgroup, mfirst, mcount)
        rowsC = _axis_of(dM_C, ws.tile_m_buf_C, 0)
        nfirst = 0
        while nfirst < Qn
            ncount = min(NRk, Qn - nfirst)
            (_, dN_C) = block_descriptors!(n_bufs, plan.ngroup, nfirst, ncount)
            colsC = _axis_of(dN_C, ws.tile_n_buf_C, 0)
            _scale_micro_tile!(plan.Cstorage, plan.Cbase, rowsC, colsC, betaT)
            nfirst += ncount
        end
        mfirst += mcount
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
    Qm = axis_length(plan.mgroup)
    Qn = axis_length(plan.ngroup)
    (Qm == 0 || Qn == 0) && return true
    if axis_length(plan.kgroup) == 0 || iszero(alphaT)
        _scale_all_of_C!(plan, betaT, mr(plan.kernel), nr(plan.kernel), Qm, Qn)
        return true
    end
    return false
end

# The path `execute!` runs past its short-circuits: the dot path, else the
# outer-product path, else the nest (B packed or read in place).
@inline _select_path(plan::ContractPlan{T}) where {T} = _select_path(
    T, _unpacked_b_kernel_eligible(plan.kernel),
    plan.Astorage, plan.Bstorage, plan.Cstorage, plan.mgroup, plan.ngroup, plan.kgroup, plan,
    _is_split(plan.mpack), _is_split(plan.npack)
)

# On the plan's parts, so `plan_contract` can predict the path before the
# plan exists (`_path_hint`). `unpack_ok`: the kernel admits unpacked B.
# `capacity`: the plan whose workspace must hold the dot path's vector, or
# `nothing` to assume it does. `split_a`/`split_b`: the plan packs A/B line by line.
@inline function _select_path(
        ::Type{T}, unpack_ok::Bool, Astorage, Bstorage, Cstorage,
        mgroup::AxisGroup, ngroup::AxisGroup, kgroup::AxisGroup, capacity,
        split_a::Bool = false, split_b::Bool = false
    ) where {T}
    Qm = axis_length(mgroup)
    Qn = axis_length(ngroup)
    Qk = axis_length(kgroup)
    if (Qm == 1 || Qn == 1) && _dot_applicable(T, Astorage, Bstorage, kgroup, Qm, Qn, Qk) &&
            _dot_capacity_ok(capacity)
        W = _dot_lanewidth(T)
        return Qm == 1 ? _lane_path(_DotPath{true}, W) : _lane_path(_DotPath{false}, W)
    end
    Qk == 1 && _outer_applicable(T, Astorage, Cstorage, mgroup, Qm) &&
        return _lane_path(_OuterPath, _dot_lanewidth(T))
    return _nest_path(
        unpack_ok && _unpacked_b_rule(mgroup, kgroup), mgroup, ngroup, kgroup, split_a, split_b
    )
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
    req.Astorage, req.Bstorage, req.Cstorage, req.mgroup, req.ngroup, req.kgroup, nothing
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
@inline _splits(plan::ContractPlan, ::_NestPath) = _is_split(plan.mpack) || _is_split(plan.npack)
@inline _split_path(plan::ContractPlan, ::_NestPath{U}) where {U} = _nest_path(
    U, plan.mgroup, plan.ngroup, plan.kgroup, _is_split(plan.mpack), _is_split(plan.npack)
)

# The prediction differs from `_select_path(plan)` in two inputs only: the
# kernel's unpacked-B eligibility (predicted from its method; folds) and the
# dot path's workspace capacity (assumed).
@inline _hint_holds(plan::ContractPlan, ::_NestPath) =
    _unpacked_b_method_eligible(complex_method(plan.kernel)) === _unpacked_b_kernel_eligible(plan.kernel)
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
    # Panels and `PtrScatterAxis`es borrow pointers into `ws`.
    GC.@preserve ws begin
        _execute_nest!(
            plan, ws, kernel, mr(kernel), nr(kernel),
            axis_length(plan.mgroup), axis_length(plan.ngroup), axis_length(plan.kgroup),
            plan.blocking.mc, plan.blocking.kc, plan.blocking.nc, alphaT, betaT, path
        )
    end
    return nothing
end

function _execute_nest!(
        plan::ContractPlan{T}, ws, kernel::K, MRk::Int, NRk::Int,
        Qm::Int, Qn::Int, Qk::Int, mc_eff::Int, kc_eff::Int, nc_eff::Int,
        alphaT::T, betaT::T, ::_NestPath{UNPACKED_B, AFF, SPLIT}
    ) where {T, K, UNPACKED_B, AFF, SPLIT}
    # GUARDRAIL: reals per sliver per K step address the packed panels;
    # `MRk`/`NRk` count register-tile rows. They differ for complex kernels.
    MRp = packed_a_per_k(kernel)
    NRp = packed_b_per_k(kernel)

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

    # --- loop 5: jc over N in steps of nc_eff ---
    jc = 0
    while jc < Qn
        nblock = min(nc_eff, Qn - jc)
        n_slivers = cld(nblock, NRk)
        (rng_nB, rng_nC) = if n_ramp
            _ramp_slivers!(
                ws.n_desc_B, ws.n_desc_C, n_step[1], n_step[2], jc,
                nblock, NRk, n_slivers
            )
        else
            fill_offsets!((ws.n_buf_B, ws.n_buf_C), ngroup, jc, nblock)
            _classify_slivers!(
                ws.n_desc_B, ws.n_desc_C, ws.n_buf_B, ws.n_buf_C,
                nblock, NRk, n_slivers
            )
        end

        # --- loop 4: pc over K in steps of kc_eff ---
        pc = 0
        firstpanel = true
        while pc < Qk
            kblock = min(kc_eff, Qk - pc)
            dK_A, dK_B, rng_kA, rng_kB = if k_ramp
                (
                    _ramp_descriptor(k_step[1], pc, kblock),
                    _ramp_descriptor(k_step[2], pc, kblock),
                    _ramp_offset_range(k_step[1], pc, kblock),
                    _ramp_offset_range(k_step[2], pc, kblock),
                )
            else
                fill_offsets!((ws.k_buf_A, ws.k_buf_B), plan.kgroup, pc, kblock)
                dA = describe_block(ws.k_buf_A, 0, kblock)
                dB = describe_block(ws.k_buf_B, 0, kblock)
                (
                    dA, dB,
                    descriptor_offset_range(dA, ws.k_buf_A, 0),
                    descriptor_offset_range(dB, ws.k_buf_B, 0),
                )
            end
            colsA_k = _axis_of(dK_A, ws.k_buf_A, 0, aff_kA)
            rowsB_k = _axis_of(dK_B, ws.k_buf_B, 0, aff_kB)

            # Hoisted bounds checks (B here, A and C per ic block): each
            # rectangle is exactly the union of the per-sliver/per-tile
            # regions, so the `unsafe_*` calls below read nothing unchecked.
            checked_span_bounds(plan.Bbase, rng_kB, rng_nB, lenB)

            beta_eff = firstpanel ? betaT : one(T)

            if split_b
                _pack_block_transposed!(
                    packed_b_plane_offset, b_format(kernel),
                    packed_panel(ws.packed_b, 1, NRp * kblock * n_slivers), kernel, Val(nr(kernel)), NRp,
                    plan.Bstorage, plan.Bbase, ws.n_buf_B, rowsB_k, btransform, nblock, kblock,
                    plan.npack
                )
            elseif !UNPACKED_B
                for s in 0:(n_slivers - 1)
                    sfirst = s * NRk
                    colsB = _axis_of(ws.n_desc_B[s + 1], ws.n_buf_B, sfirst, aff_nB)
                    bpanel = _sliver_panel(ws.packed_b, NRp, kblock, s)
                    _pack_sliver!(
                        unsafe_pack_b!, bpanel, plan.Bstorage, plan.Bbase, rowsB_k, colsB,
                        kernel, btransform
                    )
                end
            end

            # --- loop 3: ic over M in steps of mc_eff ---
            ic = 0
            while ic < Qm
                mblock = min(mc_eff, Qm - ic)
                m_slivers = cld(mblock, MRk)
                (rng_mA, rng_mC) = if m_ramp
                    _ramp_slivers!(
                        ws.m_desc_A, ws.m_desc_C, m_step[1], m_step[2], ic,
                        mblock, MRk, m_slivers
                    )
                else
                    fill_offsets!((ws.m_buf_A, ws.m_buf_C), mgroup, ic, mblock)
                    _classify_slivers!(
                        ws.m_desc_A, ws.m_desc_C, ws.m_buf_A, ws.m_buf_C,
                        mblock, MRk, m_slivers
                    )
                end

                checked_span_bounds(plan.Abase, rng_mA, rng_kA, lenA)
                checked_span_bounds(plan.Cbase, rng_mC, rng_nC, lenC)

                if split_a
                    _pack_block_transposed!(
                        packed_a_plane_offset, a_format(kernel),
                        packed_panel(ws.packed_a, 1, MRp * kblock * m_slivers), kernel, Val(mr(kernel)), MRp,
                        plan.Astorage, plan.Abase, ws.m_buf_A, colsA_k, atransform, mblock, kblock,
                        plan.mpack
                    )
                else
                    for r in 0:(m_slivers - 1)
                        rfirst = r * MRk
                        rowsA = _axis_of(ws.m_desc_A[r + 1], ws.m_buf_A, rfirst, aff_mA)
                        apanel = _sliver_panel(ws.packed_a, MRp, kblock, r)
                        _pack_sliver!(
                            unsafe_pack_a!, apanel, plan.Astorage, plan.Abase, rowsA, colsA_k,
                            kernel, atransform
                        )
                    end
                end

                # --- loop 2: jr over N-slivers; loop 1: ir over M-slivers ---
                if UNPACKED_B
                    _micro_tiles_unpacked_b!(
                        kernel, plan, ws, rowsB_k, m_slivers, n_slivers,
                        MRk, NRk, MRp, kblock, alphaT, beta_eff, aff_mC, aff_nC
                    )
                else
                    _micro_tiles_packed_b!(
                        kernel, plan, ws, m_slivers, n_slivers,
                        MRk, NRk, MRp, NRp, kblock, alphaT, beta_eff, aff_mC, aff_nC
                    )
                end

                ic += mblock
            end

            firstpanel = false
            pc += kblock
        end

        jc += nblock
    end

    return plan.Cstorage
end

# Loops 2/1 over one (jc, pc, ic) block. `@noinline`: one call per block, and
# the inlined microkernel is most of a nest's size; compile cost grows faster
# than linearly with function size.
@noinline function _micro_tiles_packed_b!(
        kernel::K, plan::ContractPlan, ws, m_slivers::Int, n_slivers::Int,
        MRk::Int, NRk::Int, MRp::Int, NRp::Int, kblock::Int, alphaT, beta_eff,
        aff_mC::Val{MC}, aff_nC::Val{NC}
    ) where {K, MC, NC}
    for s in 0:(n_slivers - 1)
        sfirst = s * NRk
        colsC = _axis_of(ws.n_desc_C[s + 1], ws.n_buf_C, sfirst, aff_nC)
        bpanel = _sliver_panel(ws.packed_b, NRp, kblock, s)
        for r in 0:(m_slivers - 1)
            rfirst = r * MRk
            rowsC = _axis_of(ws.m_desc_C[r + 1], ws.m_buf_C, rfirst, aff_mC)
            apanel = _sliver_panel(ws.packed_a, MRp, kblock, r)
            unsafe_execute_micro_tile!(
                kernel, plan.Cstorage, plan.Cbase, rowsC, colsC,
                apanel, bpanel, kblock, alphaT, beta_eff
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
              indC::NTuple{NC,Int}) where {NA,NB,NC}

Compute `C[indC] = alpha * sum_K A[indA] * B[indB] + beta * C[indC]`, with one
`Int` label per axis: a label in `indA` and `indB` but not `indC` is contracted
(K), and a label in `indC` and exactly one of `indA`/`indB` is free (M or N).
Any other label pattern, or a label repeated within one tuple, throws an
`ArgumentError`; matched labels of unequal axis length throw a
`DimensionMismatch`. Equivalent to
`execute!(plan_contract(C, A, indA, B, indB, indC), alpha, beta)` — use
those directly to reuse a plan across calls. Returns `C`.
"""
function contract!(
        C::StridedView, alpha::Number,
        A::StridedView, indA::NTuple{NA, Int},
        B::StridedView, indB::NTuple{NB, Int},
        beta::Number,
        indC::NTuple{NC, Int}
    ) where {NA, NB, NC}
    plan = plan_contract(C, A, indA, B, indB, indC)
    execute!(plan, alpha, beta)
    return C
end
