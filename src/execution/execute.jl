# Apply beta once to every element of C at MR x NR granularity, without
# reading A or B. Shared by the Qk==0/alpha==0 short-circuit of `execute!` and
# `execute_tilewise!`; uses the MR/NR-sized `tile_*` offset buffers, since a
# beta-only pass needs no blocking -- which is why those four, unlike the
# oracle-only `tw_*` ones, are allocated even under `oracle = false`.
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

Execution phase of [`contract!`](@ref): a BLIS five-loop macro-blocking nest
over `plan.blocking` (`nc`/loop 5, `kc`/loop 4, `mc`/loop 3), packing the
whole B panel once per `(jc,pc)` and the whole A panel once per `(jc,pc,ic)`,
then running `execute_tile!` over every micro-tile of that block (loops 2/1).
`beta` applies exactly once per output element (on the first K block only;
later ones accumulate with `beta = one(T)`). Empty output is a no-op; empty K
or `alpha == 0` applies `beta` once without reading `A`/`B`. Allocation-free.
Returns `plan.Cstorage`. See [`execute_tilewise!`](@ref) for the independent
tile-by-tile oracle this is checked against.

Storage-bounds validation is done **once per macro block**, not once per
sliver or per micro-tile: each of the three operand regions a `(jc, pc, ic)`
iteration touches is validated with one [`checked_span_bounds`](@ref) call
before anything is packed or written, and the packing/micro-kernel calls
inside it then go through `unsafe_pack_a!`/`unsafe_pack_b!`/
`unsafe_execute_tile!`. The test performed is exactly the conjunction of the
per-sliver and per-tile tests (see `checked_span_bounds`), so no address the
macro-blocking pack/execute path can reach is unvalidated, and a contraction
is rejected exactly when a per-tile check would reject it; `execute_tilewise!`
runs the per-tile checked path as an independent oracle for both the values
and the rejections.

Scoped to that path deliberately: the `Qk == 0 || alpha == 0` beta-only
short-circuit above it goes to `_scale_all_of_C!`, which performs no
storage-bounds check at all (it writes through `scale_tile!`'s `@inbounds`
path, relying on the `AxisGroup`s' own construction-time validation). It is
named here only so this paragraph is not read as a claim about it.
"""
function execute!(plan::ContractPlan{T}, alpha::Number, beta::Number) where {T}
    alphaT = convert(T, alpha)
    betaT = convert(T, beta)

    kernel = plan.kernel
    MRk = mr(kernel)
    NRk = nr(kernel)

    Qm = axis_length(plan.mgroup)
    Qn = axis_length(plan.ngroup)
    Qk = axis_length(plan.kgroup)

    # Empty output: no-op, nothing read or written at all.
    (Qm == 0 || Qn == 0) && return plan.Cstorage

    # Nothing to contract: beta-only pass, A and B never read.
    if Qk == 0 || iszero(alphaT)
        _scale_all_of_C!(plan, betaT, MRk, NRk, Qm, Qn)
        return plan.Cstorage
    end

    # Degenerate extents take a dedicated path instead of the nest when their
    # layout allows: a K-contiguous matrix operand with `M == 1` or `N == 1`
    # goes to the K-vectorized dot path (src/execution/dot.jl); a real
    # `K == 1` outer product with unit-stride M goes to the streaming
    # outer-product path (src/execution/outer.jl). Each declines (returns
    # `false`) on an ineligible plan, and the nest below runs.
    (Qm == 1 || Qn == 1) && _try_execute_dot!(plan, alphaT, betaT, Qm, Qn, Qk) &&
        return plan.Cstorage
    Qk == 1 && _try_execute_outer!(plan, alphaT, betaT, Qm, Qn) && return plan.Cstorage

    mc_eff = plan.blocking.mc
    kc_eff = plan.blocking.kc
    nc_eff = plan.blocking.nc

    # A reused workspace may be oversized: every extent below comes from the
    # *current* block, never from a buffer's length.
    ws = plan.workspace

    # Panels below borrow pointers into ws.packed_a/_b (src/packing/panel.jl), as do
    # the PtrScatterAxes from `_axis_of`; this is their lifetime.
    return GC.@preserve ws begin
        _execute_nest!(
            plan, ws, kernel, MRk, NRk, Qm, Qn, Qk,
            mc_eff, kc_eff, nc_eff, alphaT, betaT
        )
    end
end

# Split out so the `GC.@preserve` above has one obvious scope.
function _execute_nest!(
        plan::ContractPlan{T}, ws, kernel::K, MRk::Int, NRk::Int,
        Qm::Int, Qn::Int, Qk::Int, mc_eff::Int, kc_eff::Int, nc_eff::Int,
        alphaT::T, betaT::T
    ) where {T, K}
    # GUARDRAIL: reals per sliver per LOGICAL K step, which is what addresses
    # the packed panels. NOT interchangeable with `MRk`/`NRk`, which keep their
    # meaning everywhere else here (sliver counts, block extents,
    # `_classify_slivers!`): one counts register-tile rows, the other reals.
    # `MRp === MRk` for every real kernel (pinned in test/planning/test_kernel_selection.jl), so
    # the substitution below is provably the identity on the real path.
    MRp = packed_a_per_k(kernel)
    NRp = packed_b_per_k(kernel)

    atransform = plan.atransform
    btransform = plan.btransform

    # Resolved once per `execute!`, not per block: each composite's type is
    # concrete here, so `affine_ramp` unrolls to a few integer compares and the
    # `if`s below are cheap, predictable branches outside every inner loop.
    (m_ramp, m_step) = affine_ramp(plan.mgroup)
    (n_ramp, n_step) = affine_ramp(plan.ngroup)
    (k_ramp, k_step) = affine_ramp(plan.kgroup)

    # Hoisted storage-bounds validation: storage lengths read once here rather
    # than per sliver / per micro-tile.
    lenA = length(plan.Astorage)
    lenB = length(plan.Bstorage)
    lenC = length(plan.Cstorage)

    # Whether B is read in place by the microkernel instead of packed
    # (src/execution/unpackedb.jl). Decided once per call; the B pack below
    # is skipped and loops 2/1 run through `_micro_tiles_unpacked_b!`.
    unpacked_b = _use_unpacked_b(plan)

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
            fill_offsets!((ws.n_buf_B, ws.n_buf_C), plan.ngroup, jc, nblock)
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
            colsA_k = _axis_of(dK_A, ws.k_buf_A, 0)
            rowsB_k = _axis_of(dK_B, ws.k_buf_B, 0)

            # HOISTED CHECK 1 of 3 -- the whole B panel of this (jc, pc).
            # `rowsB_k` is shared by every N-sliver and `rng_nB` is the union
            # of the slivers' own column ranges, so this rectangle is exactly
            # the union of the addresses the `unsafe_pack_b!` calls below read;
            # see `checked_span_bounds` for why checking the union is
            # equivalent to checking each sliver, not weaker.
            checked_span_bounds(plan.Bbase, rng_kB, rng_nB, lenB)

            beta_eff = firstpanel ? betaT : one(T)

            # Pack the whole B panel for this (jc, pc): every N-sliver --
            # unless the kernel reads B in place for this plan.
            if !unpacked_b
                for s in 0:(n_slivers - 1)
                    sfirst = s * NRk
                    colsB = _axis_of(ws.n_desc_B[s + 1], ws.n_buf_B, sfirst)
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
                    fill_offsets!((ws.m_buf_A, ws.m_buf_C), plan.mgroup, ic, mblock)
                    _classify_slivers!(
                        ws.m_desc_A, ws.m_desc_C, ws.m_buf_A, ws.m_buf_C,
                        mblock, MRk, m_slivers
                    )
                end

                # HOISTED CHECK 2 of 3 -- the whole A panel of this
                # (jc, pc, ic): every M-sliver's rows against the shared K
                # columns.
                checked_span_bounds(plan.Abase, rng_mA, rng_kA, lenA)

                # HOISTED CHECK 3 of 3 -- every micro-tile of this (ic, jc)
                # block at once. The micro-tile loop below is the full cross
                # product of the M-sliver rows and the N-sliver columns, and
                # those two families partition the block's row and column
                # offset sets, so this rectangle is exactly their union. It
                # precedes every write to C, as a per-tile check would.
                checked_span_bounds(plan.Cbase, rng_mC, rng_nC, lenC)

                # Pack the whole A panel for this (jc, pc, ic): every M-sliver.
                for r in 0:(m_slivers - 1)
                    rfirst = r * MRk
                    rowsA = _axis_of(ws.m_desc_A[r + 1], ws.m_buf_A, rfirst)
                    apanel = _sliver_panel(ws.packed_a, MRp, kblock, r)
                    _pack_sliver!(
                        unsafe_pack_a!, apanel, plan.Astorage, plan.Abase, rowsA, colsA_k,
                        kernel, atransform
                    )
                end

                # --- loop 2: jr over N-slivers; loop 1: ir over M-slivers ---
                if unpacked_b
                    _micro_tiles_unpacked_b!(
                        kernel, plan, ws, rowsB_k, m_slivers, n_slivers,
                        MRk, NRk, MRp, kblock, alphaT, beta_eff
                    )
                else
                    for s in 0:(n_slivers - 1)
                        sfirst = s * NRk
                        colsC = _axis_of(ws.n_desc_C[s + 1], ws.n_buf_C, sfirst)
                        bpanel = _sliver_panel(ws.packed_b, NRp, kblock, s)
                        for r in 0:(m_slivers - 1)
                            rfirst = r * MRk
                            rowsC = _axis_of(ws.m_desc_C[r + 1], ws.m_buf_C, rfirst)
                            apanel = _sliver_panel(ws.packed_a, MRp, kblock, r)
                            unsafe_execute_micro_tile!(
                                kernel, plan.Cstorage, plan.Cbase, rowsC, colsC,
                                apanel, bpanel, kblock, alphaT, beta_eff
                            )
                        end
                    end
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

# ----------------------------------------------------------------------------
# contract!
# ----------------------------------------------------------------------------

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
