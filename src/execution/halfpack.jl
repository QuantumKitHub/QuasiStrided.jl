# "Half-packing": pack B exactly as `execute!` does, but read A straight out
# of its own storage when it already has the one layout the SIMD microkernel
# can consume without a copy. Opt-in only, through `execute_half_packed!`.

"""
    UnpackedAView{S,C<:Axis}(storage::S, rowbase::Int, cols::C, nrows::Int)

Stands in for a packed A micro-panel when A is read in place: `nrows` logical
rows (the kernel's `mr`), unit-stride in storage, at `axis_length(cols)` K
steps. Logical row `i`, K step `p` lives at zero-based storage address
`rowbase + i + axis_offset(cols, p)` -- the address
[`_a_step_offset`](@ref) returns for it, which `_accumulate_step` hands
straight to `panel_vload`. So unlike a `PackedPanel`, whose `panel_vload`
offset is relative to the panel pointer, this view's offset IS the full
storage address; `rowbase` already includes the operand's own base offset.
`cols` may be affine or scattered (any `Axis`); only the row direction has to
be unit-stride, because that is what makes one `Vec{W,T}` load valid.

Borrowed, like `PackedPanel`: `cols` may be a `PtrScatterAxis` into a
workspace buffer, so the caller keeps that buffer alive. `length` is the
packed-equivalent element count `nrows * axis_length(cols)` -- what
`_execute_tile_prologue!` compares against `packed_a_length(kernel, kc)` --
and NOT the storage length. Not bounds-checked; see
[`execute_half_packed!`](@ref) for where the check is made.
"""
struct UnpackedAView{S, C <: Axis}
    storage::S
    rowbase::Int
    cols::C
    nrows::Int
end

Base.length(u::UnpackedAView) = u.nrows * axis_length(u.cols)

# The K-step address of an in-place A. Does not depend on `kernel` (the
# packed formula does); the argument is there only so both methods share one
# call shape in `_accumulate_step`.
@inline _a_step_offset(u::UnpackedAView, kernel, i::Int, p::Int) =
    u.rowbase + i + axis_offset(u.cols, p)

# Dense storage only, pinned in the signature: any other storage (or an
# element-type mismatch, e.g. complex storage under a real inner kernel) is a
# MethodError here, never a wrong read. `o` is the zero-based storage address
# (tiles.jl's "address + 1 = index" convention), unchecked -- the caller has
# validated the whole A region once.
#
# DELIBERATELY NO `panel_load` METHOD: `ScalarKernel`'s K step
# (src/microkernels/scalar.jl) addresses A through `packed_a_offset` directly,
# not through `_a_step_offset`, so an `UnpackedAView` reaching it would read
# packed-layout addresses out of unpacked storage. Leaving `panel_load`
# undefined makes that a MethodError instead of a silently wrong result.
@inline function panel_vload(
        ::Type{Vec{W, T}}, u::UnpackedAView{S}, o::Int
    ) where {W, T, S <: DenseVector{T}}
    s = u.storage
    return GC.@preserve s vload(Vec{W, T}, pointer(s) + sizeof(T) * o)
end

# Type-only half of the eligibility test; folds to a constant per plan type.
# `_copies_unchanged` is the same value gate `_pack_a_contiguous_eligible`
# uses: for a real `T` both of the driver's transforms are the identity (and
# `plan_contract` only ever stores `identity` for a real `T` anyway).
@inline _half_pack_kernel_eligible(::SIMDKernel) = true
@inline _half_pack_kernel_eligible(::Any) = false

@inline function _half_pack_static_eligible(plan::ContractPlan{T}) where {T}
    return T <: Real && _half_pack_kernel_eligible(plan.kernel) &&
        plan.Astorage isa DenseVector{T} && _copies_unchanged(plan.atransform, T)
end

# The M composite's whole-block descriptors `(dM_A, dM_C)` and the eligibility
# verdict. Only called once `Qm == mr(kernel)`, so this is the single register
# sliver. For an affine-ramp composite (the common case) they come from
# `_ramp_descriptor` in closed form, `==` to what `describe_block` would
# produce (see src/execution/macrokernel.jl), and no buffer is touched;
# otherwise they are materialized into the macro M buffers (`ws.m_buf_A`/
# `m_buf_C`, at least `mc_eff >= mr` long). An irregular `dM_C`'s
# `PtrScatterAxis` then borrows `ws.m_buf_C`, which nothing else in the
# half-packed path writes.
@inline function _half_pack_m_descriptors!(plan::ContractPlan)
    ws = plan.workspace
    Qm = axis_length(plan.mgroup)
    (m_ramp, m_step) = affine_ramp(plan.mgroup)
    (dM_A, dM_C) = if m_ramp
        (_ramp_descriptor(m_step[1], 0, Qm), _ramp_descriptor(m_step[2], 0, Qm))
    else
        block_descriptors!((ws.m_buf_A, ws.m_buf_C), plan.mgroup, 0, Qm)
    end
    ok = dM_A.regular && _unit_stride_rows(AffineAxis(dM_A.base, dM_A.stride, dM_A.count))
    return (ok, dM_A, dM_C)
end

"""
    _half_pack_a_eligible(plan::ContractPlan) -> Bool

Whether [`execute_half_packed!`](@ref) can read A in place for `plan`. The
same shape of test as `_pack_a_contiguous_eligible`
(src/packing/pack_contiguous.jl), lifted from one sliver to the whole
contraction: real element type, a [`SIMDKernel`](@ref), dense
(`DenseVector{T}`) A storage, a copy-unchanged A transform, and the whole M
extent equal to exactly one full register tile, `axis_length(plan.mgroup) ==
mr(plan.kernel)`, whose A map is a unit-stride `AffineAxis`. All but the last
two clauses fold at compile time. On a non-ramp M composite this writes the
plan's M offset buffers (scratch space every driver refills before reading).
"""
function _half_pack_a_eligible(plan::ContractPlan)
    _half_pack_static_eligible(plan) || return false
    axis_length(plan.mgroup) == mr(plan.kernel) || return false
    return _half_pack_m_descriptors!(plan)[1]
end

"""
    execute_half_packed!(plan::ContractPlan, alpha::Number, beta::Number)

Unexported and opt-in: nothing in [`plan_contract`](@ref) or
[`contract!`](@ref) dispatches here. Same contract and result as
[`execute!`](@ref) on every plan; on an eligible one it skips packing A.

When [`_half_pack_a_eligible`](@ref) holds, the whole M extent is one full
`mr`-row register sliver whose rows are unit-stride in A's (dense) storage, so
each K step of A is already `mr` consecutive elements -- exactly what one
`_pack_a_contiguous!` step would copy into the packed panel. Instead of the
copy, the microkernel reads those elements in place through an
[`UnpackedAView`](@ref), which `_accumulate_step` addresses via
`_a_step_offset` (resolved on the view's type at compile time). B is packed
exactly as `execute!` packs it, and the loop nest is `execute!`'s with the
`ic` loop and the M-sliver loop each fixed at one iteration: `jc` over N in
`nc` blocks, `pc` over K in `kc` panels, then every N-sliver's micro-tile
against the one A view. K may be scattered; B and C may have any layout.

On any ineligible plan, and for the empty-output and `Qk == 0 || alpha == 0`
short-circuits, this is exactly `execute!(plan, alpha, beta)`. The function is
therefore always safe to call: it is never wrong, only sometimes no faster.

Per K step the arithmetic is `execute!`'s: the same `_accumulate_step`, FMA
order, K blocking and store, on the same A values read from a different
address. Results are therefore bitwise identical to `execute!`'s (and to
`execute_tilewise!`'s) in the eligible case -- observed so in
test/execution/test_halfpack.jl, which compares with `==`.

Bounds: as in `execute!`, storage bounds are validated once per block with
[`checked_span_bounds`](@ref) -- C once per `jc` block, B and A once per
`(jc, pc)` panel -- before any read or write, and the tile calls inside go
through `unsafe_pack_b!`/`unsafe_execute_tile!`. The A check covers exactly
the addresses the in-place vector loads read (the sliver's `mr` rows against
the panel's K columns); the `UnpackedAView` loads themselves are unchecked.
Allocation-free. Returns `plan.Cstorage`.

**Measured, 2026-09-25, jobs 7110225 (ccq rome/znver2, AVX2) and 7110226 (ccq
icelake-server, AVX-512), after rebasing onto the analytical blocking model
and `FMAddSubKernel` (main commits 454a5ec/bc2c917/c5254aa) -- superseding an
earlier measurement (jobs 7110151/7110152) taken under the old hand-tuned
blocking constants, since this path reads `plan.blocking` and could in
principle have shifted under the new model; it did not, materially: see
`benchmark/bench_half_packing.jl` and its `benchmark/submit_half_packing.sh`**,
sweeping K/N at `M = mr(kernel)` exactly (the only eligible shape): a real but
modest win, roughly 1.01-1.08x across most of the sweep, peaking around K=N
in the 16-256 range (up to ~1.05x Float64/AVX2, ~1.05x Float64/AVX-512,
~1.03-1.08x Float32/AVX-512) and settling to ~1.0-1.02x at K,N >= ~1024 as B's
packing/compute (which scales with N*K, unlike A's fixed `mr`-row packing cost
that scales with K alone) increasingly dominates the total. One regression
pocket: on AVX2 at the tiniest K/N (1-8), this path is 6-20% SLOWER than
`execute!` -- the eligibility check and loop setup cost more than the (tiny)
packing they avoid there; AVX-512 mostly does not show this (one noise-level
dip at Float64 K=N=4, ~0.93x). Net: worth keeping as opt-in exactly as
shipped; auto-wiring into `plan_contract`'s dispatch would
need an additional lower K/N gate to avoid the AVX2 regression pocket, which
has not been attempted here.

The measurement above used the SIMPLEST eligible fixture only: a plain,
contiguous, dense M-by-K/K-by-N matrix pair at the default kernel shape.
**Extended, 2026-09-26, jobs 7111552 (rome/AVX2) and 7111553
(icelake-server/AVX-512), same canary discipline**, to check this
generalizes, since most real multi-label tensor contractions do NOT look
like a plain two-index GEMM:

  * *Non-ramp K*: A's own rows are still unit-stride (eligible), but the K
    composite as a whole (both operands' maps together) is not one affine
    ramp -- e.g. `A[m,k1,k2]` dense against `B[k2,k1,n]` dense, same K label
    order, incompatible per-operand strides -- so the in-place A read takes
    the indexed `PtrScatterAxis` path, not a compile-time affine stride. Same
    band, 1.01-1.05x typical in the mid-range, same near-parity/slight-loss
    pocket at the tiniest K/N on AVX2 (0.93-0.96x); one noisy outlier
    (1.46x at a tiny, sub-microsecond K=16/N=8 point on AVX-512) is a
    small-sample artifact, not a new regime.
  * *Alternate register shapes*: every non-default menu shape
    (`kernel_shapes(T)`) also shows a positive 1.01-1.07x win at a
    fixed mid-range (K,N) = (128,128) -- not an artifact of one specific
    `(mr,nr,W)`.

So the modest-win/AVX2-tiny-K-regression verdict above is not an artifact of
the simplest fixture; it holds across the realistic non-ramp-K case and
every measured register shape.
"""
function execute_half_packed!(plan::ContractPlan{T}, alpha::Number, beta::Number) where {T}
    alphaT = convert(T, alpha)
    betaT = convert(T, beta)

    kernel = plan.kernel
    Qm = axis_length(plan.mgroup)
    Qn = axis_length(plan.ngroup)
    Qk = axis_length(plan.kgroup)

    # Empty output, and the beta-only pass: nothing for A to be skipped in.
    (Qm == 0 || Qn == 0 || Qk == 0 || iszero(alphaT)) && return execute!(plan, alpha, beta)

    (_half_pack_static_eligible(plan) && Qm == mr(kernel)) || return execute!(plan, alpha, beta)
    (ok, _, _) = _half_pack_m_descriptors!(plan)
    ok || return execute!(plan, alpha, beta)

    # Through `execute!`'s dispatch barrier (src/execution/barrier.jl), so the
    # nest is compiled only for the axis types this plan's maps produce.
    _execute_across_barrier!(
        plan, alphaT, betaT, _half_pack_path(plan.mgroup, plan.ngroup, plan.kgroup)
    )
    return plan.Cstorage
end

function _execute_path!(plan::ContractPlan{T}, alphaT::T, betaT::T, path::_HalfPackPath) where {T}
    # Recomputed rather than carried across the barrier: two descriptors,
    # already known to be eligible.
    (_, dM_A, dM_C) = _half_pack_m_descriptors!(plan)
    ws = plan.workspace
    Astorage = plan.Astorage
    # `ws`: the packed B panel and every `PtrScatterAxis` borrow its buffers.
    # `Astorage`: the in-place A loads are raw-pointer loads.
    GC.@preserve ws Astorage begin
        _execute_half_packed_nest!(
            plan, ws, plan.kernel, dM_A, dM_C, axis_length(plan.ngroup), axis_length(plan.kgroup),
            plan.blocking.kc, plan.blocking.nc, alphaT, betaT, path
        )
    end
    return nothing
end

# `_execute_nest!` (src/execution/execute.jl) with the M side fixed at one full
# sliver and no A packing. The B half -- descriptors, the hoisted B check, the
# pack -- is `_execute_nest!`'s verbatim.
function _execute_half_packed_nest!(
        plan::ContractPlan{T}, ws, kernel::K, dM_A::BlockDescriptor, dM_C::BlockDescriptor,
        Qn::Int, Qk::Int, kc_eff::Int, nc_eff::Int, alphaT::T, betaT::T,
        ::_HalfPackPath{AFF}
    ) where {T, K, AFF}
    # The static affine-axis flags (`_NestPath`'s order; M in A is unused).
    _, aff_mC, aff_nB, aff_nC, aff_kA, aff_kB = map(Val, AFF)
    MRk = mr(kernel)
    NRk = nr(kernel)
    NRp = packed_b_per_k(kernel)

    btransform = plan.btransform

    (n_ramp, n_step) = affine_ramp(plan.ngroup)
    (k_ramp, k_step) = affine_ramp(plan.kgroup)

    lenA = length(plan.Astorage)
    lenB = length(plan.Bstorage)
    lenC = length(plan.Cstorage)

    # The one M sliver: A's rows are `dM_A`, unit-stride (checked by the
    # caller); C's rows are `dM_C`, any layout. Both ranges are fixed for the
    # whole call.
    rng_mA = descriptor_offset_range(dM_A, ws.m_buf_A, 0)
    rng_mC = descriptor_offset_range(dM_C, ws.m_buf_C, 0)
    arowbase = plan.Abase + dM_A.base

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

        # HOISTED CHECK (C) -- every micro-tile of this jc block: the one M
        # sliver's rows against the union of the N slivers' columns. Its
        # region does not depend on pc, so once per jc, before any write.
        checked_span_bounds(plan.Cbase, rng_mC, rng_nC, lenC)

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

            # HOISTED CHECK (B) -- the whole B panel of this (jc, pc), exactly
            # as `_execute_nest!`'s check 1.
            checked_span_bounds(plan.Bbase, rng_kB, rng_nB, lenB)

            # HOISTED CHECK (A) -- the A region this panel reads in place: the
            # sliver's `mr` rows against this panel's K columns. This is the
            # ONLY check covering the `UnpackedAView` vector loads, which are
            # raw-pointer reads; it is exactly `_execute_nest!`'s check 2.
            checked_span_bounds(plan.Abase, rng_mA, rng_kA, lenA)

            beta_eff = firstpanel ? betaT : one(T)

            # Pack the whole B panel for this (jc, pc): every N-sliver.
            for s in 0:(n_slivers - 1)
                sfirst = s * NRk
                colsB = _axis_of(ws.n_desc_B[s + 1], ws.n_buf_B, sfirst, aff_nB)
                bpanel = _sliver_panel(ws.packed_b, NRp, kblock, s)
                _pack_sliver!(
                    unsafe_pack_b!, bpanel, plan.Bstorage, plan.Bbase, rowsB_k, colsB,
                    kernel, btransform
                )
            end

            # --- loop 2: jr over N-slivers; loop 1 (ir) is the one M sliver ---
            _half_packed_micro_tiles!(
                kernel, plan.Cstorage, plan.Cbase, dM_C, ws,
                plan.Astorage, arowbase, colsA_k, MRk, NRk, NRp,
                n_slivers, kblock, alphaT, beta_eff, aff_mC, aff_nC
            )

            firstpanel = false
            pc += kblock
        end

        jc += nblock
    end

    return plan.Cstorage
end

# Function barrier over the K axis type (the `_axis_of` Union dies here, per
# the guardrail at the top of src/execution/macrokernel.jl), so the
# `UnpackedAView` built below is concretely typed and the micro-tile calls
# specialize on it. `unsafe_execute_micro_tile!`'s precondition is the caller's
# hoisted C check. `@noinline` for compile time, as `_micro_tiles_packed_b!`
# (src/execution/execute.jl): inlined, this nest's inference was 60 of the
# 207 s test/execution/test_halfpack.jl spends inferring; out of line, 14.
@noinline function _half_packed_micro_tiles!(
        kernel::K, Cstorage::SC, Cbase::Int, dM_C::BlockDescriptor, ws,
        Astorage::SA, arowbase::Int, colsA_k::CA, MRk::Int, NRk::Int, NRp::Int,
        n_slivers::Int, kblock::Int, alphaT, beta_eff, aff_mC::Val{MC}, aff_nC::Val{NC}
    ) where {K, SC, SA, CA <: Axis, MC, NC}
    apanel = UnpackedAView(Astorage, arowbase, colsA_k, MRk)
    for s in 0:(n_slivers - 1)
        sfirst = s * NRk
        rowsC = _axis_of(dM_C, ws.m_buf_C, 0, aff_mC)
        colsC = _axis_of(ws.n_desc_C[s + 1], ws.n_buf_C, sfirst, aff_nC)
        bpanel = _sliver_panel(ws.packed_b, NRp, kblock, s)
        unsafe_execute_micro_tile!(
            kernel, Cstorage, Cbase, rowsC, colsC,
            apanel, bpanel, kblock, alphaT, beta_eff
        )
    end
    return nothing
end
