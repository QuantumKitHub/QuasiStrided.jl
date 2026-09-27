# "Unpacked B": run the microkernel against B read in place from its own
# storage instead of from a packed panel. The mirror image of half-packing
# (src/execution/halfpack.jl), but far more general: the kernels only ever
# consume B one scalar (or one complex scalar) at a time and broadcast it, so
# B needs no layout at all to be read unpacked -- any affine or scattered K
# axis, any column offsets, any `AbstractVector` storage. Packing B buys
# locality (one contiguous stream instead of `nr` strided ones) and padding
# columns; it costs an O(N*K) scalar gather loop per (jc, pc) block, which at
# small M is comparable to the whole microkernel time (measured below).
#
# Selected automatically by `_use_unpacked_b` (see there for the measured
# rule) for every kernel that reads B through `_b_step_load`/`_b_step_load2`
# -- `SIMDKernel`, `PlanarKernel`, `FMAddSubKernel` -- and never for
# `OneMKernel` (its inner real kernel walks the planar B panel as `2kc` real
# steps, a layout only a packed panel has) or `ScalarKernel` (reference
# kernel, addresses the panel directly).

"""
    UnpackedBView{S,K<:Axis,NR,F}(storage, colbase::NTuple{NR,Int}, ksteps::K, per_k::Int, transform::F)

Stands in for a packed B micro-panel of `NR` logical columns when B is read in
place. Column `j` at K step `p` is the storage element at zero-based address
`colbase[j+1] + axis_offset(ksteps, p)`; `colbase` already includes the
operand's base offset. Padding columns of a tail sliver (`j >= n`) ALIAS the
sliver's last valid column, so every address the kernel reads is one the
hoisted B bounds check (`_execute_nest!`, check 1) has validated -- the kernel
computes garbage for those columns and the store discards it (`j < n`), exactly
as with a zero-padded packed panel, only without the zeros.

`transform` is the plan's `btransform` (`identity` or `conj`), applied to each
loaded element, so a conjugated complex B needs no packing either. `per_k` is
`packed_b_per_k(kernel)`: `length` is the packed-equivalent real count, what
`_execute_tile_prologue!` compares against `packed_b_length(kernel, kc)`, not
the storage length. Borrowed: `ksteps` may be a `PtrScatterAxis` into a
workspace buffer, which the caller keeps alive. Not bounds-checked; see
`_micro_tiles_unpacked_b!`.
"""
struct UnpackedBView{S, K <: Axis, NR, F}
    storage::S
    colbase::NTuple{NR, Int}
    ksteps::K
    per_k::Int
    transform::F
end

Base.length(u::UnpackedBView) = u.per_k * axis_length(u.ksteps)

# One element of B, transformed. `@inbounds`: the caller has validated the
# whole B panel region once (see `_micro_tiles_unpacked_b!`). Linear indexing,
# so any `AbstractVector` storage works -- `Memory`, `Vector`, or a wrapper.
@inline function _unpacked_b_element(u::UnpackedBView{S, K, NR, F}, j::Int, p::Int) where {S, K, NR, F}
    @inbounds z = u.storage[u.colbase[j + 1] + axis_offset(u.ksteps, p) + 1]
    return u.transform(z)
end

# Real kernels (`_accumulate_step`, src/microkernels/simd.jl).
@inline _b_step_load(u::UnpackedBView, kernel, j::Int, p::Int) = _unpacked_b_element(u, j, p)

# Complex kernels (`_accumulate_step_planar`/`_accumulate_step_fmaddsub`): the
# planar `(re, im)` pair of one complex element.
@inline function _b_step_load2(u::UnpackedBView, kernel, j::Int, p::Int)
    z = _unpacked_b_element(u, j, p)
    return (real(z), imag(z))
end

# Which kernels can take an `UnpackedBView`: exactly those whose K step reads
# B through `_b_step_load`/`_b_step_load2`. Folds to a constant per plan type.
@inline _unpacked_b_kernel_eligible(::SIMDKernel) = true
@inline _unpacked_b_kernel_eligible(::PlanarKernel) = true
@inline _unpacked_b_kernel_eligible(::FMAddSubKernel) = true
@inline _unpacked_b_kernel_eligible(::Any) = false

# The same, for the kernel an automatic `(shape, method)` choice builds
# (`_kernel_type`: `SIMDKernel`, `PlanarKernel`, `FMAddSubKernel`, `OneMKernel`),
# as `plan_contract` needs it to predict the path before the kernel exists.
@inline _unpacked_b_method_eligible(::Union{RealMethod, PlanarMethod, FMAddSubMethod}) = true
@inline _unpacked_b_method_eligible(::Any) = false

# Test/benchmark override of the automatic rule: `:auto` (the rule below),
# `:always`, `:never`. Read once per `execute!` (by `_select_path`).
const _UNPACKED_B_MODE = Ref{Symbol}(:auto)

"""
    _use_unpacked_b(plan::ContractPlan) -> Bool

Whether this `execute!` reads B in place (`UnpackedBView`) rather than packing
it. Requires an eligible kernel (`_unpacked_b_kernel_eligible`); then, under
`_UNPACKED_B_MODE[] === :auto`, the rule is

    Qm <= _UNPACKED_B_MMAX  &&  the K composite is an affine ramp whose B step is +-1

i.e. every B column the kernel streams is contiguous along K. Packing a B
sliver costs an O(nr*kc) scalar gather per (jc, pc) block that only the M
slivers reusing it repay, while reading `nr` contiguous columns in place costs
the kernel nothing measurable. With a LARGE K stride in B (B stored N-fastest,
or a scattered K), each K step touches its own cache line and, for a
power-of-two stride, only a few L1 sets: that is the case packing exists for,
and it stays packed.

Measured 2026-09-26, ccqlin038 (Cascade Lake, Julia 1.12.7), `:always` vs
`:never` on the same tree, back-to-back pairs, `execute!` time ratio
unpacked/packed:

    K unit-stride in B (C[a,c] = A[a,b] B[b,c] and the rank-4 gemm_ready cases):
      Float64    16^3 0.70  32^3 0.70  64^3 0.79  128^3 0.90  256^3 0.93  512^3 1.02
                 dim8_1_3_1 (8x512x8) 0.48   dim6_2_2_2 (36^3) 0.87   dim8_2_1_2 (64x8x64) 0.91
      ComplexF64 16^3 1.00  32^3 0.86  64^3 0.93  128^3 0.91  256^3 0.96  512^3 1.01
                 M=1 gemv (1x64x512 .. 1x256x4096, FMAddSub 8x8) 0.55-0.65   dim8_1_3_1 0.68
    K strided in B (B[c,b], stride N):
      Float64    16^3 0.71  64^3 0.94  256^3 1.20     ComplexF64 16^3 1.12  64^3 1.05  256^3 2.12
    K = 1 (outer product, rank-0 K group, step 0 -- excluded by the rule):
      Float64 ~1.00   ComplexF64 1.03-1.08

So the rule takes the unit-stride column and stops at M = 256, the last size
with a measured win on both dtypes; 512 is a wash. The strided-K wins at
small K are left on the table deliberately: the same rule would have to
model L1 set conflicts to avoid the 1.2-2.1x losses above.
"""
@inline _use_unpacked_b(plan::ContractPlan) =
    _unpacked_b_kernel_eligible(plan.kernel) && _unpacked_b_rule(plan.mgroup, plan.kgroup)

# Everything but the kernel's eligibility, on the plan's groups (`_select_path`
# also applies it before the plan exists, src/execution/execute.jl).
@inline function _unpacked_b_rule(mgroup::AxisGroup, kgroup::AxisGroup)
    mode = _UNPACKED_B_MODE[]
    mode === :always && return true
    mode === :never && return false
    axis_length(mgroup) <= _UNPACKED_B_MMAX || return false
    (k_ramp, k_step) = affine_ramp(kgroup)
    return k_ramp && abs(k_step[2]) == 1
end

# Largest M extent at which B is read in place under `:auto`; see
# `_use_unpacked_b`.
const _UNPACKED_B_MMAX = 256

# Build the view for N sliver `s` of the current (jc, pc) block. `d` is the
# sliver's B descriptor (`ws.n_desc_B[s+1]`), `buf`/`sfirst` locate its
# offsets in `ws.n_buf_B` when it is irregular. Padding columns alias the last
# valid one (see `UnpackedBView`). `NR` comes from the kernel's type, so the
# tuple is statically sized.
@inline function _unpacked_b_view(
        kernel::DescriptorKernel{MR, NR, T}, storage::S, base::Int,
        d::BlockDescriptor, buf::Vector{Int}, sfirst::Int, ksteps::K, transform::F
    ) where {MR, NR, T, S, K <: Axis, F}
    last = d.count - 1
    colbase = ntuple(Val(NR)) do j1
        jj = min(j1 - 1, last)
        off = d.regular ? d.base + jj * d.stride : (@inbounds buf[sfirst + jj + 1])
        base + off
    end
    return UnpackedBView(storage, colbase, ksteps, packed_b_per_k(kernel), transform)
end

# Loops 2/1 of `_execute_nest!` (src/execution/execute.jl) with every B sliver
# read in place: one `UnpackedBView` per N sliver, then the M slivers against
# it. Function barrier over the K axis type (the `_axis_of` Union dies here,
# per the guardrail at the top of src/execution/macrokernel.jl), so each view
# is concretely typed and the micro-tile calls specialize on it.
#
# `@noinline`: one call per (jc, pc, ic) block, and inlined it grows
# `_execute_nest!` for EVERY plan, including the packed ones -- measured
# 2026-09-27 (ccqlin038) as ~+4% on the packed-B TCCG case ccsd_2 dim16
# Float64 (7.2 -> 7.5 us), gone out of line, with no measurable cost to the
# unpacked 16^3/32^3 matmuls.
#
# Bounds: the addresses a view can read are `Bbase + colbase[j] + koffset(p)`
# for the sliver's own columns (padding aliases a valid one) and this panel's
# K offsets -- a subset of the `(rng_kB, rng_nB)` rectangle the caller has
# already passed through `checked_span_bounds` (check 1 of 3), which is why
# the element loads are `@inbounds`. C is covered by the caller's check 3.
@noinline function _micro_tiles_unpacked_b!(
        kernel::K, plan::ContractPlan, ws, rowsB_k::KA, m_slivers::Int, n_slivers::Int,
        MRk::Int, NRk::Int, MRp::Int, kblock::Int, alphaT, beta_eff,
        aff_mC::Val{MC}, aff_nC::Val{NC}
    ) where {K, KA <: Axis, MC, NC}
    Bstorage = plan.Bstorage
    Bbase = plan.Bbase
    btransform = plan.btransform
    for s in 0:(n_slivers - 1)
        sfirst = s * NRk
        colsC = _axis_of(ws.n_desc_C[s + 1], ws.n_buf_C, sfirst, aff_nC)
        bview = _unpacked_b_view(
            kernel, Bstorage, Bbase, ws.n_desc_B[s + 1], ws.n_buf_B, sfirst, rowsB_k, btransform
        )
        for r in 0:(m_slivers - 1)
            rfirst = r * MRk
            rowsC = _axis_of(ws.m_desc_C[r + 1], ws.m_buf_C, rfirst, aff_mC)
            apanel = _sliver_panel(ws.packed_a, MRp, kblock, r)
            unsafe_execute_micro_tile!(
                kernel, plan.Cstorage, plan.Cbase, rowsC, colsC,
                apanel, bview, kblock, alphaT, beta_eff
            )
        end
    end
    return nothing
end
