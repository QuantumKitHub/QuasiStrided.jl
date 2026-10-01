# "Unpacked B": the microkernel reads B in place instead of from a packed
# panel. The kernels consume B one (complex) scalar at a time and broadcast
# it, so any K axis, column offsets and `AbstractVector` storage work.
# Packing B buys locality and padding columns at an O(N*K) gather per
# (N, K) block, which at small M costs as much as the kernel itself.

# Stands in for a packed B micro-panel of `NR` columns: column `j`, K step
# `p` is storage address `colbase[j] + ksteps[p]` (base included). Padding
# columns of a tail sliver alias the last valid column, so every address read
# is inside the hoisted B check; their results are never stored. `length` is
# the packed-equivalent count that `_execute_tile_prologue!` checks, not the
# storage length.
struct UnpackedBView{S, K <: AbstractVector{Int}, NR, F}
    storage::S
    colbase::NTuple{NR, Int}
    ksteps::K
    per_k::Int
    transform::F
end

Base.length(u::UnpackedBView) = u.per_k * length(u.ksteps)

# `@inbounds`: the caller has checked the whole B panel region.
@inline function _unpacked_b_element(u::UnpackedBView{S, K, NR, F}, j::Int, p::Int) where {S, K, NR, F}
    @inbounds z = u.storage[u.colbase[j] + u.ksteps[p] + 1]
    return u.transform(z)
end

@inline _b_step_load(u::UnpackedBView, kernel, j::Int, p::Int) =
    convert(scalartype(kernel), _unpacked_b_element(u, j, p))

# Complex kernels: the planar `(re, im)` pair.
@inline function _b_step_load2(u::UnpackedBView, kernel, j::Int, p::Int)
    z = convert(scalartype(kernel), _unpacked_b_element(u, j, p))
    return (real(z), imag(z))
end

# The kernels whose K step reads B through `_b_step_load`/`_b_step_load2`
# (not 1m, whose inner kernel walks the planar panel, nor the scalar one).
@inline _unpacked_b_kernel_eligible(::SIMDKernel) = true
@inline _unpacked_b_kernel_eligible(::PlanarKernel) = true
@inline _unpacked_b_kernel_eligible(::FMAddSubKernel) = true
@inline _unpacked_b_kernel_eligible(::ComplexRealKernel) = true
@inline _unpacked_b_kernel_eligible(::Any) = false

# The same by method, to predict the path before the kernel exists.
@inline _unpacked_b_method_eligible(::Union{RealMethod, PlanarMethod, FMAddSubMethod, ComplexRealMethod}) = true
@inline _unpacked_b_method_eligible(::Any) = false

# `:always`/`:never` override the rule below, for benchmarks and tests.
const _UNPACKED_B_MODE = Ref{Symbol}(:auto)

# Whether B is read in place (given an eligible kernel): small M, and every B
# column contiguous along K. Packing pays off only through reuse across M
# slivers, while `n_tile` contiguous columns read in place cost the kernel
# nothing. A large K stride in B (each step its own cache line, and for a
# power of two only a few L1 sets) is what packing exists for.
@inline function _unpacked_b_rule(mgroup::AxisGroup, kgroup::AxisGroup)
    mode = _UNPACKED_B_MODE[]
    mode === :always && return true
    mode === :never && return false
    axis_length(mgroup) <= _UNPACKED_B_MMAX || return false
    (k_ramp, k_step) = affine_ramp(kgroup)
    return k_ramp && abs(k_step[2]) == 1
end

# Beyond this M the packed B's reuse wins.
const _UNPACKED_B_MMAX = 256

# The view for one N sliver; `buf`/`n_tile_start` locate an irregular sliver's
# offsets in `ws.n_buf_B`.
@inline function _unpacked_b_view(
        kernel::DescriptorKernel{MR, NR, T}, storage::S, base::Int,
        d::BlockDescriptor, buf::Vector{Int}, n_tile_start::Int, ksteps::K, transform::F
    ) where {MR, NR, T, S, K <: AbstractVector{Int}, F}
    last = d.count - 1
    colbase = ntuple(Val(NR)) do j1
        jj = min(j1 - 1, last)
        off = d.regular ? d.base + jj * d.stride : (@inbounds buf[n_tile_start + jj + 1])
        base + off
    end
    return UnpackedBView(storage, colbase, ksteps, sliver_width(kernel, 2), transform)
end

# The tile loops of `_execute_nest!` with B read in place. A barrier over the K
# axis type, so each view is concretely typed. `@noinline` so that the
# packed-B nest does not grow by this body. The view's addresses are within
# the caller's B check.
@noinline function _micro_tiles_unpacked_b!(
        kernel::K, plan::ContractPlan, ws, rowsB_k::KA, m_tiles::Int, n_tiles::Int,
        m_tile::Int, n_tile::Int, a_sliver_width::Int, k_block_length::Int, alphaT, beta_eff,
        aff_mC::Val{MC}, aff_nC::Val{NC}
    ) where {K, KA <: AbstractVector{Int}, MC, NC}
    Bstorage = plan.Bstorage
    Bbase = plan.Bbase
    btransform = plan.btransform
    for n_tile_index in 0:(n_tiles - 1)
        n_tile_start = n_tile_index * n_tile
        bview = _unpacked_b_view(
            kernel, Bstorage, Bbase, ws.n_desc_B[n_tile_index + 1], ws.n_buf_B, n_tile_start, rowsB_k, btransform
        )
        colsC = _axis_of(ws.n_desc_C[n_tile_index + 1], ws.n_buf_C, n_tile_start, aff_nC)
        for m_tile_index in 0:(m_tiles - 1)
            m_tile_start = m_tile_index * m_tile
            apanel = _sliver_panel(ws.packed_a, a_sliver_width, k_block_length, m_tile_index)
            rowsC = _axis_of(ws.m_desc_C[m_tile_index + 1], ws.m_buf_C, m_tile_start, aff_mC)
            unsafe_execute_micro_tile!(
                kernel, plan.Cstorage, plan.Cbase, rowsC, colsC,
                apanel, bview, k_block_length, alphaT, beta_eff
            )
        end
    end
    return nothing
end
