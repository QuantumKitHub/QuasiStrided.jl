# "Unpacked B": the microkernel reads B in place instead of from a packed
# panel. The kernels that read B by element (`reads_b_by_element`) broadcast
# one (complex) scalar at a time, so any K axis, column offsets and
# `AbstractVector` storage work. Packing B buys locality and padding columns
# at an O(N*K) gather per (N, K) block, which at small M costs as much as the
# kernel itself.

# Stands in for a packed B micro-panel of `NR` columns: column `j`, K step
# `p` is storage address `colbase[j] + ksteps[p]` (base included). Padding
# columns of a tail sliver alias the last valid column, so every address read
# is inside the hoisted B check; their results are never stored. `length` is
# the packed-equivalent count that `execute_tile!` checks, not the
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
@inline function unpacked_b_element(u::UnpackedBView{S, K, NR, F}, j::Int, p::Int) where {S, K, NR, F}
    @inbounds z = u.storage[u.colbase[j] + u.ksteps[p] + 1]
    return u.transform(z)
end
# Column pointer plus one K index shared by all columns: indexed, LLVM's SLP
# vectoriser can turn the `NR` per-column indices into vector arithmetic inside
# the K loop, which costs a vector register and spills accumulators.
@inline function unpacked_b_element(u::UnpackedBView{S, K, NR, F}, j::Int, p::Int) where {S <: DenseVector, K, NR, F}
    s = u.storage
    z = GC.@preserve s unsafe_load(pointer(s) + u.colbase[j] * sizeof(eltype(s)), @inbounds(u.ksteps[p]) + 1)
    return u.transform(z)
end

@inline b_scalar(u::UnpackedBView, kernel, j::Int, p::Int) =
    convert(scalartype(kernel), unpacked_b_element(u, j, p))

# Complex kernels: the planar `(re, im)` pair.
@inline function b_complex(u::UnpackedBView, kernel, j::Int, p::Int)
    z = convert(scalartype(kernel), unpacked_b_element(u, j, p))
    return (real(z), imag(z))
end

# The view for one N sliver; `buf`/`n_tile_start` locate an irregular sliver's
# offsets in `ws.n.offsets[1]`.
@inline function unpacked_b_view(
        kernel::Microkernel{MR, NR, T}, storage::S, base::Int,
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
