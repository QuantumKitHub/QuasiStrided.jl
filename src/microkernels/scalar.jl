"""
    ScalarKernel(::Val{MR}, ::Val{NR}, ::Type{T})

Scalar reference microkernel for register tile `(MR, NR)` and element type `T`.
"""
struct ScalarKernel{MR, NR, T} <: DescriptorKernel{MR, NR, T}
    descriptor::RealDescriptor{MR, NR, T}
end

ScalarKernel(::Val{MR}, ::Val{NR}, ::Type{T}) where {MR, NR, T} =
    ScalarKernel(Descriptor(Val(MR), Val(NR), T))

zero_accumulator(kernel::ScalarKernel{MR, NR, T}) where {MR, NR, T} = zeros(T, MR, NR)

# Extends `Base.accumulate` so `using QuasiStrided` does not clash with it.
function Base.accumulate(
        kernel::ScalarKernel{MR, NR, T}, acc::AbstractMatrix{T},
        packed_a::PA, packed_b::PB, k_block_length::Int
    ) where {MR, NR, T, PA <: PackedPanel, PB}
    k_block_length == 0 && return acc
    k_block_length > 0 || throw(ArgumentError("accumulate requires k_block_length >= 0, got k_block_length = $k_block_length"))
    @inbounds for p in 1:k_block_length
        for j in 1:NR
            bj = panel_load(packed_b, packed_b_offset(kernel, j, p))
            for i in 1:MR
                ai = panel_load(packed_a, packed_a_offset(kernel, i, p))
                acc[i, j] = muladd(ai, bj, acc[i, j])
            end
        end
    end
    return acc
end

function store_tile!(
        destination::Tile, acc::AbstractMatrix{T},
        alpha::T, beta::T, kernel::ScalarKernel
    ) where {T}
    m, n = _store_prologue!(destination, alpha, beta)
    (m == 0 || n == 0) && return destination
    @inbounds for j in 1:n, i in 1:m
        _axpby_tile!(destination, i, j, alpha, acc[i, j], beta)
    end
    return destination
end
