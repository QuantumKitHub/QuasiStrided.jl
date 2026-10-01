"""
    ScalarKernel(::Val{MR}, ::Val{NR}, ::Type{T})

Scalar reference microkernel for register tile `(MR, NR)` and element type `T`.
Its accumulator is an `NTuple{MR * NR, T}` holding the tile column-major.
"""
struct ScalarKernel{MR, NR, T} <: Microkernel{MR, NR, T}
    descriptor::RealDescriptor{MR, NR, T}
end

ScalarKernel(::Val{MR}, ::Val{NR}, ::Type{T}) where {MR, NR, T} =
    ScalarKernel(Descriptor(Val(MR), Val(NR), T))

zero_accumulator(::ScalarKernel{MR, NR, T}) where {MR, NR, T} = ntuple(_ -> zero(T), Val(MR * NR))

function add_tile(
        kernel::ScalarKernel{MR, NR, T}, acc::NTuple{N, T},
        packed_a::PA, packed_b::PB, k_block_length::Int
    ) where {MR, NR, T, N, PA <: PackedPanel, PB}
    k_block_length == 0 && return acc
    k_block_length > 0 || throw_negative_k_block_length(:add_tile, k_block_length)
    for p in 1:k_block_length
        acc = add_step(kernel, acc, packed_a, packed_b, p)
    end
    return acc
end

# One K step: element `(i, j)` of the tile is `acc[i + MR * (j - 1)]`.
@inline function add_step(
        kernel::ScalarKernel{MR, NR, T}, acc::NTuple{N, T}, packed_a, packed_b, p::Int
    ) where {MR, NR, T, N}
    return ntuple(Val(N)) do q
        i, j = mod1(q, MR), cld(q, MR)
        a = panel_load(packed_a, packed_a_offset(kernel, i, p))
        b = panel_load(packed_b, packed_b_offset(kernel, j, p))
        muladd(a, b, acc[q])
    end
end

function store_tile!(
        destination::Tile, acc::NTuple{N, T},
        alpha::T, beta::T, kernel::ScalarKernel{MR}
    ) where {MR, N, T}
    m, n = store_prologue!(destination, alpha, beta)
    (m == 0 || n == 0) && return destination
    if iszero(beta)
        @inbounds for j in 1:n, i in 1:m
            destination[i, j] = alpha * acc[i + MR * (j - 1)]
        end
    elseif isone(beta)
        @inbounds for j in 1:n, i in 1:m
            destination[i, j] = muladd(alpha, acc[i + MR * (j - 1)], convert(T, destination[i, j]))
        end
    else
        @inbounds for j in 1:n, i in 1:m
            destination[i, j] = muladd(alpha, acc[i + MR * (j - 1)], beta * convert(T, destination[i, j]))
        end
    end
    return destination
end
