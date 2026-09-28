"""
    ScalarKernel(::Val{MR}, ::Val{NR}, ::Type{T})

Scalar reference microkernel for register tile `(MR, NR)` and element type `T`.
"""
struct ScalarKernel{MR, NR, T} <: DescriptorKernel{MR, NR, T}
    descriptor::KernelDescriptor{MR, NR, T}
end

ScalarKernel(::Val{MR}, ::Val{NR}, ::Type{T}) where {MR, NR, T} =
    ScalarKernel(KernelDescriptor(Val(MR), Val(NR), T))

# `acc[i+1, j+1]` for zero-based `(i, j)`.
zero_accumulator(kernel::ScalarKernel{MR, NR, T}) where {MR, NR, T} = zeros(T, MR, NR)

# Extends `Base.accumulate` so `using QuasiStrided` does not clash with it.
function Base.accumulate(
        kernel::ScalarKernel{MR, NR, T}, acc::AbstractMatrix{T},
        packed_a::PA, packed_b::PB, kc::Int
    ) where {MR, NR, T, PA, PB}
    kc == 0 && return acc
    kc > 0 || throw(ArgumentError("accumulate requires kc >= 0, got kc = $kc"))
    @inbounds for p in 0:(kc - 1)
        for j in 0:(NR - 1)
            bj = panel_load(packed_b, packed_b_offset(kernel, j, p))
            for i in 0:(MR - 1)
                ai = panel_load(packed_a, packed_a_offset(kernel, i, p))
                acc[i + 1, j + 1] = muladd(ai, bj, acc[i + 1, j + 1])
            end
        end
    end
    return acc
end

function store_tile!(
        destination::QSTile, acc::AbstractMatrix{T},
        alpha::T, beta::T, kernel::ScalarKernel
    ) where {T}
    m, n = _store_prologue!(destination, alpha, beta)
    (m == 0 || n == 0) && return destination
    @inbounds for j in 0:(n - 1), i in 0:(m - 1)
        _axpby_tile!(destination, i, j, alpha, acc[i + 1, j + 1], beta)
    end
    return destination
end
