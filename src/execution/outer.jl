# The outer-product path for `K == 1`, real `T`, M unit-stride in A and C:
# `C[:, n] = alpha * A * B[n] + beta * C[:, n]` as a streaming write. Through
# the nest a K = 1 contraction is all per-tile fixed cost. The arithmetic is
# the nest's term for term (`r = a * b`, then `vector_store!`'s
# expressions), so the two agree bitwise except on a signed zero. Real only:
# complex K = 1 tiles are not the bottleneck, and the path would need its own
# interleaved arithmetic.

# B is read one scalar per column.
function execute_path!(plan::ContractPlan{T}, alphaT::T, betaT::T, ::OuterPath{W}) where {T, W}
    ws = plan.workspace
    Astorage = plan.Astorage
    Bstorage = plan.Bstorage
    Cstorage = plan.Cstorage
    Bbase = plan.Bbase
    bufB = ws.n.offsets[1]
    bufC = ws.n.offsets[2]
    m_length = axis_length(plan.mgroup)
    n_length = axis_length(plan.ngroup)
    n_block = plan.blocking.n_block
    lenB = length(Bstorage)
    lenC = length(Cstorage)

    checked_span_bounds(plan.Abase, (0, m_length - 1), (0, 0), length(Astorage))

    GC.@preserve ws Astorage Cstorage begin
        ap = pointer(Astorage) + sizeof(T) * plan.Abase
        cp = pointer(Cstorage) + sizeof(T) * plan.Cbase
        n_block_start = 0
        while n_block_start < n_length
            n_block_length = min(n_block, n_length - n_block_start)
            fill_offsets!((bufB, bufC), plan.ngroup, n_block_start, n_block_length)
            checked_span_bounds(Bbase, extrema(view(bufB, 1:n_block_length)), (0, 0), lenB)
            checked_span_bounds(plan.Cbase, (0, m_length - 1), extrema(view(bufC, 1:n_block_length)), lenC)
            outer_block!(
                cp, bufC, ap, Bstorage, Bbase, bufB, n_block_length, m_length,
                alphaT, static_beta(betaT), plan.btransform, Val(W)
            )
            n_block_start += n_block_length
        end
    end
    return nothing
end

# One N block, W rows at a time with a scalar tail, for a `beta` case.
@inline function outer_block!(
        cp::Ptr{T}, bufC::Vector{Int}, ap::Ptr{T}, Bstorage::SB, Bbase::Int,
        bufB::Vector{Int}, n_block_length::Int, m_length::Int, alpha::T, beta, btransform::F, ::Val{W}
    ) where {T, SB, F, W}
    sz = sizeof(T)
    mmain = (m_length ÷ W) * W
    @inbounds for n in 1:n_block_length
        b = convert(T, btransform(Bstorage[Bbase + bufB[n] + 1]))
        bv = Vec{W, T}(b)
        cpn = cp + sz * bufC[n]
        m = 0
        while m < mmain
            r = vload(Vec{W, T}, ap + sz * m) * bv
            vstore(axpby(alpha, r, old_c(Vec{W, T}, cpn + sz * m, beta), beta), cpn + sz * m)
            m += W
        end
        while m < m_length
            r = unsafe_load(ap + sz * m) * b
            unsafe_store!(cpn + sz * m, axpby(alpha, r, old_c(T, cpn + sz * m, beta), beta))
            m += 1
        end
    end
    return nothing
end
