# The outer-product path for `K == 1`, real `T`, M unit-stride in A and C:
# `C[:, n] = alpha * A * B[n] + beta * C[:, n]` as a streaming write. Through
# the nest a K = 1 contraction is all per-tile fixed cost. The arithmetic is
# the nest's term for term (`r = a * b`, then `vector_store!`'s
# expressions), so the two agree bitwise except on a signed zero. Real only:
# complex K = 1 tiles are not the bottleneck, and the path would need its own
# interleaved arithmetic.

# N and B may have any layout: B is read one scalar per column.
function _outer_applicable(::Type{T}, Astorage, Cstorage, mgroup::AxisGroup, m_length::Int) where {T}
    T <: Real || return false
    (Astorage isa DenseVector{T} && Cstorage isa DenseVector{T}) || return false
    m_length >= _dot_lanewidth(T) || return false
    map_ramp_step(mgroup, 1) == 1 || return false
    return map_ramp_step(mgroup, 2) == 1
end

function _execute_outer!(
        plan::ContractPlan{T}, alphaT::T, betaT::T, m_length::Int, n_length::Int, ::Val{W}
    ) where {T, W}
    ws = plan.workspace
    Astorage = plan.Astorage
    Bstorage = plan.Bstorage
    Cstorage = plan.Cstorage
    Bbase = plan.Bbase
    bufB = ws.n.offsets[1]
    bufC = ws.n.offsets[2]
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
            checked_span_bounds(Bbase, _buffer_range(bufB, n_block_length), (0, 0), lenB)
            checked_span_bounds(plan.Cbase, (0, m_length - 1), _buffer_range(bufC, n_block_length), lenC)
            _outer_block!(
                cp, bufC, ap, Bstorage, Bbase, bufB, n_block_length, m_length,
                alphaT, betaT, plan.btransform, Val(W)
            )
            n_block_start += n_block_length
        end
    end
    return plan.Cstorage
end

# One N block, W rows at a time with a scalar tail; the `beta` regime is
# chosen once per block.
@inline function _outer_block!(
        cp::Ptr{T}, bufC::Vector{Int}, ap::Ptr{T}, Bstorage::SB, Bbase::Int,
        bufB::Vector{Int}, n_block_length::Int, m_length::Int, alpha::T, beta::T, btransform::F, ::Val{W}
    ) where {T, SB, F, W}
    sz = sizeof(T)
    mmain = (m_length ÷ W) * W
    @inbounds for n in 1:n_block_length
        b = convert(T, btransform(Bstorage[Bbase + bufB[n] + 1]))
        bv = Vec{W, T}(b)
        cpn = cp + sz * bufC[n]
        m = 0
        if iszero(beta)
            while m < mmain
                r = vload(Vec{W, T}, ap + sz * m) * bv
                vstore(alpha * r, cpn + sz * m)
                m += W
            end
            while m < m_length
                r = unsafe_load(ap + sz * m) * b
                unsafe_store!(cpn + sz * m, alpha * r)
                m += 1
            end
        elseif isone(beta)
            while m < mmain
                r = vload(Vec{W, T}, ap + sz * m) * bv
                vstore(muladd(alpha, r, vload(Vec{W, T}, cpn + sz * m)), cpn + sz * m)
                m += W
            end
            while m < m_length
                r = unsafe_load(ap + sz * m) * b
                unsafe_store!(cpn + sz * m, muladd(alpha, r, unsafe_load(cpn + sz * m)))
                m += 1
            end
        else
            betav = Vec{W, T}(beta)
            while m < mmain
                r = vload(Vec{W, T}, ap + sz * m) * bv
                vstore(muladd(alpha, r, betav * vload(Vec{W, T}, cpn + sz * m)), cpn + sz * m)
                m += W
            end
            while m < m_length
                r = unsafe_load(ap + sz * m) * b
                unsafe_store!(cpn + sz * m, muladd(alpha, r, beta * unsafe_load(cpn + sz * m)))
                m += 1
            end
        end
    end
    return nothing
end
