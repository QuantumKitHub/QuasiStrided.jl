# The outer-product path for `K == 1`, real `T`, M unit-stride in A and C:
# `C[:, n] = alpha * A * B[n] + beta * C[:, n]` as a streaming write. Through
# the nest a K = 1 contraction is all per-tile fixed cost. The arithmetic is
# the nest's term for term (`r = a * b`, then `_store_tile_vector!`'s
# expressions), so the two agree bitwise except on a signed zero. Real only:
# complex K = 1 tiles are not the bottleneck, and the path would need its own
# interleaved arithmetic.

# `:never` forces the path off, for benchmarks and tests.
const _OUTER_MODE = Ref{Symbol}(:auto)

# N and B may have any layout: B is read one scalar per column.
function _outer_applicable(::Type{T}, Astorage, Cstorage, mgroup::AxisGroup, Qm::Int) where {T}
    _OUTER_MODE[] === :never && return false
    T <: Real || return false
    (Astorage isa DenseVector{T} && Cstorage isa DenseVector{T}) || return false
    Qm >= _dot_lanewidth(T) || return false
    _map_ramp_step(mgroup, 1) == 1 || return false
    return _map_ramp_step(mgroup, 2) == 1
end

function _execute_outer!(
        plan::ContractPlan{T}, alphaT::T, betaT::T, Qm::Int, Qn::Int, ::Val{W}
    ) where {T, W}
    ws = plan.workspace
    Astorage = plan.Astorage
    Bstorage = plan.Bstorage
    Cstorage = plan.Cstorage
    Bbase = plan.Bbase
    bufB = ws.n_buf_B
    bufC = ws.n_buf_C
    nc_eff = plan.blocking.nc
    lenB = length(Bstorage)
    lenC = length(Cstorage)

    checked_span_bounds(plan.Abase, (0, Qm - 1), (0, 0), length(Astorage))

    GC.@preserve ws Astorage Cstorage begin
        ap = pointer(Astorage) + sizeof(T) * plan.Abase
        cp = pointer(Cstorage) + sizeof(T) * plan.Cbase
        jc = 0
        while jc < Qn
            nblock = min(nc_eff, Qn - jc)
            fill_offsets!((bufB, bufC), plan.ngroup, jc, nblock)
            checked_span_bounds(Bbase, _buffer_range(bufB, nblock), (0, 0), lenB)
            checked_span_bounds(plan.Cbase, (0, Qm - 1), _buffer_range(bufC, nblock), lenC)
            _outer_block!(
                cp, bufC, ap, Bstorage, Bbase, bufB, nblock, Qm,
                alphaT, betaT, plan.btransform, Val(W)
            )
            jc += nblock
        end
    end
    return plan.Cstorage
end

# One N block, W rows at a time with a scalar tail; the `beta` regime is
# chosen once per block.
@inline function _outer_block!(
        cp::Ptr{T}, bufC::Vector{Int}, ap::Ptr{T}, Bstorage::SB, Bbase::Int,
        bufB::Vector{Int}, nblock::Int, Qm::Int, alpha::T, beta::T, btransform::F, ::Val{W}
    ) where {T, SB, F, W}
    sz = sizeof(T)
    mmain = (Qm ÷ W) * W
    @inbounds for n in 1:nblock
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
            while m < Qm
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
            while m < Qm
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
            while m < Qm
                r = unsafe_load(ap + sz * m) * b
                unsafe_store!(cpn + sz * m, muladd(alpha, r, beta * unsafe_load(cpn + sz * m)))
                m += 1
            end
        end
    end
    return nothing
end
