# The outer-product path for `K == 1`: `C[m, n] = alpha * A[m] * B[n] + beta *
# C[m, n]`, for a real `T` whose M composite is unit-stride in both A and C.
#
# Through the nest a K = 1 contraction is all fixed cost: every micro-tile
# runs one K step of FMAs and then a full store, and there are `cld(M, mr) *
# cld(N, nr)` of them -- 176 tiles for a 128 x 128 Float64 outer product,
# at ~26 ns each once the store is inlined (src/microkernels/simd.jl), i.e.
# 4.6 us for 131 KB of output that a plain streaming write covers in ~1 us.
# This path is that streaming write: for each n, one W-wide `A * B[n]` per W
# rows of C, no packing, no tiles, no accumulator. Its arithmetic is the
# nest's, term for term -- `r = a * b` (a K = 1 accumulation from zero) and
# then `_store_tile_vector!`'s `alpha * r` / `muladd(alpha, r, C)` /
# `muladd(alpha, r, beta * C)` -- so the two paths agree bitwise except on a
# signed zero (an FMA from `+0.0` returns `+0.0` where `a * b` returns
# `-0.0`).
#
# Real `T` only: the complex kernels' K = 1 tiles are not the bottleneck
# there (the per-call floor is), and a complex outer product would need its
# own interleaved arithmetic. Selected by `execute!` through `_select_path`
# when `_outer_applicable` holds; `_OUTER_MODE` overrides the choice for tests
# and benchmarks.

const _OUTER_MODE = Ref{Symbol}(:auto)

"""
    _outer_applicable(plan::ContractPlan, Qm) -> Bool

Whether `execute!` takes the outer-product path on a `Qk == 1` plan
(`_select_path`, src/execution/execute.jl, which checks `Qk` and then runs it
as `_OuterPath{W}` with `W = _dot_lanewidth(T)`). Eligible when
`_OUTER_MODE[]` is not `:never`, `T` is real, the M composite is
the unit ramp `offset(m) = m` in BOTH its A and C maps (`_map_ramp_step`),
`Qm >= W`, and A and C are
`DenseVector{T}` (raw-pointer vector loads/stores). N and B may have any
layout: B is read one scalar per column through the N composite's offsets.

Measured 2026-09-27, ccqlin038 (Cascade Lake, AVX-512, Julia 1.12.7), the
suite's `dim*_1_0_1_gemm_ready` cases `C[a,b] = A[a] B[b]`, Float64,
`execute!` time, this path vs the (store-inlined) nest on the same tree
(`benchmark/bench_degenerate.jl`, interleaved medians), and OpenBLAS on the
same data: 16x16 319 -> 227 ns (84), 63x63 2.40 -> 1.16 us (0.81), 128x128
6.15 -> 2.70 us (3.07). ComplexF64 is not taken: the nest is already within
~10% of OpenBLAS there from 63x63 up (4.6 vs 4.2 us, 15.1 vs 15.6 us).
"""
_outer_applicable(plan::ContractPlan{T}, Qm::Int) where {T} =
    _outer_applicable(T, plan.Astorage, plan.Cstorage, plan.mgroup, Qm)

# The rule on the plan's parts, so that `plan_contract` can predict the path
# before the plan exists (`_path_hint`).
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

    # A's M rows, once: `Abase + m`, `0 <= m < Qm` (the K offset is 0).
    checked_span_bounds(plan.Abase, (0, Qm - 1), (0, 0), length(Astorage))

    # `ws`: the offset buffers; A and C: raw-pointer loads/stores.
    GC.@preserve ws Astorage Cstorage begin
        ap = pointer(Astorage) + sizeof(T) * plan.Abase
        cp = pointer(Cstorage) + sizeof(T) * plan.Cbase
        jc = 0
        while jc < Qn
            nblock = min(nc_eff, Qn - jc)
            fill_offsets!((bufB, bufC), plan.ngroup, jc, nblock)
            # Every address this block touches, before any of it: B's
            # `nblock` scalars and the `Qm x nblock` rectangle of C.
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

# One N block: column `n` of C is `alpha * (A .* B[n]) + beta * C[:, n]`, W
# rows at a time with a scalar row tail. The `beta` regime is chosen once per
# block; inside, the expressions are `_store_tile_vector!`'s with `acc = a * b`.
@inline function _outer_block!(
        cp::Ptr{T}, bufC::Vector{Int}, ap::Ptr{T}, Bstorage::SB, Bbase::Int,
        bufB::Vector{Int}, nblock::Int, Qm::Int, alpha::T, beta::T, btransform::F, ::Val{W}
    ) where {T, SB, F, W}
    sz = sizeof(T)
    mmain = (Qm ÷ W) * W
    @inbounds for n in 1:nblock
        b = btransform(Bstorage[Bbase + bufB[n] + 1])::T
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
