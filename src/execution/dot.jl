# The dot-product ("gemv") path for a degenerate free extent: `M == 1` or
# `N == 1`, when the operand that still has a free extent -- the MATRIX
# operand, `Qfree x K` -- is stored with K unit-stride.
#
# Why the register microkernel is the wrong tool here: it vectorizes along M.
# With `M == 1` every A vector is 1/mr full, and with the roles swapped
# (`N == 1`) the matrix operand feeds M through a pack that gathers its K-major
# rows one element at a time -- on the suite's `C[cde] = A[ab] B[abcde]`
# cases that gather is 70% (Float64) of the call. A gemv wants to vectorize
# along K instead: `C[q] = sum_k v[k] * Mat[q, k]` with `Mat[q, :]` a
# contiguous run, i.e. one W-wide FMA per W elements of the matrix, which is
# read exactly once, plus one horizontal reduction per output. Nothing is
# packed except the K-vector `v` (gathered once per K block, with its
# conjugation folded in). This is what `?gemv` does for the transposed case.
#
# Selected by `execute!` through `_select_path` when `_dot_applicable` holds
# (see there for the eligibility rule and the measurement), before the
# five-loop nest; on an ineligible plan the nest runs. `_DOT_MODE` overrides the
# automatic choice for tests and benchmarks.

const _DOT_MODE = Ref{Symbol}(:auto)

# One full hardware vector register of `real(T)`: 8 Float64 on AVX-512, 4 on
# AVX2; the fallback lane width where the register width is unknown.
@inline function _dot_lanewidth(::Type{T}) where {T}
    R = real(T)
    vb = target_profile().vector_bytes
    return (vb > 0 && vb % sizeof(R) == 0) ? vb ÷ sizeof(R) : _default_lanewidth(R)
end

# Outputs computed together, each with its own accumulator(s): 8 for a real
# `T` (8 FMA chains in flight cover the FMA latency on every x86 we ship for),
# 4 for a complex one (two accumulator planes each, so 8 chains as well).
_dot_group_width(::Type{T}) where {T} = T <: Complex ? 4 : 8

# The affine step of map `p` of `g` alone, or `nothing` when that map is not
# a single ramp in the logical coordinate. `affine_ramp` (src/layout/axis_group.jl)
# asks the same of EVERY map at once; here the two operands' K maps are judged
# separately, because only the matrix operand's needs to be contiguous (the
# vector is gathered through its own offsets, whatever they are). Same
# singleton-skipping and Int128 comparison as `affine_ramp`.
function _map_ramp_step(g::AxisGroup{D, P}, p::Int) where {D, P}
    step = 0
    run = 1
    started = false
    for d in 1:D
        L = g.lengths[d]
        L == 0 && return 0
        L == 1 && continue
        S = g.strides[p][d]
        if !started
            step = S
            run = L
            started = true
        else
            Int128(run) * Int128(step) == Int128(S) || return nothing
            run *= L
        end
    end
    return step
end

"""
    _dot_applicable(plan::ContractPlan, Qm, Qn, Qk) -> Bool

Whether `execute!` takes the dot-product path on `plan` (`_select_path`,
src/execution/execute.jl, which then runs it as `_DotPath{Qm == 1, W}` with
`W = _dot_lanewidth(T)`). Eligible when `_DOT_MODE[]` is not `:never` and

  * `Qm == 1` (the matrix operand is B, its free composite the N group) or
    `Qn == 1` (the matrix operand is A, the M group);
  * the matrix operand's K map is the unit ramp `offset(k) = k` (checked per
    map by `_map_ramp_step`, so a permuted or strided K on the VECTOR operand
    does not disqualify), and its storage is a `DenseVector{T}` (the K runs
    are read with raw-pointer vector loads);
  * `Qk >= W` for the lane width `W = _dot_lanewidth(T)`: below one vector
    the path would be all scalar tail;
  * the plan's packed-A buffer can hold one K block of the gathered vector
    (always true for the shipped kernel shapes; checked, not assumed).

Measured 2026-09-27, ccqlin038 (Cascade Lake, AVX-512, Julia 1.12.7), the
suite's `dim{6,8,12,16}_0_2_3_gemm_ready` cases `C[cde] = A[ab] B[abcde]` (K
= dim^2, N = dim^3), `execute!` time, this path vs the nest on the same tree
(`benchmark/bench_degenerate.jl`, interleaved medians), and OpenBLAS `gemv`
on the same data:

                1x36x216   1x64x512   1x144x1728   1x256x4096
    Float64     4.9 -> 1.9  18.9 -> 4.2  176 -> 81   728 -> 331 us    (gemv 1.0 / 3.4 / 76 / 323)
    ComplexF64  12.1 -> 2.6 41.7 -> 7.6  304 -> 157  1359 -> 680 us   (gemv 2.4 / 8.6 / 152 / 679)

The nest's time there is the gather pack of the matrix operand, which this
path does not perform at all. The K tail is one masked step (not a scalar
loop): 1x36x216 Float64 went 3.0 -> 1.9 us with it.
"""
function _dot_applicable(plan::ContractPlan{T}, Qm::Int, Qn::Int, Qk::Int) where {T}
    return _dot_applicable(T, plan.Astorage, plan.Bstorage, plan.kgroup, Qm, Qn, Qk) &&
        _dot_capacity_ok(plan)
end

# Every clause but the workspace capacity, on the plan's parts, so that
# `plan_contract` can predict the path before the plan exists (`_path_hint`).
function _dot_applicable(::Type{T}, Astorage, Bstorage, kgroup::AxisGroup, Qm::Int, Qn::Int, Qk::Int) where {T}
    _DOT_MODE[] === :never && return false
    (Qm == 1 || Qn == 1) || return false
    Qk >= _dot_lanewidth(T) || return false
    if Qm == 1
        Bstorage isa DenseVector{T} || return false
        _map_ramp_step(kgroup, 2) == 1 || return false
    else
        Astorage isa DenseVector{T} || return false
        _map_ramp_step(kgroup, 1) == 1 || return false
    end
    return true
end

# The gathered vector (and, complex, its pair-swapped copy) lives in the
# packed-A buffer, reinterpreted as `T`. Always true for an automatically
# chosen kernel (the buffer holds `mr >= 2` K columns of reals per K step);
# `nothing` is the prediction's "assume so".
function _dot_capacity_ok(plan::ContractPlan{T}) where {T}
    need = (T <: Complex ? 2 : 1) * plan.blocking.kc
    return (length(plan.workspace.packed_a) * sizeof(real(T))) ÷ sizeof(T) >= need
end
_dot_capacity_ok(::Nothing) = true

function _execute_dot!(plan::ContractPlan{T}, alphaT::T, betaT::T, matB::Bool, ::Val{W}) where {T, W}
    ws = plan.workspace
    if matB
        _dot_nest!(
            plan, ws, plan.Bstorage, plan.Bbase, plan.btransform,
            plan.Astorage, plan.Abase, plan.atransform,
            plan.ngroup, ws.n_buf_B, ws.n_buf_C, plan.blocking.nc, ws.k_buf_A,
            alphaT, betaT, Val(W)
        )
    else
        _dot_nest!(
            plan, ws, plan.Astorage, plan.Abase, plan.atransform,
            plan.Bstorage, plan.Bbase, plan.btransform,
            plan.mgroup, ws.m_buf_A, ws.m_buf_C, plan.blocking.mc, ws.k_buf_B,
            alphaT, betaT, Val(W)
        )
    end
    return plan.Cstorage
end

# Conjugation is folded rather than applied per element in the kernel:
# `sum conj(m)*v == conj(sum m*conj(v))`, so a conjugated matrix operand
# becomes a flip of the gathered vector plus one `conj` of each output, and
# the vector's own transform composes with that flip by parity (XOR), the
# same rule `plan_contract` uses for a flag against a view's `op`.
@inline _is_conj(::typeof(conj)) = true
@inline _is_conj(::typeof(identity)) = false

# `(lo, hi)` of `buf[1:count]`, `count >= 1`.
@inline function _buffer_range(buf::Vector{Int}, count::Int)
    @inbounds lo = hi = buf[1]
    @inbounds for t in 2:count
        v = buf[t]
        lo = min(lo, v)
        hi = max(hi, v)
    end
    return (lo, hi)
end

function _dot_nest!(
        plan::ContractPlan{T}, ws, mstorage::SM, mbase::Int, mtrans::FM,
        vstorage::SV, vbase::Int, vtrans::FV, g::G, bufM::Vector{Int}, bufC::Vector{Int},
        blocklen::Int, vkbuf::Vector{Int}, alphaT::T, betaT::T, ::Val{W}
    ) where {T, SM, FM, SV, FV, G <: AxisGroup, W}
    Qf = axis_length(g)
    Qk = axis_length(plan.kgroup)
    kc_eff = plan.blocking.kc
    Cstorage = plan.Cstorage
    cbase = plan.Cbase
    lenm = length(mstorage)
    lenv = length(vstorage)
    lenc = length(Cstorage)
    mconj = _is_conj(mtrans)
    vflip = _is_conj(vtrans) ⊻ mconj
    NB = _dot_group_width(T)

    # `ws`: the gather buffer and every offset buffer. `mstorage`: the K runs
    # are raw-pointer vector loads.
    GC.@preserve ws mstorage begin
        gptr = reinterpret(Ptr{T}, pointer(ws.packed_a))
        mptr = pointer(mstorage)
        pc = 0
        firstblock = true
        while pc < Qk
            kblock = min(kc_eff, Qk - pc)
            beta_eff = firstblock ? betaT : one(T)

            # The vector's K offsets for this block (both K maps are filled;
            # only the vector's is read -- the matrix's is the unit ramp).
            fill_offsets!((ws.k_buf_A, ws.k_buf_B), plan.kgroup, pc, kblock)
            checked_span_bounds(vbase, _buffer_range(vkbuf, kblock), (0, 0), lenv)
            if vflip
                _dot_gather!(gptr, vstorage, vbase, vkbuf, kblock, conj)
            else
                _dot_gather!(gptr, vstorage, vbase, vkbuf, kblock, identity)
            end

            q0 = 0
            while q0 < Qf
                qcount = min(blocklen, Qf - q0)
                fill_offsets!((bufM, bufC), g, q0, qcount)
                # Every address this block reads/writes, validated before any
                # of it: the matrix rows against this K block (the matrix's
                # K offsets are `pc .. pc+kblock-1` by the unit-ramp
                # eligibility), and the C elements.
                checked_span_bounds(mbase, _buffer_range(bufM, qcount), (pc, pc + kblock - 1), lenm)
                checked_span_bounds(cbase, _buffer_range(bufC, qcount), (0, 0), lenc)
                if mconj
                    _dot_block!(
                        Cstorage, cbase, bufC, mptr, mbase + pc, bufM, qcount, gptr, kblock,
                        alphaT, beta_eff, Val(W), Val(NB), Val(true)
                    )
                else
                    _dot_block!(
                        Cstorage, cbase, bufC, mptr, mbase + pc, bufM, qcount, gptr, kblock,
                        alphaT, beta_eff, Val(W), Val(NB), Val(false)
                    )
                end
                q0 += qcount
            end

            firstblock = false
            pc += kblock
        end
    end
    return nothing
end

# Gather `v[t] = transform(vstorage[vbase + vkbuf[t+1] + 1])`, `0 <= t <
# kblock`, into `gptr`; for a complex `T` also its pair-swapped copy
# `Complex(imag, real)` at `gptr + kblock`, which the complex kernel
# multiplies by to get the imaginary part without a per-step shuffle. The
# caller has bounds-checked the vector region.
@inline function _dot_gather!(
        gptr::Ptr{T}, vstorage::SV, vbase::Int, vkbuf::Vector{Int}, kblock::Int, transform::F
    ) where {T, SV, F}
    @inbounds for t in 0:(kblock - 1)
        z = transform(vstorage[vbase + vkbuf[t + 1] + 1])::T
        unsafe_store!(gptr + sizeof(T) * t, z)
        if T <: Complex
            unsafe_store!(gptr + sizeof(T) * (kblock + t), Complex(imag(z), real(z)))
        end
    end
    return nothing
end

# One block of outputs, `NB` at a time. The last group's padding outputs alias
# the block's last valid output (computed, never stored) -- the same trick as
# `UnpackedBView`'s padding columns.
@inline function _dot_block!(
        Cstorage::SC, cbase::Int, bufC::Vector{Int}, mptr::Ptr{T}, mbase_k::Int,
        bufM::Vector{Int}, qcount::Int, gptr::Ptr{T}, kblock::Int,
        alpha::T, beta::T, ::Val{W}, ::Val{NB}, ::Val{MCONJ}
    ) where {SC, T, W, NB, MCONJ}
    q = 0
    while q < qcount
        nvalid = min(NB, qcount - q)
        mbases = _dot_bases(bufM, q, nvalid, mbase_k, Val(NB))
        sums = _dot_group(mptr, gptr, mbases, kblock, Val(W), Val(NB))
        @inbounds for j in 1:nvalid
            s = MCONJ ? conj(sums[j]) : sums[j]
            _axpby_at!(Cstorage, cbase + bufC[q + j] + 1, alpha, s, beta)
        end
        q += nvalid
    end
    return nothing
end

# `(mbase_k + bufM[q + min(j, nvalid)])` for `j in 1:NB`, as a literal tuple
# expression: an `ntuple` closure over the loop-carried `q` would box it.
@generated function _dot_bases(bufM::Vector{Int}, q::Int, nvalid::Int, mbase_k::Int, ::Val{NB}) where {NB}
    ex = [:(mbase_k + @inbounds(bufM[q + min($j, nvalid)])) for j in 1:NB]
    return quote
        Base.@_inline_meta
        return $(Expr(:tuple, ex...))
    end
end

# ----------------------------------------------------------------------------
# The group kernels: `NB` dot products of length `kblock` between the gathered
# vector at `gptr` and the matrix rows at `mptr + mbases[j]`, all unit-stride.
# A W-wide main loop, one masked step for the `kblock % W` tail, and a
# horizontal reduction; every tuple index is a literal (Cliff B, as in the
# microkernels), which is why the step bodies are `@generated`.
# ----------------------------------------------------------------------------

# --- real ------------------------------------------------------------------

@inline function _dot_group(
        mptr::Ptr{T}, gptr::Ptr{T}, mbases::NTuple{NB, Int}, kblock::Int, ::Val{W}, ::Val{NB}
    ) where {T <: Real, NB, W}
    accs = ntuple(_ -> zero(Vec{W, T}), Val(NB))
    kmain = (kblock ÷ W) * W
    t = 0
    while t < kmain
        accs = _dot_step_real(accs, mptr, gptr, mbases, t, nothing)
        t += W
    end
    # The K tail (`kblock % W` elements) as ONE masked step: masked-off lanes
    # load as zero and are never read from memory, so no address past a row's
    # end is touched.
    if t < kblock
        accs = _dot_step_real(accs, mptr, gptr, mbases, t, _dot_tailmask(Val(W), kblock - t))
    end
    return _dot_reduce_real(accs)
end

# Lanes `0 .. rem-1` of a `W`-lane mask, `0 < rem < W`.
@inline _dot_tailmask(::Val{W}, rem::Int) where {W} =
    Vec{W, Int}(ntuple(i -> i - 1, Val(W))) < rem

# One W-wide K step for every output of the group; with a `mask`, the masked
# tail step (`mask === nothing` for the main loop, resolved at compile time;
# passed explicitly -- a default argument adds a non-inlined wrapper method).
@generated function _dot_step_real(
        accs::NTuple{NB, Vec{W, T}}, mptr::Ptr{T}, gptr::Ptr{T}, mbases::NTuple{NB, Int}, t::Int,
        mask::M
    ) where {NB, W, T, M}
    sz = sizeof(T)
    mk = M === Nothing ? () : (:mask,)
    loads = [:($(Symbol(:m, j)) = vload(Vec{$W, $T}, mptr + $sz * (mbases[$j] + t), $(mk...))) for j in 1:NB]
    fmas = [:(muladd($(Symbol(:m, j)), vv, accs[$j])) for j in 1:NB]
    return quote
        Base.@_inline_meta
        vv = vload(Vec{$W, $T}, gptr + $sz * t, $(mk...))
        $(loads...)
        return $(Expr(:tuple, fmas...))
    end
end

@generated function _dot_reduce_real(accs::NTuple{NB, Vec{W, T}}) where {NB, W, T}
    ex = [:(sum(accs[$j])) for j in 1:NB]
    return quote
        Base.@_inline_meta
        return $(Expr(:tuple, ex...))
    end
end

# --- complex ---------------------------------------------------------------
#
# Interleaved (native) layout throughout, `W` REAL lanes = `W ÷ 2` complex
# elements per vector. For output `j`, two accumulator planes over the K loop:
#
#     P1 += m .* v          lanes (mr*vr, mi*vi, ...)  ->  re = sum(P1 .* (+1,-1,+1,-1,...))
#     P2 += m .* swap(v)    lanes (mr*vi, mi*vr, ...)  ->  im = sum(P2)
#
# with `swap(v)` the gathered pair-swapped copy at `gptr + kblock`. Two FMAs
# per matrix vector, no shuffle in the loop. `accs[j]` is `P1_j`, `accs[NB+j]`
# is `P2_j`.

@inline function _dot_group(
        mptr::Ptr{T}, gptr::Ptr{T}, mbases::NTuple{NB, Int}, kblock::Int, ::Val{W}, ::Val{NB}
    ) where {T <: Complex, NB, W}
    R = real(T)
    WC = W ÷ 2
    accs = ntuple(_ -> zero(Vec{W, R}), Val(2 * NB))
    kmain = (kblock ÷ WC) * WC
    t = 0
    while t < kmain
        accs = _dot_step_cplx(accs, mptr, gptr, gptr + sizeof(T) * kblock, mbases, t, nothing)
        t += WC
    end
    # The K tail as one masked step over its `2 * (kblock - t)` real lanes.
    if t < kblock
        accs = _dot_step_cplx(
            accs, mptr, gptr, gptr + sizeof(T) * kblock, mbases, t,
            _dot_tailmask(Val(W), 2 * (kblock - t))
        )
    end
    return _dot_reduce_cplx(accs, T)
end

@generated function _dot_step_cplx(
        accs::NTuple{NA, Vec{W, R}}, mptr::Ptr{T}, gptr::Ptr{T}, gsptr::Ptr{T},
        mbases::NTuple{NB, Int}, t::Int, mask::M
    ) where {NA, W, R, T, NB, M}
    NA == 2 * NB || return :(throw(ArgumentError("_dot_step_cplx: expected 2NB accumulators")))
    T === Complex{R} || return :(throw(ArgumentError("_dot_step_cplx: lane type must be real(T)")))
    sz = sizeof(T)   # bytes per complex element; offsets below are in elements
    mk = M === Nothing ? () : (:mask,)
    loads = [
        :($(Symbol(:m, j)) = vload(Vec{$W, $R}, reinterpret(Ptr{$R}, mptr + $sz * (mbases[$j] + t)), $(mk...)))
            for j in 1:NB
    ]
    p1 = [:(muladd($(Symbol(:m, j)), vv, accs[$j])) for j in 1:NB]
    p2 = [:(muladd($(Symbol(:m, j)), vs, accs[$(NB + j)])) for j in 1:NB]
    return quote
        Base.@_inline_meta
        vv = vload(Vec{$W, $R}, reinterpret(Ptr{$R}, gptr + $sz * t), $(mk...))
        vs = vload(Vec{$W, $R}, reinterpret(Ptr{$R}, gsptr + $sz * t), $(mk...))
        $(loads...)
        return $(Expr(:tuple, p1..., p2...))
    end
end

@generated function _dot_reduce_cplx(accs::NTuple{NA, Vec{W, R}}, ::Type{T}) where {NA, W, R, T}
    NB = NA ÷ 2
    sgn = ntuple(i -> isodd(i) ? one(R) : -one(R), W)
    ex = [:(Complex(sum(accs[$j] * SGN), sum(accs[$(NB + j)]))) for j in 1:NB]
    return quote
        Base.@_inline_meta
        SGN = Vec{$W, $R}($sgn)
        return $(Expr(:tuple, ex...))
    end
end
