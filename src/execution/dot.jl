# The dot-product ("gemv") path for `M == 1` or `N == 1` when the matrix
# operand (`Qfree x K`) is stored with K unit-stride. The register microkernel
# vectorizes along M, which is 1/m_tile full here, and gathering the K-major
# matrix into packed panels would dominate the call. This path vectorizes
# along K instead: one W-wide FMA per W matrix elements, the matrix read once
# and unpacked, one horizontal reduction per output. Only the K-vector is
# gathered, once per K block, with its conjugation folded in.

# Forces the path off (`:never`) for benchmarks and tests; `:auto` otherwise.
const _DOT_MODE = Ref{Symbol}(:auto)

# One full hardware vector register of `real(T)`.
@inline function _dot_lanewidth(::Type{T}) where {T}
    R = real(T)
    vb = target_profile().vector_bytes
    return (vb > 0 && vb % sizeof(R) == 0) ? vb ÷ sizeof(R) : _default_lanewidth(R)
end

# Outputs computed together: 8 independent FMA chains cover the FMA latency
# (a complex output has two accumulator planes).
_dot_group_width(::Type{T}) where {T} = T <: Complex ? 4 : 8

# Whether the dot path applies (all but the workspace capacity): a degenerate
# free extent, a matrix operand with unit-ramp K in dense storage (raw-pointer
# loads), and at least one vector of K.
function _dot_applicable(::Type{T}, Astorage, Bstorage, kgroup::AxisGroup, m_length::Int, n_length::Int, k_length::Int) where {T}
    _DOT_MODE[] === :never && return false
    (m_length == 1 || n_length == 1) || return false
    k_length >= _dot_lanewidth(T) || return false
    if m_length == 1
        Bstorage isa DenseVector{T} || return false
        map_ramp_step(kgroup, 2) == 1 || return false
    else
        Astorage isa DenseVector{T} || return false
        map_ramp_step(kgroup, 1) == 1 || return false
    end
    return true
end

# The gathered vector (and, complex, its pair-swapped copy) lives in the
# packed-A buffer. Always true for an automatically chosen kernel; `nothing`
# is the prediction's "assume so".
function _dot_capacity_ok(plan::ContractPlan{T}) where {T}
    need = (T <: Complex ? 2 : 1) * plan.blocking.k_block
    return (length(plan.workspace.packed_a) * sizeof(real(T))) ÷ sizeof(T) >= need
end
_dot_capacity_ok(::Nothing) = true

function _execute_dot!(plan::ContractPlan{T}, alphaT::T, betaT::T, matB::Bool, ::Val{W}) where {T, W}
    ws = plan.workspace
    if matB
        _dot_nest!(
            plan, ws, plan.Bstorage, plan.Bbase, plan.btransform,
            plan.Astorage, plan.Abase, plan.atransform,
            plan.ngroup, ws.n_buf_B, ws.n_buf_C, plan.blocking.n_block, ws.k_buf_A,
            alphaT, betaT, Val(W)
        )
    else
        _dot_nest!(
            plan, ws, plan.Astorage, plan.Abase, plan.atransform,
            plan.Bstorage, plan.Bbase, plan.btransform,
            plan.mgroup, ws.m_buf_A, ws.m_buf_C, plan.blocking.m_block, ws.k_buf_B,
            alphaT, betaT, Val(W)
        )
    end
    return plan.Cstorage
end

# Conjugation is folded out of the kernel: `sum conj(m)*v == conj(sum m*conj(v))`,
# so a conjugated matrix becomes a flipped vector plus a `conj` per output.
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
    k_length = axis_length(plan.kgroup)
    k_block = plan.blocking.k_block
    Cstorage = plan.Cstorage
    cbase = plan.Cbase
    lenm = length(mstorage)
    lenv = length(vstorage)
    lenc = length(Cstorage)
    mconj = _is_conj(mtrans)
    vflip = _is_conj(vtrans) ⊻ mconj
    NB = _dot_group_width(T)

    GC.@preserve ws mstorage begin
        gptr = reinterpret(Ptr{T}, pointer(ws.packed_a))
        mptr = pointer(mstorage)
        k_block_start = 0
        firstblock = true
        while k_block_start < k_length
            k_block_length = min(k_block, k_length - k_block_start)
            beta_eff = firstblock ? betaT : one(T)

            fill_offsets!((ws.k_buf_A, ws.k_buf_B), plan.kgroup, k_block_start, k_block_length)
            checked_span_bounds(vbase, _buffer_range(vkbuf, k_block_length), (0, 0), lenv)
            if vflip
                _dot_gather!(gptr, vstorage, vbase, vkbuf, k_block_length, conj)
            else
                _dot_gather!(gptr, vstorage, vbase, vkbuf, k_block_length, identity)
            end

            q0 = 0
            while q0 < Qf
                qcount = min(blocklen, Qf - q0)
                fill_offsets!((bufM, bufC), g, q0, qcount)
                # The matrix's K offsets are
                # `k_block_start .. k_block_start+k_block_length-1` (unit ramp).
                checked_span_bounds(mbase, _buffer_range(bufM, qcount), (k_block_start, k_block_start + k_block_length - 1), lenm)
                checked_span_bounds(cbase, _buffer_range(bufC, qcount), (0, 0), lenc)
                if mconj
                    _dot_block!(
                        Cstorage, cbase, bufC, mptr, mbase + k_block_start, bufM, qcount, gptr, k_block_length,
                        alphaT, beta_eff, Val(W), Val(NB), Val(true)
                    )
                else
                    _dot_block!(
                        Cstorage, cbase, bufC, mptr, mbase + k_block_start, bufM, qcount, gptr, k_block_length,
                        alphaT, beta_eff, Val(W), Val(NB), Val(false)
                    )
                end
                q0 += qcount
            end

            firstblock = false
            k_block_start += k_block_length
        end
    end
    return nothing
end

# Gather the transformed vector into `gptr`; for a complex `T` also its
# pair-swapped copy at `gptr + k_block_length`, which gives the imaginary part
# without a per-step shuffle.
@inline function _dot_gather!(
        gptr::Ptr{T}, vstorage::SV, vbase::Int, vkbuf::Vector{Int}, k_block_length::Int, transform::F
    ) where {T, SV, F}
    @inbounds for t in 0:(k_block_length - 1)
        z = convert(T, transform(vstorage[vbase + vkbuf[t + 1] + 1]))
        unsafe_store!(gptr + sizeof(T) * t, z)
        if T <: Complex
            unsafe_store!(gptr + sizeof(T) * (k_block_length + t), Complex(imag(z), real(z)))
        end
    end
    return nothing
end

# One block of outputs, `NB` at a time. The last group's padding outputs alias
# the last valid one (computed, never stored).
@inline function _dot_block!(
        Cstorage::SC, cbase::Int, bufC::Vector{Int}, mptr::Ptr{T}, mbase_k::Int,
        bufM::Vector{Int}, qcount::Int, gptr::Ptr{T}, k_block_length::Int,
        alpha::T, beta::T, ::Val{W}, ::Val{NB}, ::Val{MCONJ}
    ) where {SC, T, W, NB, MCONJ}
    q = 0
    while q < qcount
        nvalid = min(NB, qcount - q)
        mbases = _dot_bases(bufM, q, nvalid, mbase_k, Val(NB))
        sums = _dot_group(mptr, gptr, mbases, k_block_length, Val(W), Val(NB))
        @inbounds for j in 1:nvalid
            s = MCONJ ? conj(sums[j]) : sums[j]
            _axpby_at!(Cstorage, cbase + bufC[q + j] + 1, alpha, s, beta)
        end
        q += nvalid
    end
    return nothing
end

# A literal tuple: an `ntuple` closure over the loop-carried `q` would box it.
@generated function _dot_bases(bufM::Vector{Int}, q::Int, nvalid::Int, mbase_k::Int, ::Val{NB}) where {NB}
    ex = [:(mbase_k + @inbounds(bufM[q + min($j, nvalid)])) for j in 1:NB]
    return quote
        Base.@_inline_meta
        return $(Expr(:tuple, ex...))
    end
end

# `NB` dot products of the gathered vector with unit-stride matrix rows: a
# W-wide main loop, one masked tail step, a horizontal reduction. The step
# bodies are `@generated` so every tuple index is a literal.

@inline function _dot_group(
        mptr::Ptr{T}, gptr::Ptr{T}, mbases::NTuple{NB, Int}, k_block_length::Int, ::Val{W}, ::Val{NB}
    ) where {T <: Real, NB, W}
    accs = ntuple(_ -> zero(Vec{W, T}), Val(NB))
    kmain = (k_block_length ÷ W) * W
    t = 0
    while t < kmain
        accs = _dot_step_real(accs, mptr, gptr, mbases, t, nothing)
        t += W
    end
    # Masked-off lanes are never read, so nothing past a row's end is touched.
    if t < k_block_length
        accs = _dot_step_real(accs, mptr, gptr, mbases, t, _dot_tailmask(Val(W), k_block_length - t))
    end
    return _dot_reduce_real(accs)
end

# Lanes `0 .. rem-1` of a `W`-lane mask, `0 < rem < W`.
@inline _dot_tailmask(::Val{W}, rem::Int) where {W} =
    Vec{W, Int}(ntuple(i -> i - 1, Val(W))) < rem

# `mask === nothing` for the main loop. Passed explicitly: a default argument
# adds a non-inlined wrapper method.
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

# Complex, interleaved layout (`W` real lanes). Per output two planes,
#     P1 += m .* v        ->  re = sum(P1 .* (+1,-1,+1,-1,...))
#     P2 += m .* swap(v)  ->  im = sum(P2)
# so two FMAs per matrix vector and no shuffle. `accs[NB+j]` is `P2_j`.

@inline function _dot_group(
        mptr::Ptr{T}, gptr::Ptr{T}, mbases::NTuple{NB, Int}, k_block_length::Int, ::Val{W}, ::Val{NB}
    ) where {T <: Complex, NB, W}
    R = real(T)
    WC = W ÷ 2
    accs = ntuple(_ -> zero(Vec{W, R}), Val(2 * NB))
    kmain = (k_block_length ÷ WC) * WC
    t = 0
    while t < kmain
        accs = _dot_step_cplx(accs, mptr, gptr, gptr + sizeof(T) * k_block_length, mbases, t, nothing)
        t += WC
    end
    if t < k_block_length
        accs = _dot_step_cplx(
            accs, mptr, gptr, gptr + sizeof(T) * k_block_length, mbases, t,
            _dot_tailmask(Val(W), 2 * (k_block_length - t))
        )
    end
    return _dot_reduce_cplx(accs, T)
end

@generated function _dot_step_cplx(
        accs::NTuple{NA, Vec{W, R}}, mptr::Ptr{T}, gptr::Ptr{T}, gsptr::Ptr{T},
        mbases::NTuple{NB, Int}, t::Int, mask::M
    ) where {NA, W, R, T, NB, M}
    sz = sizeof(T)
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
