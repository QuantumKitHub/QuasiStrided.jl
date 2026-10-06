# The dot-product ("gemv") path for `M == 1` or `N == 1` when the matrix
# operand (`Qfree x K`) is stored with K unit-stride. The register microkernel
# vectorizes along M, which is 1/m_tile full here, and gathering the K-major
# matrix into packed panels would dominate the call. This path vectorizes
# along K instead: one W-wide FMA per W matrix elements, the matrix read once
# and unpacked, one horizontal reduction per output. Only the K-vector is
# gathered, once per K block, with its conjugation folded in.

# Outputs computed together: 8 independent FMA chains cover the FMA latency
# (a complex output has two accumulator planes).
dot_group_width(::Type{T}) where {T} = T <: Complex ? 4 : 8

function execute_path!(plan::ContractPlan{T}, alphaT::T, betaT::T, ::DotPath{MATB, W}) where {T, MATB, W}
    ws = plan.workspace
    matrix, vector = dot_operands(plan, Val(MATB))
    mstorage, mbase, mtrans, g, (bufM, bufC), blocklen = matrix
    vstorage, vbase, vtrans, vkbuf = vector
    Qf = axis_length(g)
    k_length = axis_length(plan.kgroup)
    k_block = plan.blocking.k_block
    Cstorage = plan.Cstorage
    cbase = plan.Cbase
    lenm = length(mstorage)
    lenv = length(vstorage)
    lenc = length(Cstorage)
    # Conjugation is folded out of the kernel: `sum conj(m)*v == conj(sum m*conj(v))`,
    # so a conjugated matrix becomes a flipped vector plus a `conj` per output.
    mconj = op_conjugates(mtrans)
    vflip = op_conjugates(vtrans) ⊻ mconj
    NB = dot_group_width(T)

    GC.@preserve ws mstorage begin
        gptr = pointer(ws.dot_vector)
        mptr = pointer(mstorage)
        k_block_start = 0
        first_k_block = true
        while k_block_start < k_length
            k_block_length = min(k_block, k_length - k_block_start)
            beta_eff = first_k_block ? betaT : one(T)

            fill_offsets!(ws.k, plan.kgroup, k_block_start, k_block_length)
            checked_span_bounds(vbase, extrema(view(vkbuf, 1:k_block_length)), (0, 0), lenv)
            if vflip
                dot_gather!(gptr, vstorage, vbase, vkbuf, k_block_length, conj)
            else
                dot_gather!(gptr, vstorage, vbase, vkbuf, k_block_length, identity)
            end

            q0 = 0
            while q0 < Qf
                qcount = min(blocklen, Qf - q0)
                fill_offsets!((bufM, bufC), g, q0, qcount)
                # The matrix's K offsets are
                # `k_block_start .. k_block_start+k_block_length-1` (unit ramp).
                checked_span_bounds(mbase, extrema(view(bufM, 1:qcount)), (k_block_start, k_block_start + k_block_length - 1), lenm)
                checked_span_bounds(cbase, extrema(view(bufC, 1:qcount)), (0, 0), lenc)
                if mconj
                    dot_block!(
                        Cstorage, cbase, bufC, mptr, mbase + k_block_start, bufM, qcount, gptr, k_block_length,
                        alphaT, beta_eff, Val(W), Val(NB), Val(true)
                    )
                else
                    dot_block!(
                        Cstorage, cbase, bufC, mptr, mbase + k_block_start, bufM, qcount, gptr, k_block_length,
                        alphaT, beta_eff, Val(W), Val(NB), Val(false)
                    )
                end
                q0 += qcount
            end

            first_k_block = false
            k_block_start += k_block_length
        end
    end
    return nothing
end

# The matrix side (B for `MATB`): storage, base, transform, free group, the
# group's offset buffers and block length. The vector side: storage, base,
# transform and K offsets.
@inline dot_operands(plan::ContractPlan, ::Val{true}) = (
    (plan.Bstorage, plan.Bbase, plan.btransform, plan.ngroup, plan.workspace.n.offsets, plan.blocking.n_block),
    (plan.Astorage, plan.Abase, plan.atransform, plan.workspace.k[1]),
)
@inline dot_operands(plan::ContractPlan, ::Val{false}) = (
    (plan.Astorage, plan.Abase, plan.atransform, plan.mgroup, plan.workspace.m.offsets, plan.blocking.m_block),
    (plan.Bstorage, plan.Bbase, plan.btransform, plan.workspace.k[2]),
)

# Gather the transformed vector into `gptr`; for a complex `T` also its
# pair-swapped copy at `gptr + k_block_length`, which gives the imaginary part
# without a per-step shuffle.
@inline function dot_gather!(
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
@inline function dot_block!(
        Cstorage::SC, cbase::Int, bufC::Vector{Int}, mptr::Ptr{T}, mbase_k::Int,
        bufM::Vector{Int}, qcount::Int, gptr::Ptr{T}, k_block_length::Int,
        alpha::T, beta::T, ::Val{W}, ::Val{NB}, ::Val{MCONJ}
    ) where {SC, T, W, NB, MCONJ}
    q = 0
    while q < qcount
        nvalid = min(NB, qcount - q)
        mbases = dot_bases(bufM, q, nvalid, mbase_k, Val(NB))
        sums = dot_group(mptr, gptr, mbases, k_block_length, Val(W), Val(NB))
        @inbounds for j in 1:nvalid
            s = MCONJ ? conj(sums[j]) : sums[j]
            axpby_at!(Cstorage, cbase + bufC[q + j] + 1, alpha, s, static_beta(beta))
        end
        q += nvalid
    end
    return nothing
end

# A literal tuple: an `ntuple` closure over the loop-carried `q` would box it.
@generated function dot_bases(bufM::Vector{Int}, q::Int, nvalid::Int, mbase_k::Int, ::Val{NB}) where {NB}
    ex = [:(mbase_k + @inbounds(bufM[q + min($j, nvalid)])) for j in 1:NB]
    return quote
        Base.@_inline_meta
        return $(Expr(:tuple, ex...))
    end
end

# `NB` dot products of the gathered vector with unit-stride matrix rows: a
# W-wide main loop, one masked tail step, a horizontal reduction. The step
# bodies are `@generated` so every tuple index is a literal.

@inline function dot_group(
        mptr::Ptr{T}, gptr::Ptr{T}, mbases::NTuple{NB, Int}, k_block_length::Int, ::Val{W}, ::Val{NB}
    ) where {T <: Real, NB, W}
    accs = ntuple(_ -> zero(Vec{W, T}), Val(NB))
    kmain = (k_block_length ÷ W) * W
    t = 0
    while t < kmain
        accs = dot_step_real(accs, mptr, gptr, mbases, t, nothing)
        t += W
    end
    # Masked-off lanes are never read, so nothing past a row's end is touched.
    if t < k_block_length
        accs = dot_step_real(accs, mptr, gptr, mbases, t, dot_tailmask(Val(W), k_block_length - t))
    end
    return dot_reduce_real(accs)
end

# Lanes `0 .. rem-1` of a `W`-lane mask, `0 < rem < W`.
@inline dot_tailmask(::Val{W}, rem::Int) where {W} =
    Vec{W, Int}(ntuple(i -> i - 1, Val(W))) < rem

# `mask === nothing` for the main loop. Passed explicitly: a default argument
# adds a non-inlined wrapper method.
@generated function dot_step_real(
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

@generated function dot_reduce_real(accs::NTuple{NB, Vec{W, T}}) where {NB, W, T}
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

@inline function dot_group(
        mptr::Ptr{T}, gptr::Ptr{T}, mbases::NTuple{NB, Int}, k_block_length::Int, ::Val{W}, ::Val{NB}
    ) where {T <: Complex, NB, W}
    R = real(T)
    WC = W ÷ 2
    accs = ntuple(_ -> zero(Vec{W, R}), Val(2 * NB))
    kmain = (k_block_length ÷ WC) * WC
    t = 0
    while t < kmain
        accs = dot_step_cplx(accs, mptr, gptr, gptr + sizeof(T) * k_block_length, mbases, t, nothing)
        t += WC
    end
    if t < k_block_length
        accs = dot_step_cplx(
            accs, mptr, gptr, gptr + sizeof(T) * k_block_length, mbases, t,
            dot_tailmask(Val(W), 2 * (k_block_length - t))
        )
    end
    return dot_reduce_cplx(accs, T)
end

@generated function dot_step_cplx(
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

@generated function dot_reduce_cplx(accs::NTuple{NA, Vec{W, R}}, ::Type{T}) where {NA, W, R, T}
    NB = NA ÷ 2
    sgn = ntuple(i -> isodd(i) ? one(R) : -one(R), W)
    ex = [:(Complex(sum(accs[$j] * SGN), sum(accs[$(NB + j)]))) for j in 1:NB]
    return quote
        Base.@_inline_meta
        SGN = Vec{$W, $R}($sgn)
        return $(Expr(:tuple, ex...))
    end
end
