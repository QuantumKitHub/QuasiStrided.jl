# Planar (split-complex, BLIS "1r") microkernel, the default complex method.
# Both panels are `PlanarFormat` (`[re_0..re_{n-1} | im_0..im_{n-1}]` per K
# step), so the data is already in the right lanes: four real FMAs per
# (A-vector, B-scalar) pair, no shuffles. Everything works in `real(T)`.

using SIMD: Vec, vload, vstore, shufflevector

# GUARDRAIL: the real part is `muladd(-ai, bi, muladd(ar, br, c))`. `c - ai*bi`
# does NOT fuse (Julia sets no LLVM `contract` flag): two instructions, and
# different rounding. `-ai` is hoisted out of the `j` loop and folds into
# `vfnmadd`, so the negation is free.
@generated function accumulate_step(
        kernel::PlanarKernel{MR, NR, T, W}, acc::NTuple{NA, Vec{W, R}},
        packed_a::PA, packed_b::PB, p::Int
    ) where {MR, NR, T, W, R, NA, PA, PB}
    MV = MR ÷ W
    NV = MV * NR
    check_acc(:accumulate_step, R, T, NA, 2NV)

    arv = [Symbol(:ar, v) for v in 0:(MV - 1)]
    aiv = [Symbol(:ai, v) for v in 0:(MV - 1)]
    naiv = [Symbol(:nai, v) for v in 0:(MV - 1)]
    brv = [Symbol(:br, j) for j in 1:NR]
    biv = [Symbol(:bi, j) for j in 1:NR]

    load_a = Any[]
    for v in 0:(MV - 1)
        push!(
            load_a,
            :(
                $(arv[v + 1]) = panel_vload(
                    Vec{$W, $R}, packed_a, packed_a_offset(kernel, $(v * W + 1), p)
                )
            )
        )
        push!(
            load_a,
            :(
                $(aiv[v + 1]) = panel_vload(
                    Vec{$W, $R}, packed_a, packed_a_offset(kernel, $(v * W + 1), p, 1)
                )
            )
        )
        push!(load_a, :($(naiv[v + 1]) = -$(aiv[v + 1])))
    end

    load_b = Any[]
    for j in 1:NR
        push!(load_b, :(($(brv[j]), $(biv[j])) = b_complex(packed_b, kernel, $j, p)))
    end

    acc_exprs = Vector{Any}(undef, NA)
    for j in 1:NR, v in 0:(MV - 1)
        idx = acc_index(MV, v, j)
        acc_exprs[idx] = :(  # re: ar*br - ai*bi
            muladd(
                $(naiv[v + 1]), $(biv[j]),
                muladd($(arv[v + 1]), $(brv[j]), acc[$idx])
            )
        )
        acc_exprs[NV + idx] = :(  # im: ar*bi + ai*br
            muladd(
                $(aiv[v + 1]), $(brv[j]),
                muladd($(arv[v + 1]), $(biv[j]), acc[$(NV + idx)])
            )
        )
    end

    return quote
        Base.@_inline_meta
        @inbounds begin
            $(load_a...)
            $(load_b...)
            return $(Expr(:tuple, acc_exprs...))
        end
    end
end

# Shuffle patterns built from `W` at specialization time, never hardcoded to
# one ISA. `v` holds `W` complex values `[re_0, im_0, ..., re_{W-1}, im_{W-1}]`.
@generated function deinterleave_re(v::Vec{N, R}, ::Val{W}) where {N, R, W}
    N == 2 * W || return :(throw(ArgumentError("deinterleave_re: expected N == 2W")))
    idx = ntuple(k -> 2 * (k - 1), W)
    return :(shufflevector(v, Val($idx)))
end

@generated function deinterleave_im(v::Vec{N, R}, ::Val{W}) where {N, R, W}
    N == 2 * W || return :(throw(ArgumentError("deinterleave_im: expected N == 2W")))
    idx = ntuple(k -> 2 * (k - 1) + 1, W)
    return :(shufflevector(v, Val($idx)))
end

@generated function interleave_planes(re::Vec{W, R}, im::Vec{W, R}, ::Val{W}) where {W, R}
    idx = ntuple(k -> isodd(k) ? (k - 1) ÷ 2 : W + (k - 1) ÷ 2, 2 * W)
    return :(shufflevector(re, im, Val($idx)))
end

# One full `W`-row block; `at` is the zero-based index of its first real.
# The arithmetic transcribes Base's `Complex` expression trees (the ones
# `axpby_tile!` reaches), so full blocks match them bitwise: `*` is unfused,
# `muladd(z, w, x) = (muladd(zr, wr, -muladd(zi, wi, -xr)), muladd(zr, wi,
# muladd(zi, wr, xi)))`. Against the scalar fallback it is exact at
# `beta == 0/1` and ~1 ULP otherwise, because LLVM contracts Base's scalar
# complex `muladd` depending on inlining context.
@inline function split_store_block!(
        sp::Ptr{RC}, at::Int, rev::Vec{W, R}, imv::Vec{W, R},
        ar::Vec{W, R}, ai::Vec{W, R}, br::Vec{W, R}, bi::Vec{W, R},
        beta::Complex{R}, ::Val{W}
    ) where {RC, R, W}
    if iszero(beta)
        newre = ar * rev - ai * imv
        newim = ar * imv + ai * rev
    else
        old = convert(Vec{2 * W, R}, vload(Vec{2 * W, RC}, sp + sizeof(RC) * at))
        orv = deinterleave_re(old, Val(W))
        oiv = deinterleave_im(old, Val(W))
        if isone(beta)
            xr, xi = orv, oiv
        else
            xr = br * orv - bi * oiv
            xi = br * oiv + bi * orv
        end
        newre = muladd(ar, rev, -muladd(ai, imv, -xr))
        newim = muladd(ar, imv, muladd(ai, rev, xi))
    end
    vstore(convert(Vec{2 * W, RC}, interleave_planes(newre, newim, Val(W))), sp + sizeof(RC) * at)
    return nothing
end
