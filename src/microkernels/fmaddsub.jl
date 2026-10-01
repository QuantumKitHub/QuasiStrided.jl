# FMAddSub (interleaved-accumulator) complex microkernel. One accumulator plane
# in 1m's layout (lanes `(2u-1, 2u)` are `(re, im)` of one complex row), updated
# per (A-vector, B-column) pair by two chained x86 `vfmaddsub` ops:
#
#     acc = fmaddsub(a, br, fmaddsub(swap(a), bi, acc))
#
# where `fmaddsub(x, y, c)` is `x*y - c` in even (0-based) lanes and `x*y + c`
# in odd lanes, `a = [ar0, ai0, ...]`, `swap(a) = [ai0, ar0, ...]`. The two
# sign flips compose back to an accumulation:
#
#     even:  ar*br - (ai*bi - c_re)  =  c_re + ar*br - ai*bi
#     odd:   ai*br + (ar*bi + c_im)  =  c_im + ai*br + ar*bi
#
# The lane-parity sign needs re/im of one element in adjacent lanes, hence
# `InterleavedFormat` A; B is read as broadcast scalars, so it stays planar.
# Same FMA count as planar and 1m, plus one pair-swap per A vector per K step.

using SIMD: Vec, shufflevector

# `x*y - c` in even lanes, `x*y + c` in odd lanes, each one fused rounding.
# Generic IR (`fneg` + two `llvm.fma` + a blend), which the X86 backend folds
# into one `vfmaddsub` wherever FMA3 exists (and elsewhere stays correct).
# GUARDRAIL: the SIMD.jl spelling `shufflevector(muladd(x, y, -c), muladd(x, y,
# c), ...)` also selects `vfmaddsub` but leaves a dead stack store of the
# accumulator in the K loop, every step.
@generated function fmaddsub(x::Vec{N, R}, y::Vec{N, R}, c::Vec{N, R}) where {N, R}
    R === Float64 || R === Float32 ||
        return :(throw(ArgumentError("fmaddsub: unsupported lane type $R")))
    ty = "<$N x $(R === Float64 ? "double" : "float")>"
    fn = "llvm.fma.v$(N)$(R === Float64 ? "f64" : "f32")"
    # lane k: even -> `s` (x*y - c); odd -> `d` (x*y + c), index N + k.
    mask = join(("i32 $(iseven(k) ? k : N + k)" for k in 0:(N - 1)), ", ")
    ir = """
    declare $ty @$fn($ty, $ty, $ty)
    define $ty @entry($ty %x, $ty %y, $ty %c) #0 {
    top:
      %nc = fneg $ty %c
      %s = call $ty @$fn($ty %x, $ty %y, $ty %nc)
      %d = call $ty @$fn($ty %x, $ty %y, $ty %c)
      %r = shufflevector $ty %s, $ty %d, <$N x i32> <$mask>
      ret $ty %r
    }
    attributes #0 = { alwaysinline }
    """
    VT = NTuple{N, VecElement{R}}
    return quote
        Base.@_inline_meta
        Vec{$N, $R}(
            Base.llvmcall(($ir, "entry"), $VT, Tuple{$VT, $VT, $VT}, x.data, y.data, c.data)
        )
    end
end

# `[x1, x0, x3, x2, ...]`: one in-lane `vshufpd`/`vpermilps`.
@generated function swap_pairs(x::Vec{N, R}) where {N, R}
    iseven(N) || return :(throw(ArgumentError("swap_pairs: expected even N, got $N")))
    idx = ntuple(k -> isodd(k) ? k : k - 2, N)
    return :(Base.@_inline_meta; shufflevector(x, Val($idx)))
end

# The INNER op must be the `swap(a) * bi` one: only the inner product is
# subtracted in the real lanes. Reversed, the real part is `ai*bi - ar*br`.
@generated function accumulate_step(
        kernel::FMAddSubKernel{MR, NR, T, W}, acc::NTuple{NA, Vec{W, R}},
        packed_a::PA, packed_b::PB, p::Int
    ) where {MR, NR, T, W, R, NA, PA, PB}
    MV = (2 * MR) ÷ W
    check_acc(:accumulate_step, R, T, NA, MV * NR)

    av = [Symbol(:a, v) for v in 0:(MV - 1)]
    sv = [Symbol(:s, v) for v in 0:(MV - 1)]
    brv = [Symbol(:br, j) for j in 1:NR]
    biv = [Symbol(:bi, j) for j in 1:NR]

    load_a = Any[]
    for v in 0:(MV - 1)
        push!(
            load_a,
            :(
                $(av[v + 1]) = panel_vload(
                    Vec{$W, $R}, packed_a, packed_a_offset(kernel, $(v * W + 1), p)
                )
            )
        )
        push!(load_a, :($(sv[v + 1]) = swap_pairs($(av[v + 1]))))
    end

    load_b = Any[]
    for j in 1:NR
        push!(load_b, :((br_s, bi_s) = b_complex(packed_b, kernel, $j, p)))
        push!(load_b, :($(brv[j]) = Vec{$W, $R}(br_s)))
        push!(load_b, :($(biv[j]) = Vec{$W, $R}(bi_s)))
    end

    acc_exprs = Vector{Any}(undef, NA)
    for j in 1:NR, v in 0:(MV - 1)
        idx = acc_index(MV, v, j)
        acc_exprs[idx] = :(
            fmaddsub(
                $(av[v + 1]), $(brv[j]),
                fmaddsub($(sv[v + 1]), $(biv[j]), acc[$idx])
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

# `p - q` in even lanes, `p + q` in odd lanes: Base's unfused complex `*` on
# interleaved data.
@generated function addsub(p::Vec{N, R}, q::Vec{N, R}) where {N, R}
    iseven(N) || return :(throw(ArgumentError("addsub: expected even N, got $N")))
    idx = ntuple(k -> iseven(k - 1) ? k - 1 : N + k - 1, N)
    return :(Base.@_inline_meta; shufflevector(p - q, p + q, Val($idx)))
end

# One full `W÷2`-row block, already in `Complex`'s memory order, so no
# interleave shuffle. Base's `Complex` expression trees in lanes, as planar's
# `split_store_block!`:
#     beta == 0:  addsub(ar*r, ai*swap(r))
#     beta == 1:  fmaddsub(ar, r, fmaddsub(ai, swap(r), C))
#     otherwise:  as beta == 1 with C := addsub(br*C, bi*swap(C))
@inline function lanepair_store_block!(
        sp::Ptr{RC}, at::Int, r::Vec{W, R},
        ar::Vec{W, R}, ai::Vec{W, R}, br::Vec{W, R}, bi::Vec{W, R},
        ::Val{B}
    ) where {RC, R, W, B}
    s = swap_pairs(r)
    if B === :zero
        new = addsub(ar * r, ai * s)
    else
        old = convert(Vec{W, R}, vload(Vec{W, RC}, sp + sizeof(RC) * at))
        x = B === :one ? old : addsub(br * old, bi * swap_pairs(old))
        new = fmaddsub(ar, r, fmaddsub(ai, s, x))
    end
    vstore(convert(Vec{W, RC}, new), sp + sizeof(RC) * at)
    return nothing
end
