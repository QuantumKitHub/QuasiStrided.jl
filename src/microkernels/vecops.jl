# SIMD building blocks shared by the K steps and the stores. Shuffle patterns are
# built from the vector width at specialization time, never hardcoded to one ISA.

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

# `p - q` in even lanes, `p + q` in odd lanes: Base's unfused complex `*` on
# interleaved data.
@generated function addsub(p::Vec{N, R}, q::Vec{N, R}) where {N, R}
    iseven(N) || return :(throw(ArgumentError("addsub: expected even N, got $N")))
    idx = ntuple(k -> iseven(k - 1) ? k - 1 : N + k - 1, N)
    return :(Base.@_inline_meta; shufflevector(p - q, p + q, Val($idx)))
end

# `v` holds `W` complex values `[re_0, im_0, ..., re_{W-1}, im_{W-1}]`.
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
