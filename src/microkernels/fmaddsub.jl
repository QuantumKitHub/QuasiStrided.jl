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

"""
    FMAddSubKernel(::Val{MR}, ::Val{NR}, ::Type{T}[, ::Val{W}])

Interleaved-accumulator complex microkernel using x86 `vfmaddsub`, over
`SIMD.Vec{W,real(T)}` lanes with `InterleavedFormat` A and `PlanarFormat` B.
`2MR` must be a multiple of `W`, and `W` must be even. The default for small-M
complex contractions on AVX-512.
"""
struct FMAddSubKernel{MR, NR, T, W} <: DescriptorKernel{MR, NR, T}
    descriptor::ComplexKernelDescriptor{MR, NR, T, InterleavedFormat, PlanarFormat}

    function FMAddSubKernel{MR, NR, T, W}(
            descriptor::ComplexKernelDescriptor{MR, NR, T, InterleavedFormat, PlanarFormat}
        ) where {MR, NR, T, W}
        _check_vector_shape("FMAddSubKernel", 2 * MR, W, true)
        return new{MR, NR, T, W}(descriptor)
    end
end

function FMAddSubKernel(::Val{MR}, ::Val{NR}, ::Type{T}, ::Val{W}) where {MR, NR, T, W}
    T <: Complex ||
        throw(ArgumentError("FMAddSubKernel requires a complex element type, got $T"))
    return FMAddSubKernel{MR, NR, T, W}(
        ComplexKernelDescriptor(Val(MR), Val(NR), T, InterleavedFormat(), PlanarFormat())
    )
end
function FMAddSubKernel(::Val{MR}, ::Val{NR}, ::Type{T}) where {MR, NR, T}
    T <: Complex ||
        throw(ArgumentError("FMAddSubKernel requires a complex element type, got $T"))
    return FMAddSubKernel(Val(MR), Val(NR), T, Val(_default_lanewidth(real(T))))
end

complex_method(::FMAddSubKernel) = FMAddSubMethod()
lanewidth(::FMAddSubKernel{MR, NR, T, W}) where {MR, NR, T, W} = W

# `x*y - c` in even lanes, `x*y + c` in odd lanes, each one fused rounding.
# Generic IR (`fneg` + two `llvm.fma` + a blend), which the X86 backend folds
# into one `vfmaddsub` wherever FMA3 exists (and elsewhere stays correct).
# GUARDRAIL: the SIMD.jl spelling `shufflevector(muladd(x, y, -c), muladd(x, y,
# c), ...)` also selects `vfmaddsub` but leaves a dead stack store of the
# accumulator in the K loop, every step.
@generated function _fmaddsub(x::Vec{N, R}, y::Vec{N, R}, c::Vec{N, R}) where {N, R}
    R === Float64 || R === Float32 ||
        return :(throw(ArgumentError("_fmaddsub: unsupported lane type $R")))
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
@generated function _swap_pairs(x::Vec{N, R}) where {N, R}
    iseven(N) || return :(throw(ArgumentError("_swap_pairs: expected even N, got $N")))
    idx = ntuple(k -> isodd(k) ? k : k - 2, N)
    return :(Base.@_inline_meta; shufflevector(x, Val($idx)))
end

function zero_accumulator(kernel::FMAddSubKernel{MR, NR, T, W}) where {MR, NR, T, W}
    z = zero(Vec{W, real(T)})
    return ntuple(_ -> z, Val(((2 * MR) ÷ W) * NR))
end

# The INNER op must be the `swap(a) * bi` one: only the inner product is
# subtracted in the real lanes. Reversed, the real part is `ai*bi - ar*br`.
@generated function _accumulate_step_fmaddsub(
        kernel::FMAddSubKernel{MR, NR, T, W}, acc::NTuple{NA, Vec{W, R}},
        packed_a::PA, packed_b::PB, p::Int
    ) where {MR, NR, T, W, R, NA, PA, PB}
    MV = (2 * MR) ÷ W
    _check_acc(:_accumulate_step_fmaddsub, R, T, NA, MV * NR)

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
                    Vec{$W, $R}, packed_a, packed_a_plane_offset(kernel, 0, $(v * W + 1), p)
                )
            )
        )
        push!(load_a, :($(sv[v + 1]) = _swap_pairs($(av[v + 1]))))
    end

    load_b = Any[]
    for j in 1:NR
        push!(load_b, :((br_s, bi_s) = _b_step_load2(packed_b, kernel, $j, p)))
        push!(load_b, :($(brv[j]) = Vec{$W, $R}(br_s)))
        push!(load_b, :($(biv[j]) = Vec{$W, $R}(bi_s)))
    end

    acc_exprs = Vector{Any}(undef, NA)
    for j in 1:NR, v in 0:(MV - 1)
        idx = v + MV * (j - 1) + 1
        acc_exprs[idx] = :(
            _fmaddsub(
                $(av[v + 1]), $(brv[j]),
                _fmaddsub($(sv[v + 1]), $(biv[j]), acc[$idx])
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

function Base.accumulate(
        kernel::FMAddSubKernel{MR, NR, T, W}, acc::NTuple{NA, Vec{W, R}},
        packed_a::PA, packed_b::PB, k_block_length::Int
    ) where {MR, NR, T, W, R, NA, PA, PB}
    k_block_length == 0 && return acc
    k_block_length > 0 || _throw_negative_k_block_length(:accumulate, k_block_length)
    @inbounds for p in 1:k_block_length
        acc = _accumulate_step_fmaddsub(kernel, acc, packed_a, packed_b, p)
    end
    return acc
end

# `p - q` in even lanes, `p + q` in odd lanes: Base's unfused complex `*` on
# interleaved data.
@generated function _addsub(p::Vec{N, R}, q::Vec{N, R}) where {N, R}
    iseven(N) || return :(throw(ArgumentError("_addsub: expected even N, got $N")))
    idx = ntuple(k -> iseven(k - 1) ? k - 1 : N + k - 1, N)
    return :(Base.@_inline_meta; shufflevector(p - q, p + q, Val($idx)))
end

# One full `W÷2`-row block, already in `Complex`'s memory order, so no
# interleave shuffle. Base's `Complex` expression trees in lanes, as planar's
# `_planar_store_block!`:
#     beta == 0:  addsub(ar*r, ai*swap(r))
#     beta == 1:  fmaddsub(ar, r, fmaddsub(ai, swap(r), C))
#     otherwise:  as beta == 1 with C := addsub(br*C, bi*swap(C))
@inline function _fmaddsub_store_block!(
        sp::Ptr{RC}, at::Int, r::Vec{W, R},
        ar::Vec{W, R}, ai::Vec{W, R}, br::Vec{W, R}, bi::Vec{W, R},
        beta::Complex{R}
    ) where {RC, R, W}
    s = _swap_pairs(r)
    if iszero(beta)
        new = _addsub(ar * r, ai * s)
    else
        old = convert(Vec{W, R}, vload(Vec{W, RC}, sp + sizeof(RC) * at))
        x = isone(beta) ? old : _addsub(br * old, bi * _swap_pairs(old))
        new = _fmaddsub(ar, r, _fmaddsub(ai, s, x))
    end
    vstore(convert(Vec{W, RC}, new), sp + sizeof(RC) * at)
    return nothing
end

# Same unroll and full-block / row-tail split as `_store_tile_planar_vector!`.
@generated function _store_tile_fmaddsub_vector!(
        destination::Tile{S, <:AffineAxis}, acc::NTuple{NV, Vec{W, R}},
        alpha::T, beta::T, kernel::DescriptorKernel{MR, NR, T},
        m::Int, n::Int
    ) where {S, MR, NR, T, W, R, NV}
    # The pointer reinterpretation is only sound on dense rank-1 complex storage.
    S <: DenseVector && _lane_convertible(eltype(S), T) ||
        throw(ArgumentError("_store_tile_fmaddsub_vector!: storage $S is not a dense vector convertible to $T"))
    RC = real(eltype(S))
    iseven(W) || throw(ArgumentError("_store_tile_fmaddsub_vector!: requires an even W, got $W"))
    MV = (2 * MR) ÷ W
    _check_acc(:_store_tile_fmaddsub_vector!, R, T, NV, MV * NR)
    HW = W ÷ 2

    blocks = Any[]
    for j in 1:NR
        vblocks = Any[]
        for v in 0:(MV - 1)
            idx = v + MV * (j - 1) + 1
            push!(
                vblocks, quote
                    vec = acc[$idx]
                    if $((v + 1) * HW) <= m
                        _fmaddsub_store_block!(
                            sp, 2 * (colbase + $(v * HW)), vec, ar, ai, br, bi, beta
                        )
                    elseif $(v * HW) < m
                        for u in 1:$HW
                            i = $(v * HW) + u
                            i <= m || break
                            _axpby_at!(
                                storage, colbase + i, alpha,
                                Complex(vec[2 * u - 1], vec[2 * u]), beta
                            )
                        end
                    end
                end
            )
        end
        push!(
            blocks, quote
                if $j <= n
                    colbase = rowbase0 + cols[$j]  # the address of (1, j)
                    $(vblocks...)
                end
            end
        )
    end

    return quote
        storage = destination.storage
        cols = destination.cols
        rowbase0 = @inbounds destination.base + destination.rows[1]
        ar = Vec{$W, $R}(real(alpha))
        ai = Vec{$W, $R}(imag(alpha))
        br = Vec{$W, $R}(real(beta))
        bi = Vec{$W, $R}(imag(beta))
        GC.@preserve storage begin
            sp = reinterpret(Ptr{$RC}, pointer(storage))
            @inbounds begin
                $(blocks...)
            end
        end
        return destination
    end
end

# Not `@inline`, unlike the real and planar stores: inlining it cost time
# (code growth).
function store_tile!(
        destination::Tile, acc::NTuple{NV, Vec{W, R}},
        alpha::T, beta::T, kernel::FMAddSubKernel{MR, NR, T, W}
    ) where {MR, NR, T, W, R, NV}
    m, n = _store_prologue!(destination, alpha, beta)
    (m == 0 || n == 0) && return destination

    if _complex_vector_eligible(destination, T)
        return _store_tile_fmaddsub_vector!(destination, acc, alpha, beta, kernel, m, n)
    end

    return _store_tile_lanepair!(destination, acc, alpha, beta, kernel, m, n)
end
