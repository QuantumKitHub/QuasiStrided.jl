# The K steps and the generic `add_tile`. A vector kernel's accumulator is an
# immutable tuple of `Vec`s so it stays in registers, and every K step and store
# is `@generated` straight-line code.
#
# GUARDRAIL (all kernels): every `acc[...]` must be a *literal* tuple index.
# Indexing an `NTuple` dynamically forces it to memory, and above 16 vectors
# the compiler heap-allocates it on every call. Only the lane index inside one
# `Vec` may be a runtime value.

using SIMD: Vec

zero_accumulator(::ScalarKernel{MR, NR, T}) where {MR, NR, T} = ntuple(_ -> zero(T), Val(MR * NR))

function add_tile(
        kernel::ScalarKernel{MR, NR, T}, acc::NTuple{N, T},
        packed_a::PA, packed_b::PB, k_block_length::Int
    ) where {MR, NR, T, N, PA <: PackedPanel, PB}
    k_block_length == 0 && return acc
    k_block_length > 0 || throw_negative_k_block_length(:add_tile, k_block_length)
    for p in 1:k_block_length
        acc = add_step(kernel, acc, packed_a, packed_b, p)
    end
    return acc
end

# One K step: element `(i, j)` of the tile is `acc[i + MR * (j - 1)]`.
@inline function add_step(
        kernel::ScalarKernel{MR, NR, T}, acc::NTuple{N, T}, packed_a, packed_b, p::Int
    ) where {MR, NR, T, N}
    return ntuple(Val(N)) do q
        i, j = mod1(q, MR), cld(q, MR)
        a = panel_load(packed_a, packed_a_offset(kernel, i, p))
        b = panel_load(packed_b, packed_b_offset(kernel, j, p))
        muladd(a, b, acc[q])
    end
end

function zero_accumulator(kernel::VectorKernel{MR, NR, T, W}) where {MR, NR, T, W}
    layout = AccumulatorLayout(kernel)
    z = zero(Vec{W, real(T)})
    return ntuple(_ -> z, Val(accumulator_length(layout, MR ÷ rows_per_vector(layout, W), NR)))
end

@inline function add_tile(
        kernel::VectorKernel, acc::NTuple{NA, Vec{W, R}},
        packed_a::PA, packed_b::PB, k_block_length::Int
    ) where {NA, W, R, PA <: PackedPanel, PB}
    k_block_length == 0 && return acc
    k_block_length > 0 || throw_negative_k_block_length(:add_tile, k_block_length)
    @inbounds for p in 1:k_steps(kernel, k_block_length)
        acc = accumulate_step(kernel, acc, packed_a, packed_b, p)
    end
    return acc
end

# B column `j` at K step `p`, as a real or as `(re, im)`; `UnpackedBView`
# (src/execution/unpackedb.jl) overrides both to read B in place, resolved at
# compile time.
@inline b_scalar(packed_b::PB, kernel, j::Int, p::Int) where {PB} =
    panel_load(packed_b, packed_b_offset(kernel, j, p))
@inline b_complex(packed_b::PB, kernel, j::Int, p::Int) where {PB} = (
    panel_load(packed_b, packed_b_offset(kernel, j, p)),
    panel_load(packed_b, packed_b_offset(kernel, j, p, 1)),
)

@generated function accumulate_step(
        kernel::SIMDKernel{MR, NR, T, W}, acc::NTuple{NV, Vec{W, T}},
        packed_a::PA, packed_b::PB, p::Int
    ) where {MR, NR, T, W, NV, PA, PB}
    NVECA = MR ÷ W
    check_acc(:accumulate_step, T, T, NV, NVECA * NR)

    avars = [Symbol(:a, v) for v in 0:(NVECA - 1)]
    bvars = [Symbol(:b, j) for j in 1:NR]

    load_a = [
        :($(avars[v + 1]) = panel_vload(Vec{$W, $T}, packed_a, packed_a_offset(kernel, $(v * W + 1), p)))
            for v in 0:(NVECA - 1)
    ]
    load_b = [
        :($(bvars[j]) = b_scalar(packed_b, kernel, $j, p))
            for j in 1:NR
    ]

    acc_exprs = Vector{Any}(undef, NV)
    for j in 1:NR, v in 0:(NVECA - 1)
        idx = acc_index(NVECA, v, j)
        acc_exprs[idx] = :(muladd($(avars[v + 1]), $(bvars[j]), acc[$idx]))
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

# A kernel without a K step of its own runs its `inner(kernel)`'s.
@inline accumulate_step(kernel::VectorKernel, acc::NTuple, packed_a::PA, packed_b::PB, p::Int) where {PA, PB} =
    accumulate_step(inner(kernel), acc, packed_a, packed_b, p)

# Planar (split-complex, BLIS "1r"), the default complex method. Both panels
# are `PlanarFormat` (`[re_0..re_{n-1} | im_0..im_{n-1}]` per K step), so the
# data is already in the right lanes: four real FMAs per (A-vector, B-scalar)
# pair, no shuffles. Everything works in `real(T)`.
#
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

# FMAddSub (interleaved accumulator): one accumulator plane in 1m's layout
# (lanes `(2u-1, 2u)` are `(re, im)` of one complex row), updated per
# (A-vector, B-column) pair by two chained x86 `vfmaddsub` ops:
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
#
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
