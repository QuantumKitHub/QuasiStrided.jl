# FMADDSUB (interleaved-accumulator) complex microkernel.
#
# One interleaved accumulator plane -- lane pair `(2u, 2u+1)` of a vector is
# `(re, im)` of one complex row, 1m's layout exactly -- updated per
# (A-vector, B-column) pair by TWO chained x86 `vfmaddsub` ops:
#
#     acc = fmaddsub(a, br, fmaddsub(swap(a), bi, acc))
#
# with `fmaddsub(x, y, c)` = `x*y - c` in even (0-based) lanes and `x*y + c` in
# odd lanes (Intel's `vfmaddsub` semantics), `a = [ar0, ai0, ar1, ai1, ...]`,
# `swap(a) = [ai0, ar0, ai1, ar1, ...]` and `br`/`bi` broadcast scalars. The
# alternating sign NEGATES the accumulator in the even lanes, and two such ops
# compose back to an accumulation (the `vfmsac(vfmsac(c,u,v),w,y) = c + w*y -
# u*v` identity OpenBLAS's kernels use):
#
#     even:  ar*br - (ai*bi - c_re)  =  c_re + ar*br - ai*bi       (real part)
#     odd:   ai*br + (ar*bi + c_im)  =  c_im + ai*br + ar*bi       (imag part)
#
# Each step is a single rounding of a fused op, and negation is exact, so both
# lanes are chains of exactly two correctly-rounded FMAs -- the same rounding
# budget as planar's `muladd(-ai, bi, muladd(ar, br, c))`, with a different
# order in the real part (planar folds `ar*br` first, this folds `ai*bi`
# first; the imaginary part's order coincides). Not bitwise planar: compare
# with a tolerance.
#
# WHY THIS NEEDS ITS OWN PACKED A FORMAT (checked, not assumed): `vfmaddsub`'s
# sign alternates by LANE PARITY, so it only computes complex arithmetic when
# the real and imaginary parts of one element sit in adjacent lanes of the
# same register. `PlanarFormat` puts them in different registers (all `re`,
# then all `im`), where every lane of a vector needs the same sign and a
# lane-alternating op is simply wrong. Reading planar A would need an
# unpack-lo/hi interleave of the two planes per A vector per K step (plus the
# swap), i.e. strictly more shuffles than this design, for no benefit. Hence
# `InterleavedFormat` A (src/packing/format.jl) -- the native `Complex{T}`
# order, 2 reals/element, planar's footprint and half of 1e's. B stays
# `PlanarFormat`: it is only ever read as broadcast scalars, and a broadcast is
# lane-uniform, so B's layout is irrelevant to the lane-sign question.
#
# INSTRUCTION COUNT, honestly: a complex multiply-accumulate is 8 real flops,
# i.e. 4 FMA-lanes, whatever the representation. Per logical K step at the
# same `(MR, NR, W)` this kernel issues `2*MV*NR` fmaddsub over `MV = 2MR÷W`
# vectors = `4*MR*NR÷W` fused ops, which is EXACTLY planar's `4*(MR÷W)*NR` and
# 1m's `2 * (2MR÷W)*NR`. The "half the instructions" intuition does not hold
# against this codebase's planar kernel, which already issues zero separate
# negations (see the GUARDRAIL in planar.jl). What differs is the overhead
# around the FMAs:
#
#   per logical K step     planar          1m               fmaddsub
#   fused ops              4MR*NR/W        4MR*NR/W         4MR*NR/W
#   A vector loads         2MR/W           4MR/W            2MR/W
#   B broadcasts           2NR             2NR              2NR
#   shuffles               0               0                2MR/W (the swap)
#   accumulator vectors    2MR*NR/W        2MR*NR/W         2MR*NR/W
#   packed A reals/elt     2               4                2
#
# i.e. fmaddsub = 1m's register/accumulator structure at planar's A
# footprint, paid for with one pair-swap per A vector per K step (amortised
# over NR columns). Whether that trade wins is a measurement question
# (benchmark/bench_complex_efficiency.jl arm 2), not an instruction-count one.
# Measured 2026-09-25 (jobs 7109276 Rome, 7109277 Ice Lake; full numbers at
# `KERNEL_SHAPES_C64_FMADDSUB` in src/planning/kernel_selection.jl): +4% over
# the best of planar and 1m for ComplexF64 on AVX2 (`4x5/W4`), +14-22% over 1m at
# 1m's own AVX-512 ComplexF64 shapes but still behind planar `24x3/W8`, and a
# tie-or-loss for ComplexF32 on both. Nowhere near the naive 2x, as the count
# above predicts.
#
# INSTRUCTION SELECTION: SIMD.jl has no fmaddsub primitive; `_fmaddsub` below
# is a generic-IR `llvmcall` (fneg + two `llvm.fma` + even/odd blend), which
# LLVM's X86 backend folds into `vfmaddsub{132,213,231}p{d,s}` -- base FMA3, so
# AVX2 and AVX-512 alike. Verified by `@code_native` of the `accumulate` hot
# loop with the driver's `PackedPanel` arguments
# (benchmark/probes/fmaddsub_codegen.jl), 2026-09-25, Julia 1.13.0 and 1.12.7,
# host cascadelake at every menu shape and `-C znver2` at the AVX2-native
# shapes `(4,*,4)`/`(8,*,8)`: exactly `2*MV*NR` `vfmaddsub231p*`, ZERO
# `vfmadd`/`vfnmadd`/`vmul`/`vadd`/`vsub`, and exactly `MV`
# `vshufpd`/`vpermilps` (the swap) per K step. Stack traffic in the loop: zero
# on cascadelake at every shape; under `-C znver2` (16 ymm) zero at the `NR = 5`
# shapes `(4,5,4)`/`(8,5,8)`, but 25 stores/22 reloads at `(4,6,4)`/`(8,6,8)` on
# 1.12.7 (2/2 on 1.13.0) -- pressure 18 > 16, Cliff A, as
# `fmaddsub_register_pressure` predicts. The AVX-512-sized W=8/W=16 shapes
# under `-C znver2` are emulated 2-4 ymm wide and spill heavily, exactly like
# planar's and 1m's AVX-512 shapes there (and on 1.13.0 LLVM also un-fuses them
# into plain FMA pairs); they are not AVX2 candidates. The host-ISA part is
# re-checked by test/microkernels/test_fmaddsub_kernel.jl's
# instruction-selection testset.
#
# Cliff B applies exactly as in planar.jl/onem.jl: accumulate and store are
# `@generated` with literal tuple indices, including the lane tail.

using SIMD: Vec, shufflevector

"""
    FMAddSubKernel{MR,NR,T,W}(descriptor)
    FMAddSubKernel(::Val{MR}, ::Val{NR}, ::Type{T}, ::Val{W})
    FMAddSubKernel(::Val{MR}, ::Val{NR}, ::Type{T})

Interleaved-accumulator complex microkernel over `SIMD.Vec{W,real(T)}` lanes,
implementing [`FMAddSubMethod`](@ref): [`InterleavedFormat`](@ref) A,
[`PlanarFormat`](@ref) B. `MR`/`NR` are **logical** (complex) extents, `T` the
storage type (`ComplexF32`/`ComplexF64`), `W` the lane count of the real type.
As for [`OneMKernel`](@ref), it is `2MR` (reals per A sliver per K step) that
must be a multiple of `W`, and `W` must be even so a complex element never
straddles a vector.

Never selected automatically; name it in `plan_contract(...; kernel = ...)`.
The 3-argument form defaults `W` via `_default_lanewidth(real(T))`.
"""
struct FMAddSubKernel{MR, NR, T, W} <: DescriptorKernel{MR, NR, T}
    descriptor::ComplexKernelDescriptor{MR, NR, T, InterleavedFormat, PlanarFormat}

    function FMAddSubKernel{MR, NR, T, W}(
            descriptor::ComplexKernelDescriptor{MR, NR, T, InterleavedFormat, PlanarFormat}
        ) where {MR, NR, T, W}
        _check_lanewidth("FMAddSubKernel", W)
        # GUARDRAIL, as in OneMKernel: an odd `W` can satisfy `mod(2MR, W) == 0`
        # while a complex element straddles two vectors -- then the lane-parity
        # sign of `vfmaddsub` lands on the wrong half. Wrong answer, not an error.
        iseven(W) ||
            throw(
            ArgumentError(
                "FMAddSubKernel requires an even vector width W (re/im of one " *
                    "element must be adjacent lanes of one Vec), got W = $W"
            )
        )
        mod(2 * MR, W) == 0 ||
            throw(
            ArgumentError(
                "FMAddSubKernel requires 2*mr(kernel) = $(2 * MR) to be a multiple " *
                    "of the vector width W = $W"
            )
        )
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

"""
    lanewidth(kernel::FMAddSubKernel) -> Int

The kernel's `SIMD.Vec` lane width `W`, counted in **reals**. Always even.
"""
lanewidth(::FMAddSubKernel{MR, NR, T, W}) where {MR, NR, T, W} = W

"""
    avecs_per_column(kernel::FMAddSubKernel) -> Int

`2*mr(kernel) ÷ lanewidth(kernel)`: A vectors (and accumulator vectors) per
output column, each covering `W÷2` complex rows -- 1m's count.
"""
avecs_per_column(::FMAddSubKernel{MR, NR, T, W}) where {MR, NR, T, W} = (2 * MR) ÷ W

"""
    fmaddsub_register_pressure(kernel::FMAddSubKernel) -> Int

Vector registers live at the bottom of the K loop if every A vector and its
swap are held across all `NR` columns:

    MV*NR accumulators + 2*MV A vectors (a, swap(a)) + 2 B broadcasts

with `MV = 2*mr(kernel) ÷ lanewidth(kernel)`. **Cliff A**: a *necessary*
condition against `target_profile().nregisters`, never a ranking (see
`planar_register_pressure`). It is `MV + 1` more than 1m's at the same shape:
`MV` for holding `swap(a)` next to `a` (1m loads 1e's second region instead,
one real K step later) and 1 for the second (`bi`) broadcast.
"""
fmaddsub_register_pressure(::FMAddSubKernel{MR, NR, T, W}) where {MR, NR, T, W} =
    ((2 * MR) ÷ W) * NR + 2 * ((2 * MR) ÷ W) + 2

# ----------------------------------------------------------------------------
# The two lane primitives
# ----------------------------------------------------------------------------

# `x*y - c` in even (0-based) lanes, `x*y + c` in odd lanes: x86 `vfmaddsub`.
#
# Written as ONE self-contained `llvmcall` of GENERIC LLVM IR -- `fneg`, two
# `llvm.fma` calls and an even/odd `shufflevector` blend -- never an x86
# intrinsic (the `llvm.x86.fma.vfmaddsub.*` intrinsics are auto-upgraded to
# exactly this pattern by LLVM anyway). `llvm.fma` is always fused and `fneg`
# is an exact sign flip, so each lane is ONE correctly rounded fused op on
# every target; X86ISelLowering folds the blend-of-two-FMAs into a single
# `vfmaddsub` wherever FMA3 exists, and elsewhere it stays correct (a libm
# `fma` call at worst -- slow, never wrong).
#
# GUARDRAIL: the obvious SIMD.jl spelling,
#
#     shufflevector(muladd(x, y, -c), muladd(x, y, c), Val(idx))
#
# is REJECTED. It also selects `vfmaddsub` (checked), but inside the
# `accumulate` loop it leaves the accumulator tuple's stack copy alive: one
# dead `vmovupd [rbp - ...]` store per accumulator vector but one, EVERY K
# step (9 at 4x5/W4, 15 at 8x8/W8; Julia 1.13.0, 2026-09-25,
# benchmark/probes/fmaddsub_codegen.jl). This form has zero.
_fmaddsub_llvm_type(::Type{Float64}) = "double"
_fmaddsub_llvm_type(::Type{Float32}) = "float"
_fmaddsub_llvm_suffix(::Type{Float64}) = "f64"
_fmaddsub_llvm_suffix(::Type{Float32}) = "f32"

@generated function _fmaddsub(x::Vec{N, R}, y::Vec{N, R}, c::Vec{N, R}) where {N, R}
    R === Float64 || R === Float32 ||
        return :(throw(ArgumentError("_fmaddsub: unsupported lane type $R")))
    ty = "<$N x $(_fmaddsub_llvm_type(R))>"
    fn = "llvm.fma.v$(N)$(_fmaddsub_llvm_suffix(R))"
    # lane k (0-based): even -> k of `s` (x*y - c); odd -> k of `d` (x*y + c),
    # i.e. index N + k in the two-operand index space.
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

# Adjacent-pair swap `[x1, x0, x3, x2, ...]`: one in-lane `vshufpd`/`vpermilps`.
@generated function _swap_pairs(x::Vec{N, R}) where {N, R}
    iseven(N) || return :(throw(ArgumentError("_swap_pairs: expected even N, got $N")))
    idx = ntuple(k -> isodd(k) ? k : k - 2, N)
    return :(Base.@_inline_meta; shufflevector(x, Val($idx)))
end

# ----------------------------------------------------------------------------
# zero_accumulator, accumulate
# ----------------------------------------------------------------------------

"""
    zero_accumulator(kernel::FMAddSubKernel{MR,NR,T,W}) -> NTuple{NV,SIMD.Vec{W,real(T)}}

`NV = (2MR÷W)*NR` zero vectors, entry `(v, j)` at tuple position
`v + (2MR÷W)*j + 1`. Complex row `i = v*(W÷2) + u` of column `j` is lanes
`2u+1` (re) and `2u+2` (im), 1-based -- the same layout as
`zero_accumulator(::OneMKernel)`.
"""
function zero_accumulator(kernel::FMAddSubKernel{MR, NR, T, W}) where {MR, NR, T, W}
    z = zero(Vec{W, real(T)})
    return ntuple(_ -> z, Val(((2 * MR) ÷ W) * NR))
end

# One logical K step, fully unrolled with literal tuple indices (Cliff B).
# `MV` A vector loads, `MV` swaps, `2NR` B scalar loads, `2*MV*NR` fmaddsub.
# The INNER op must be the `swap(a) * bi` one: it is the term the even (real)
# lanes subtract, and only the inner op's product is subtracted (the outer
# op's is added in both lanes). Reversing the nesting computes
# `ai*bi - ar*br` in the real part -- wrong, not merely re-rounded; the
# "nesting order" test in test_fmaddsub_kernel.jl pins this.
@generated function _accumulate_step_fmaddsub(
        kernel::FMAddSubKernel{MR, NR, T, W}, acc::NTuple{NA, Vec{W, R}},
        packed_a::PA, packed_b::PB, p::Int
    ) where {MR, NR, T, W, R, NA, PA, PB}
    R === real(T) ||
        throw(
        ArgumentError(
            "_accumulate_step_fmaddsub: accumulator lane type $R does not match " *
                "real($T) = $(real(T))"
        )
    )
    MV = (2 * MR) ÷ W
    MV * NR == NA ||
        throw(
        ArgumentError(
            "_accumulate_step_fmaddsub: accumulator length $NA does not match " *
                "(2*mr÷W)*nr = $(MV * NR) for MR=$MR, NR=$NR, W=$W"
        )
    )

    av = [Symbol(:a, v) for v in 0:(MV - 1)]
    sv = [Symbol(:s, v) for v in 0:(MV - 1)]
    brv = [Symbol(:br, j) for j in 0:(NR - 1)]
    biv = [Symbol(:bi, j) for j in 0:(NR - 1)]

    load_a = Any[]
    for v in 0:(MV - 1)
        # InterleavedFormat: plane 0, `index` over reals (see src/packing/format.jl).
        push!(
            load_a,
            :(
                $(av[v + 1]) = panel_vload(
                    Vec{$W, $R}, packed_a, packed_a_plane_offset(kernel, 0, $(v * W), p)
                )
            )
        )
        push!(load_a, :($(sv[v + 1]) = _swap_pairs($(av[v + 1]))))
    end

    load_b = Any[]
    for j in 0:(NR - 1)
        push!(
            load_b,
            :($(brv[j + 1]) = Vec{$W, $R}(panel_load(packed_b, packed_b_plane_offset(kernel, 0, $j, p))))
        )
        push!(
            load_b,
            :($(biv[j + 1]) = Vec{$W, $R}(panel_load(packed_b, packed_b_plane_offset(kernel, 1, $j, p))))
        )
    end

    acc_exprs = Vector{Any}(undef, NA)
    for j in 0:(NR - 1), v in 0:(MV - 1)
        idx = v + MV * j + 1
        acc_exprs[idx] = :(
            _fmaddsub(
                $(av[v + 1]), $(brv[j + 1]),
                _fmaddsub($(sv[v + 1]), $(biv[j + 1]), acc[$idx])
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

"""
    accumulate(kernel::FMAddSubKernel, acc, packed_a, packed_b, kc::Int) -> acc

`kc` logical K steps of [`FMAddSubMethod`](@ref). `kc == 0` returns `acc`
unchanged without reading the panels. `packed_a` must be in
[`InterleavedFormat`](@ref), `packed_b` in [`PlanarFormat`](@ref). Not bitwise
identical to planar or 1m (FMA order differs in the real part) -- compare
with a tolerance, never `==`.
"""
function Base.accumulate(
        kernel::FMAddSubKernel{MR, NR, T, W}, acc::NTuple{NA, Vec{W, R}},
        packed_a::PA, packed_b::PB, kc::Int
    ) where {MR, NR, T, W, R, NA, PA, PB}
    kc == 0 && return acc
    kc > 0 || throw(ArgumentError("accumulate requires kc >= 0, got kc = $kc"))
    @inbounds for p in 0:(kc - 1)
        acc = _accumulate_step_fmaddsub(kernel, acc, packed_a, packed_b, p)
    end
    return acc
end

# ----------------------------------------------------------------------------
# store_tile!
# ----------------------------------------------------------------------------

# The interleaved tile reader: 1m's `_store_tile_onem!` (src/microkernels/
# onem.jl) verbatim but for the kernel type -- the accumulator layout is the
# same, so is the reader, and the test suite checks the two agree bitwise on
# one accumulator. The scalar fallback of `store_tile!` below, for every
# destination `_store_tile_fmaddsub_vector!` is ineligible for.
@generated function _store_tile_fmaddsub!(
        destination::QSTile, acc::NTuple{NV, Vec{W, R}},
        alpha::T, beta::T, kernel::FMAddSubKernel{MR, NR, T, W},
        m::Int, n::Int
    ) where {MR, NR, T, W, R, NV}
    R === real(T) ||
        throw(
        ArgumentError(
            "_store_tile_fmaddsub!: accumulator lane type $R does not match " *
                "real($T) = $(real(T))"
        )
    )
    iseven(W) ||
        throw(ArgumentError("_store_tile_fmaddsub!: the lane-pair reader requires an even W, got $W"))
    MV = (2 * MR) ÷ W
    MV * NR == NV ||
        throw(
        ArgumentError(
            "_store_tile_fmaddsub!: accumulator length $NV does not match " *
                "(2*mr÷W)*nr = $(MV * NR) for MR=$MR, NR=$NR, W=$W"
        )
    )
    HW = W ÷ 2

    blocks = Any[]
    for j in 0:(NR - 1), v in 0:(MV - 1)
        idx = v + MV * j + 1
        push!(
            blocks, quote
                if $j < n
                    vec = acc[$idx]
                    for u in 0:$(HW - 1)
                        i = $(v * HW) + u
                        i < m || break
                        _axpby_tile!(
                            destination, i, $j, alpha,
                            Complex(vec[2 * u + 1], vec[2 * u + 2]), beta
                        )
                    end
                end
            end
        )
    end
    return quote
        @inbounds begin
            $(blocks...)
        end
        return destination
    end
end

# ----------------------------------------------------------------------------
# The vectorized store for a unit-stride destination
# ----------------------------------------------------------------------------
#
# The accumulator is already in `Complex{R}`'s native `[re, im, re, im, ...]`
# order, so unlike planar's `_store_tile_planar_vector!` no interleave shuffle
# is needed: one `W`-real accumulator vector IS `W÷2` consecutive destination
# elements, and `C = alpha*R + beta*C` is computed lane-wise with the two
# primitives the K loop already uses. Per full block:
#
#     beta == 0:  alpha*r            = addsub(ar*r, ai*swap(r))
#     beta == 1:  muladd(alpha, r, C) = fmaddsub(ar, r, fmaddsub(ai, swap(r), C))
#     otherwise:  as beta == 1 with C := addsub(br*C, bi*swap(C))
#
# where `swap` is `_swap_pairs`, `addsub(p, q)` is `p - q` in even (real) lanes
# and `p + q` in odd (imaginary) lanes, and `fmaddsub(x, y, c)` is `x*y - c` /
# `x*y + c` fused. These are exactly Base's `Complex` expression trees as
# `_axpby_tile!` (src/microkernels/interface.jl) reaches them -- `*` is
# `(zr*wr - zi*wi, zr*wi + zi*wr)`, two separately rounded products, and
# `muladd(z, w, x)` is `(muladd(zr, wr, -muladd(zi, wi, -xr)), muladd(zr, wi,
# muladd(zi, wr, xi)))`, fused at both levels -- so on the lanes it vectorizes
# the fast path is bitwise identical to a fused transcription of those trees
# (test/microkernels/test_fmaddsub_store_fastpath.jl pins this with `isequal`),
# and agrees with the scalar fallback exactly at `beta == 0` and `beta == 1`
# and to within LLVM's `muladd` contraction choice otherwise, as planar's does.
#
# Why it exists: the scalar reader above was 46% of an 8x8x8 ComplexF64
# `execute!` (0.19 of 0.41 us, ~3 ns per element through `_axpby_tile!`;
# ccqlin038, Julia 1.12.7, profile of `execute!` alone) -- and this kernel is
# the one the AVX-512 small-M demotion selects for every complex contraction
# with `Qm < 24`, so that store is on the per-call floor of exactly the small
# problems where the floor is what matters. Measured, same host and probe
# (medians of 15 x 2000 calls, scalar store then this one, back to back, not
# interleaved), `execute!` on a dense ComplexF64 GEMM at the `8x8/W8` shape:
# 8x8x8 0.41 -> 0.27 us, 16x16x16 1.64 -> 1.00 us, 4x4 outer product 0.20 ->
# 0.15 us; through the TensorOperations adapter the 16x16x16 call went from
# 1.33x StridedBLAS's time to 0.93x. Re-measured interleaved (two rounds,
# benchmark/probes/probe_call_floor.jl, scalar-only store vs this one, whole
# `tensorcontract!` calls): 8x8x8 0.68 -> 0.53 us, 16x16x16 1.97 -> 1.35 us,
# dim-4 outer product 1.45 -> 0.84 us.

# `p - q` in even lanes, `p + q` in odd lanes, both separately rounded: Base's
# unfused `Complex` `*` on interleaved data.
@generated function _addsub(p::Vec{N, R}, q::Vec{N, R}) where {N, R}
    iseven(N) || return :(throw(ArgumentError("_addsub: expected even N, got $N")))
    idx = ntuple(k -> iseven(k - 1) ? k - 1 : N + k - 1, N)
    return :(Base.@_inline_meta; shufflevector(p - q, p + q, Val($idx)))
end

# One full `W÷2`-row block of one column: `at` is the ZERO-based index of the
# block's first REAL in the reinterpreted storage, i.e. twice the complex
# element index. `ar`/`ai`/`br`/`bi` are pre-broadcast once per `store_tile!`
# call; `beta` itself is passed only so the two `iszero`/`isone` tests are the
# identical tests `_axpby_tile!` makes on the identical value.
@inline function _fmaddsub_store_block!(
        sp::Ptr{R}, at::Int, r::Vec{W, R},
        ar::Vec{W, R}, ai::Vec{W, R}, br::Vec{W, R}, bi::Vec{W, R},
        beta::Complex{R}
    ) where {R, W}
    s = _swap_pairs(r)
    if iszero(beta)
        new = _addsub(ar * r, ai * s)
    else
        old = vload(Vec{W, R}, sp + sizeof(R) * at)
        x = isone(beta) ? old : _addsub(br * old, bi * _swap_pairs(old))
        new = _fmaddsub(ar, r, _fmaddsub(ai, s, x))
    end
    vstore(new, sp + sizeof(R) * at)
    return nothing
end

# Vectorized FMAddSub store. The same `@generated` unroll as planar's
# `_store_tile_planar_vector!` -- every `acc[...]` a literal tuple index
# (Cliff B), the same full-block vs. row-tail split, the same `j < n` column
# guard -- with a "row block" being one accumulator vector of `W÷2` complex
# rows. `rows::AffineAxis` is pinned in the signature so an ineligible tile is
# a MethodError rather than a wrong answer; the caller checks
# `_complex_vector_eligible` first. The scalar row tail goes through
# `storage` itself via `_axpby_at!`, character-identical to the fallback's
# arithmetic. GC.@preserve: the raw `Ptr{R}` is derived from `storage` and
# every dereference happens inside the preserve block.
@generated function _store_tile_fmaddsub_vector!(
        destination::QSTile{S, <:AffineAxis}, acc::NTuple{NV, Vec{W, R}},
        alpha::T, beta::T, kernel::FMAddSubKernel{MR, NR, T, W},
        m::Int, n::Int
    ) where {S, MR, NR, T, W, R, NV}
    R === real(T) ||
        throw(
        ArgumentError(
            "_store_tile_fmaddsub_vector!: accumulator lane type $R does not match " *
                "real($T) = $(real(T))"
        )
    )
    # The bitcast below is only sound on rank-1 dense storage of exactly `T`.
    S <: DenseVector{T} ||
        throw(
        ArgumentError(
            "_store_tile_fmaddsub_vector!: destination storage $S is not a DenseVector{$T}"
        )
    )
    iseven(W) ||
        throw(ArgumentError("_store_tile_fmaddsub_vector!: requires an even W, got $W"))
    MV = (2 * MR) ÷ W
    MV * NR == NV ||
        throw(
        ArgumentError(
            "_store_tile_fmaddsub_vector!: accumulator length $NV does not match " *
                "(2*mr÷W)*nr = $(MV * NR) for MR=$MR, NR=$NR, W=$W"
        )
    )
    HW = W ÷ 2

    blocks = Any[]
    for j in 0:(NR - 1)
        vblocks = Any[]
        for v in 0:(MV - 1)
            idx = v + MV * j + 1
            push!(
                vblocks, quote
                    vec = acc[$idx]
                    if $((v + 1) * HW) <= m
                        # Rows v*HW .. v*HW+HW-1 all inside [0, m): one W-real
                        # load/store at that block's first real.
                        _fmaddsub_store_block!(
                            sp, 2 * (colbase + $(v * HW)), vec, ar, ai, br, bi, beta
                        )
                    elseif $(v * HW) < m
                        # Row tail: this vector straddles m, so store the
                        # valid rows one at a time through the SAME
                        # `_axpby_at!` arithmetic the scalar reader uses.
                        for u in 0:$(HW - 1)
                            i = $(v * HW) + u
                            i < m || break
                            _axpby_at!(
                                storage, colbase + i + 1, alpha,
                                Complex(vec[2 * u + 1], vec[2 * u + 2]), beta
                            )
                        end
                    end
                end
            )
        end
        push!(
            blocks, quote
                if $j < n
                    colbase = rowbase0 + axis_offset(cols, $j)  # zero-based element address of (i=0, j)
                    $(vblocks...)
                end
            end
        )
    end

    return quote
        storage = destination.storage
        cols = destination.cols
        # zero-based element address at (i=0, j=0)'s row contribution; rows are
        # unit-stride, so row `i` is `rowbase0 + i`.
        rowbase0 = destination.base + destination.rows.base
        ar = Vec{$W, $R}(real(alpha))
        ai = Vec{$W, $R}(imag(alpha))
        br = Vec{$W, $R}(real(beta))
        bi = Vec{$W, $R}(imag(beta))
        GC.@preserve storage begin
            sp = reinterpret(Ptr{$R}, pointer(storage))
            @inbounds begin
                $(blocks...)
            end
        end
        return destination
    end
end

"""
    store_tile!(destination::QSTile, acc, alpha::T, beta::T, kernel::FMAddSubKernel) -> destination

Same contract as every other kernel's `store_tile!`: `C = alpha*R + beta*C`
over the valid rectangle only, `alpha == 0` never reads `acc`, `beta == 0`
never reads old `C`, padding lanes are never read, an empty destination is a
no-op.

Two paths, chosen by `_complex_vector_eligible` (src/microkernels/planar.jl),
the same gate as planar's store: a unit-stride `AffineAxis` row axis into
rank-1 dense `Complex` storage on a shipped ISA takes
`_store_tile_fmaddsub_vector!`, one `W`-real load (only when `beta != 0`)
and store per full `W÷2`-row block with the arithmetic done in the
accumulator's own interleaved layout; everything else takes the scalar
reader `_store_tile_fmaddsub!`, which reads the interleaved accumulator
exactly as 1m's does. The two agree bitwise at `beta == 0` and `beta == 1`;
compare the general-`beta` case with a tolerance, as for planar.
"""
function store_tile!(
        destination::QSTile, acc::NTuple{NV, Vec{W, R}},
        alpha::T, beta::T, kernel::FMAddSubKernel{MR, NR, T, W}
    ) where {MR, NR, T, W, R, NV}
    m, n = _store_prologue!(destination, alpha, beta)
    (m == 0 || n == 0) && return destination

    if _complex_vector_eligible(destination, T)
        return _store_tile_fmaddsub_vector!(destination, acc, alpha, beta, kernel, m, n)
    end

    return _store_tile_fmaddsub!(destination, acc, alpha, beta, kernel, m, n)
end
