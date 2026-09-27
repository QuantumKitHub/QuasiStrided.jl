# Planar (split-complex, BLIS "1r") microkernel: the default complex method.
#
# Both packed panels hold `[re_0 .. re_{n-1} | im_0 .. im_{n-1}]` per logical K
# step (`PlanarFormat`), so the data is already in the right lanes and the body
# is four real FMAs per (A-vector, B-scalar) pair -- no shuffles, no `fmaddsub`,
# no duplicated lanes. Everything below this boundary works exclusively in
# `real(T)`: packed buffers, `SIMD.Vec` lanes and the accumulator.
#
# Two *independent* performance cliffs run through this file and must not be
# conflated:
#
#   Cliff A -- architectural register spill. Hardware. See
#   `planar_register_pressure`.
#
#   Cliff B -- Julia's tuple lowering. Dynamic `NTuple` indexing above NV = 16
#   makes the compiler heap-allocate the accumulator on every call. Planar at
#   16x6 is NV_total = 24, so the cliff is live at ordinary shapes. Therefore
#   *every* accumulate and store here is `@generated` with literal tuple
#   indices, including the lane tail; no runtime tuple index may be
#   introduced.

using SIMD: Vec, vload, vstore, shufflevector

"""
    PlanarKernel{MR,NR,T,W}(descriptor)
    PlanarKernel(::Val{MR}, ::Val{NR}, ::Type{T}, ::Val{W})
    PlanarKernel(::Val{MR}, ::Val{NR}, ::Type{T})

Split-complex microkernel over `SIMD.Vec{W,real(T)}` lanes, implementing
[`PlanarMethod`](@ref). `MR`/`NR` are the **logical** (complex) register-tile
extents and `T` is the **storage** element type (`ComplexF32`/`ComplexF64`);
`W` is the lane count of the *real* type, and `MR` must be a multiple of it
(checked at construction). `NR` need not be — B contributes one complex scalar
per column per K step.

The field is named `descriptor`, so `mr`/`nr`/`scalartype`/`packed_a_length`/
`packed_b_length`/`realtype`/`packed_a_per_k`, and the two plane-offset
accessors, all forward through the `DescriptorKernel` methods in
src/microkernels/interface.jl; this kernel adds no forwarding of its own.

The 3-argument form defaults `W` via `_default_lanewidth(real(T))`.
"""
struct PlanarKernel{MR, NR, T, W} <: DescriptorKernel{MR, NR, T}
    descriptor::ComplexKernelDescriptor{MR, NR, T, PlanarFormat, PlanarFormat}

    function PlanarKernel{MR, NR, T, W}(
            descriptor::ComplexKernelDescriptor{MR, NR, T, PlanarFormat, PlanarFormat}
        ) where {MR, NR, T, W}
        _check_lanewidth("PlanarKernel", W)
        _check_mr_multiple("PlanarKernel", MR, W)
        return new{MR, NR, T, W}(descriptor)
    end
end

function PlanarKernel(::Val{MR}, ::Val{NR}, ::Type{T}, ::Val{W}) where {MR, NR, T, W}
    return PlanarKernel{MR, NR, T, W}(
        ComplexKernelDescriptor(Val(MR), Val(NR), T, PlanarFormat(), PlanarFormat())
    )
end
function PlanarKernel(::Val{MR}, ::Val{NR}, ::Type{T}) where {MR, NR, T}
    T <: Complex ||
        throw(ArgumentError("PlanarKernel requires a complex element type, got $T"))
    return PlanarKernel(Val(MR), Val(NR), T, Val(_default_lanewidth(real(T))))
end

complex_method(::PlanarKernel) = PlanarMethod()

"""
    lanewidth(kernel::PlanarKernel) -> Int

The kernel's `SIMD.Vec` lane width `W`, counted in **reals**.
"""
lanewidth(::PlanarKernel{MR, NR, T, W}) where {MR, NR, T, W} = W

"""
    avecs_per_column(kernel::PlanarKernel) -> Int

`mr(kernel) ÷ lanewidth(kernel)`: A vectors (and accumulator vectors) per
output column **per plane**. The accumulator holds twice this many per column,
one set for each of the real and imaginary planes.
"""
avecs_per_column(::PlanarKernel{MR, NR, T, W}) where {MR, NR, T, W} = MR ÷ W

"""
    planar_register_pressure(kernel::PlanarKernel) -> Int

Vector registers live at the bottom of the K loop:

    2*MV*NR accumulators + 2*MV A vectors + 2 B broadcasts

with `MV = mr(kernel) ÷ lanewidth(kernel)`. **Cliff A**: this must be `<=` the
architectural register count (`target_profile().nregisters`; 32 zmm under
AVX-512, 16 ymm under AVX2) or the microkernel spills, costing 30-50%.

**The `<= nregisters` bound is optimistic, and is not a predictor.** At the
reference shape `(MV, NR) = (2, 6)` it is `24 + 4 + 2 = 30` and that shape
*does* spill (mostly store-port traffic rather than a load-use chain), while
`(24,3,8)` at pressure 26 is clean and `(8,8,8)` at pressure 20 is not -- so
spilling is not monotone in this number and aspect ratio matters
independently. Treat this as a necessary condition, never a ranking; shapes
are ranked on measured throughput or not at all.
"""
planar_register_pressure(::PlanarKernel{MR, NR, T, W}) where {MR, NR, T, W} =
    2 * (MR ÷ W) * NR + 2 * (MR ÷ W) + 2

# ----------------------------------------------------------------------------
# zero_accumulator, accumulate
# ----------------------------------------------------------------------------

"""
    zero_accumulator(kernel::PlanarKernel{MR,NR,T,W}) -> NTuple{2NV,SIMD.Vec{W,real(T)}}

A logical `MR`-by-`NR` complex zero accumulator, held as one **flat** immutable
tuple of `2NV = 2*(MR÷W)*NR` zero `Vec{W,real(T)}` values: the real plane at
tuple indices `1:NV` and the imaginary plane at `NV+1:2NV`, each using the real
path's `(v, j) -> v + (MR÷W)*j + 1` convention. Physical row `i` of vector `v`
is lane `i - v*W + 1` (1-based `SIMD.Vec` indexing).

Flat rather than nested: it keeps every signature the same *shape* as the real
path, which is the pattern known to keep the accumulator register-resident.
"""
function zero_accumulator(kernel::PlanarKernel{MR, NR, T, W}) where {MR, NR, T, W}
    z = zero(Vec{W, real(T)})
    return ntuple(_ -> z, Val(2 * (MR ÷ W) * NR))
end

# The complex counterpart of `_b_step_load` (src/microkernels/simd.jl): B's
# column `j` at K step `p` as its `(re, im)` pair of reals. The default reads
# the two planes of a `PlanarFormat` panel; `UnpackedBView`
# (src/execution/unpackedb.jl) reads one `Complex` element of B's own storage
# instead. Shared by the planar and fmaddsub kernels, both of which take B as
# planar scalars.
@inline _b_step_load2(packed_b::PB, kernel, j::Int, p::Int) where {PB} = (
    panel_load(packed_b, packed_b_plane_offset(kernel, 0, j, p)),
    panel_load(packed_b, packed_b_plane_offset(kernel, 1, j, p)),
)

# Fully unrolled, closure-free K-step body, mirroring `_accumulate_step` in
# src/microkernels/simd.jl: per logical K step, MV A vector loads per plane, NR B
# scalar loads per plane, 4*MV*NR FMAs, literal tuple indices throughout
# (Cliff B).
#
# GUARDRAIL: the real part is `muladd(-ai, bi, muladd(ar, br, c))`. `c - ai*bi`
# is REJECTED and must not be reintroduced: Julia does not set LLVM's
# `contract` fast-math flag by default, so that form does NOT fuse -- two
# instructions, and different rounding from the other three terms.
#
# `nai_v = -ai_v` is hoisted out of the `j` loop so a declined fold would cost
# MV extra ops per K step rather than MV*NR. Verified by `@code_native` at
# every menu shape: exactly `MV*NR` `vfnmadd231` + `3*MV*NR` `vfmadd231` and
# ZERO separate negations, so the hoist is free rather than merely cheap.
@generated function _accumulate_step_planar(
        kernel::PlanarKernel{MR, NR, T, W}, acc::NTuple{NA, Vec{W, R}},
        packed_a::PA, packed_b::PB, p::Int
    ) where {MR, NR, T, W, R, NA, PA, PB}
    R === real(T) ||
        throw(
        ArgumentError(
            "_accumulate_step_planar: accumulator lane type $R does not match " *
                "real($T) = $(real(T))"
        )
    )
    MV = MR ÷ W
    NV = MV * NR
    2NV == NA ||
        throw(
        ArgumentError(
            "_accumulate_step_planar: accumulator length $NA does not match " *
                "2*(mr÷W)*nr = $(2NV) for MR=$MR, NR=$NR, W=$W"
        )
    )

    arv = [Symbol(:ar, v) for v in 0:(MV - 1)]
    aiv = [Symbol(:ai, v) for v in 0:(MV - 1)]
    naiv = [Symbol(:nai, v) for v in 0:(MV - 1)]
    brv = [Symbol(:br, j) for j in 0:(NR - 1)]
    biv = [Symbol(:bi, j) for j in 0:(NR - 1)]

    load_a = Any[]
    for v in 0:(MV - 1)
        push!(
            load_a,
            :(
                $(arv[v + 1]) = panel_vload(
                    Vec{$W, $R}, packed_a, packed_a_plane_offset(kernel, 0, $(v * W), p)
                )
            )
        )
        push!(
            load_a,
            :(
                $(aiv[v + 1]) = panel_vload(
                    Vec{$W, $R}, packed_a, packed_a_plane_offset(kernel, 1, $(v * W), p)
                )
            )
        )
        push!(load_a, :($(naiv[v + 1]) = -$(aiv[v + 1])))
    end

    load_b = Any[]
    for j in 0:(NR - 1)
        push!(load_b, :(($(brv[j + 1]), $(biv[j + 1])) = _b_step_load2(packed_b, kernel, $j, p)))
    end

    acc_exprs = Vector{Any}(undef, NA)
    for j in 0:(NR - 1), v in 0:(MV - 1)
        idx = v + MV * j + 1
        # re: ar*br - ai*bi, as two chained fused ops with a negated operand.
        acc_exprs[idx] = :(
            muladd(
                $(naiv[v + 1]), $(biv[j + 1]),
                muladd($(arv[v + 1]), $(brv[j + 1]), acc[$idx])
            )
        )
        # im: ar*bi + ai*br.
        acc_exprs[NV + idx] = :(
            muladd(
                $(aiv[v + 1]), $(brv[j + 1]),
                muladd($(arv[v + 1]), $(biv[j + 1]), acc[$(NV + idx)])
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
    accumulate(kernel::PlanarKernel, acc, packed_a, packed_b, kc::Int) -> acc

Planar counterpart of `SIMDKernel`'s `accumulate`. `kc` is the **logical**
(complex) K depth; `kc == 0` returns `acc` unchanged without reading the packed
buffers. Both panels must be in [`PlanarFormat`](@ref). Not bitwise identical
to a scalar complex dot product (FMA grouping differs) — compare with a
tolerance, never `==`.

`packed_a`/`packed_b` may be a [`PackedPanel`](@ref) of `real(T)` — what the
driver passes, and the only form that keeps a large accumulator
register-resident — or any contiguous `AbstractVector{real(T)}`.
"""
function Base.accumulate(
        kernel::PlanarKernel{MR, NR, T, W}, acc::NTuple{NA, Vec{W, R}},
        packed_a::PA, packed_b::PB, kc::Int
    ) where {MR, NR, T, W, R, NA, PA, PB}
    kc == 0 && return acc
    kc > 0 || _throw_negative_kc(:accumulate, kc)
    @inbounds for p in 0:(kc - 1)
        acc = _accumulate_step_planar(kernel, acc, packed_a, packed_b, p)
    end
    return acc
end

# ----------------------------------------------------------------------------
# store_tile!
# ----------------------------------------------------------------------------

# Scattered/scalar store, `@generated` so every `acc[...]` is a compile-time
# index (Cliff B; see src/microkernels/simd.jl's `_store_tile_scattered!`).
# Only the lane index inside a single `Vec` is a runtime value.
#
# The fused form is kept deliberately: `store_tile!` is already specialised per
# `(QSTile{S,R,C}, kernel)` by dispatch, so a kernel-writes-a-stack-tile /
# separate-writeback split would buy nothing and would add a memory round-trip
# costing ~4x.
#
# This is the FALLBACK and the reference: it serves every destination
# `_store_tile_planar_vector!` below is ineligible for, and is what that fast
# path is tested against (test/microkernels/test_planar_store_fastpath.jl).
@generated function _store_tile_planar!(
        destination::QSTile, acc::NTuple{NA, Vec{W, R}},
        alpha::T, beta::T, kernel::PlanarKernel{MR, NR, T, W},
        m::Int, n::Int
    ) where {MR, NR, T, W, R, NA}
    R === real(T) ||
        throw(
        ArgumentError(
            "_store_tile_planar!: accumulator lane type $R does not match " *
                "real($T) = $(real(T))"
        )
    )
    MV = MR ÷ W
    NV = MV * NR
    2NV == NA ||
        throw(
        ArgumentError(
            "_store_tile_planar!: accumulator length $NA does not match " *
                "2*(mr÷W)*nr = $(2NV) for MR=$MR, NR=$NR, W=$W"
        )
    )

    blocks = Any[]
    for j in 0:(NR - 1), v in 0:(MV - 1)
        idx = v + MV * j + 1
        push!(
            blocks, quote
                if $j < n
                    revec = acc[$idx]
                    imvec = acc[$(NV + idx)]
                    for lane in 1:$W
                        i = $(v * W) + lane - 1
                        i < m || break
                        _axpby_tile!(
                            destination, i, $j, alpha,
                            Complex(revec[lane], imvec[lane]), beta
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
# Vectorized unit-stride store fast path
#
# The mirror image of the complex PACK fast path (src/packing/pack_contiguous.jl):
# packing deinterleaves `Complex{T}`'s native `[re,im,re,im,...]` layout into
# two planes, and this interleaves two planes back into it. Unlike packing,
# this also has to do real arithmetic -- `alpha*r + beta*C_old` with COMPLEX
# `alpha`/`beta`, which mixes the two planes -- so the whole of the difficulty
# is in reproducing that arithmetic, not in the data movement.
#
# ARITHMETIC: this reproduces the expression tree Base's own `Complex`
# arithmetic specifies -- the same tree `_axpby_tile!` reaches through --
# operand for operand, with the scalars broadcast to `Vec`. It deliberately
# does NOT reuse `_accumulate_step_planar`'s four-real-FMA grouping: that is
# *a* correct grouping but a different one, and there is no reason to accept a
# second grouping when the scalar path's own is expressible verbatim in lanes.
#
# The three `beta` regimes, identical to `_axpby_tile!`'s:
#
#   beta == 0   alpha * r
#               = Complex(ar*rr - ai*ri, ar*ri + ai*rr)        (Base `*`,
#                 four UNFUSED multiplies; Julia does not set LLVM's `contract`
#                 flag, so `ar*rr - ai*ri` stays two `vmul` + one `vsub` in
#                 both the scalar and the vector form -- the same fact the
#                 `_accumulate_step_planar` guardrail above relies on, used
#                 here to preserve rounding rather than to avoid losing it.)
#   beta == 1   muladd(alpha, r, C_old)
#   otherwise   muladd(alpha, r, beta * C_old)
#
#               with Base's `muladd(z,w,x)` =
#                 Complex(muladd(zr, wr, -muladd(zi, wi, -xr)),
#                         muladd(zr, wi, muladd(zi, wr, xi)))
#
# HOW CLOSE THE TWO PATHS AGREE (over every shipped shape, both dtypes and
# seven alpha/beta regimes; test/microkernels/test_planar_store_fastpath.jl
# checks this):
#
#   * Every VECTORIZED FULL BLOCK is bit-exact to the tree above -- zero
#     misses against an independently written, optimization-barriered
#     transcription of it, compared with `isequal` so `-0.0` and NaN payloads
#     count. `SIMD.Vec`'s `*`/`-` carry no LLVM `contract` flag, so the lanes
#     round exactly where the source text says they do.
#   * Against `_store_tile_planar!` the agreement is exact at `beta == 0` and
#     `beta == 1`, and ~1 ULP in the general-`beta` regime. The variability is
#     on the SCALAR side, not this one: LLVM SLP-vectorizes Base's
#     `muladd(::Complex, ::Complex, ::Complex)` into `<2 x double>` ops and
#     marks them `contract` (plainly visible in `@code_llvm` of
#     `muladd(a, r, b*c)` at `ComplexF64`: `fmul contract` / `fsub contract`),
#     letting the backend fuse a multiply the source text rounds separately.
#     Whether it fires depends on inlining context, so the scalar path is not
#     bit-reproducible even against ITSELF across call sites: a few elements
#     differ between this function's own scalar row tail and
#     `_store_tile_planar!`, running character-identical source. Requiring the
#     fast path to match it bitwise would be requiring it to match an LLVM
#     heuristic; compare with a tolerance, never `==`.
#
# The `beta` regimes are the SAME three `_axpby_tile!` has, chosen by the same
# `iszero`/`isone` tests on the same per-call scalar: the beta-applied-once
# contract (src/microkernels/interface.jl's `_store_prologue!`/`_axpby_tile!`) is untouched,
# and in particular `beta == 0` never loads the destination here either.
#
# NOT an `unsafe_` path: it skips no validation a caller would otherwise get.
# `_store_prologue!` runs first and unchanged, the `(v+1)*W <= m` test keeps
# every straddling row-vector on the scalar tail exactly as the real path's
# `_store_tile_vector!` does, and `j < n` keeps padding columns unwritten -- so
# the set of addresses touched is exactly the set `_store_tile_planar!` would
# have touched.
# ----------------------------------------------------------------------------

"""
    _complex_vector_eligible(tile::QSTile, ::Type{T}) -> Bool

Whether `tile` can take a vectorized complex store/load: unit-stride
`AffineAxis` rows into rank-1 dense `Complex` storage, on an ISA this ships
for. The complex counterpart of [`_vector_store_eligible`](@ref)
(src/microkernels/simd.jl) and deliberately the same shape question that
`_pack_complex_contiguous_eligible` (src/packing/pack_contiguous.jl) asks of a
source tile, sharing the ISA half literally via `_complex_fastpath_isa_eligible`
(src/hardware/target.jl).

Unit stride plus rank-1 dense storage is what makes `reinterpret`ing the
storage pointer from `Ptr{Complex{R}}` to `Ptr{R}` a sound bitcast: `W`
consecutive rows are then `W` consecutive `Complex{R}` values and hence `2W`
consecutive `R`s. It is unsound for a `ScatterAxis`/`PtrScatterAxis` row, for
a strided `AffineAxis`, and for non-`DenseArray` storage, and all three are
excluded here.

The storage clause inspects types only, so at each specialization the whole
predicate folds to `_unit_stride_rows(tile.rows) && <isa check>` or to `false`.
"""
@inline _complex_vector_eligible(tile::QSTile, ::Type{T}) where {T} =
    _unit_stride_rows(tile.rows) && tile.storage isa DenseVector{T} &&
    _complex_fastpath_isa_eligible()

# --- interleave / deinterleave ---------------------------------------------
#
# GUARDRAIL, the same one src/packing/pack_contiguous.jl's shuffle primitives
# carry: every index tuple is built HERE from `W` at specialization time, never
# hardcoded to one ISA's shape. `W` is `lanewidth(kernel)`, which the driver
# derives from `kernel_shapes`, so the patterns follow the shipped menus
# automatically. `@generated` because `shufflevector` needs a literal `Val`
# index tuple.
#
# Lane order: `v` is `[re_0, im_0, ..., re_{W-1}, im_{W-1}]`, i.e. `W`
# consecutive `Complex{R}` values read through their native binary layout.

# Even lanes: the real parts.
@generated function _deinterleave_re(v::Vec{N, R}, ::Val{W}) where {N, R, W}
    N == 2 * W || return :(throw(ArgumentError("_deinterleave_re: expected N == 2W")))
    idx = ntuple(k -> 2 * (k - 1), W)
    return :(shufflevector(v, Val($idx)))
end

# Odd lanes: the imaginary parts.
@generated function _deinterleave_im(v::Vec{N, R}, ::Val{W}) where {N, R, W}
    N == 2 * W || return :(throw(ArgumentError("_deinterleave_im: expected N == 2W")))
    idx = ntuple(k -> 2 * (k - 1) + 1, W)
    return :(shufflevector(v, Val($idx)))
end

# The inverse: two planes back into `Complex{R}`'s layout. Lane `2t` takes
# `re[t]` (first source), lane `2t+1` takes `im[t]` (second source, offset by
# `W` in the index space `shufflevector` uses for a two-operand shuffle).
@generated function _interleave_planes(re::Vec{W, R}, im::Vec{W, R}, ::Val{W}) where {W, R}
    idx = ntuple(k -> isodd(k) ? (k - 1) ÷ 2 : W + (k - 1) ÷ 2, 2 * W)
    return :(shufflevector(re, im, Val($idx)))
end

# --- the per-block arithmetic ----------------------------------------------
#
# One full `W`-row block of one column: read `W` old `Complex{R}` values (only
# when `beta != 0`), compute `alpha*r + beta*C_old` in split planes, interleave,
# write them back. `at` is the ZERO-based index of the block's first REAL in
# the reinterpreted storage, i.e. twice the complex element index.
#
# `ar`/`ai`/`br`/`bi` are pre-broadcast once per `store_tile!` call rather than
# per block; `beta` itself is passed only so the two `iszero`/`isone` tests are
# the identical tests `_axpby_tile!` makes on the identical value.
@inline function _planar_store_block!(
        sp::Ptr{R}, at::Int, rev::Vec{W, R}, imv::Vec{W, R},
        ar::Vec{W, R}, ai::Vec{W, R}, br::Vec{W, R}, bi::Vec{W, R},
        beta::Complex{R}, ::Val{W}
    ) where {R, W}
    if iszero(beta)
        # Base `*`: unfused, and deliberately so (see the section header).
        newre = ar * rev - ai * imv
        newim = ar * imv + ai * rev
    else
        old = vload(Vec{2 * W, R}, sp + sizeof(R) * at)
        orv = _deinterleave_re(old, Val(W))
        oiv = _deinterleave_im(old, Val(W))
        if isone(beta)
            xr, xi = orv, oiv
        else
            # Base `*` again, on `beta * C_old`.
            xr = br * orv - bi * oiv
            xi = br * oiv + bi * orv
        end
        # Base `muladd(::Complex, ::Complex, ::Complex)`, transcribed.
        newre = muladd(ar, rev, -muladd(ai, imv, -xr))
        newim = muladd(ar, imv, muladd(ai, rev, xi))
    end
    vstore(_interleave_planes(newre, newim, Val(W)), sp + sizeof(R) * at)
    return nothing
end

# Vectorized planar store. Structurally the same `@generated` unroll as the
# real path's `_store_tile_vector!` (src/microkernels/simd.jl) -- same `(v, j)`
# unrolling for Cliff B (every `acc[...]` a literal tuple index), same
# full-block vs. row-tail split, same `j < n` column guard -- differing only in
# that a "row block" is `2W` reals rather than `W`, and that the arithmetic is
# complex.
#
# `rows::AffineAxis` is pinned in the signature so an ineligible tile is a
# MethodError rather than a wrong answer; the caller checks
# `_complex_vector_eligible` first.
#
# GC.@preserve: the raw `Ptr{R}` is derived from
# `storage` and every dereference of it happens inside the preserve block, the
# same discipline `_pack_a_contiguous!`/`_pack_complex_contiguous!` already
# follow. The scalar row tail goes through `storage` itself (via `_axpby_at!`),
# not the pointer: simpler, and character-identical to the fallback's own tail
# arithmetic -- though not therefore bit-identical to it, see the section
# header on LLVM's context-dependent `contract` flag.
@generated function _store_tile_planar_vector!(
        destination::QSTile{S, <:AffineAxis}, acc::NTuple{NA, Vec{W, R}},
        alpha::T, beta::T, kernel::PlanarKernel{MR, NR, T, W},
        m::Int, n::Int
    ) where {S, MR, NR, T, W, R, NA}
    R === real(T) ||
        throw(
        ArgumentError(
            "_store_tile_planar_vector!: accumulator lane type $R does not match " *
                "real($T) = $(real(T))"
        )
    )
    # The bitcast below is only sound on rank-1 dense storage of exactly `T`.
    # Asserted at specialization time so an ineligible tile that somehow
    # reached here is a hard error, not a wrong answer.
    S <: DenseVector{T} ||
        throw(
        ArgumentError(
            "_store_tile_planar_vector!: destination storage $S is not a DenseVector{$T}"
        )
    )
    MV = MR ÷ W
    NV = MV * NR
    2NV == NA ||
        throw(
        ArgumentError(
            "_store_tile_planar_vector!: accumulator length $NA does not match " *
                "2*(mr÷W)*nr = $(2NV) for MR=$MR, NR=$NR, W=$W"
        )
    )

    blocks = Any[]
    for j in 0:(NR - 1)
        vblocks = Any[]
        for v in 0:(MV - 1)
            idx = v + MV * j + 1
            push!(
                vblocks, quote
                    revec = acc[$idx]
                    imvec = acc[$(NV + idx)]
                    if $((v + 1) * W) <= m
                        # Rows v*W .. v*W+W-1 are all inside [0, m): one
                        # 2W-real load/store at that block's first real.
                        _planar_store_block!(
                            sp, 2 * (colbase + $(v * W)), revec, imvec,
                            ar, ai, br, bi, beta, Val($W)
                        )
                    elseif $(v * W) < m
                        # Row tail: this vector straddles m, so store the valid
                        # lanes one at a time, through the SAME `_axpby_at!`
                        # the real path's tail uses -- bitwise the scalar path.
                        for lane in 1:$W
                            i = $(v * W) + lane - 1
                            i < m || break
                            _axpby_at!(
                                storage, colbase + i + 1, alpha,
                                Complex(revec[lane], imvec[lane]), beta
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
        Base.@_inline_meta
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
    store_tile!(destination::QSTile, acc::NTuple{2NV,SIMD.Vec{W,real(T)}}, alpha::T, beta::T, kernel::PlanarKernel) -> destination

Planar counterpart of `SIMDKernel`'s `store_tile!`, with the **same contract**:
`C[i,j] = alpha*R[i,j] + beta*C[i,j]` over the valid rectangle only;
`alpha == 0` never reads `acc`; `beta == 0` never reads old `C`; `beta == 1`
skips the multiplication; padding lanes (`i >= m`, `j >= n`) are never read, so
nonfinite padding in `acc` cannot propagate; an empty destination is a no-op in
every branch.

Two paths, chosen by [`_complex_vector_eligible`](@ref):

  * Fast path, `_store_tile_planar_vector!`: unit-stride `AffineAxis` rows into
    rank-1 dense `Complex` storage on a shipped ISA. Whole `W`-row blocks get
    one `2W`-real load (only when `beta != 0`), a plane-wise complex multiply-
    add, an interleave shuffle and one `2W`-real store; a straddling row-vector
    falls to the scalar tail.
  * Fallback, `_store_tile_planar!`: everything else -- scattered or strided
    rows, non-dense storage, an un-shipped ISA. Recombines
    `Complex(re[lane], im[lane])` and delegates to the generic `_axpby_tile!`
    (src/microkernels/interface.jl), which provides the `beta` shortcuts.

The fast path is bitwise identical (`isequal`, so `-0.0`/NaN payloads count)
to a from-scratch, optimization-barriered transcription of the expression
tree documented above this function's implementation. Against the fallback
specifically, it is exact at `beta == 0` and `beta == 1`, and within ~1 ULP
in the general-`beta` regime -- the fallback itself is not bit-reproducible
against ITSELF across call sites there (LLVM inconsistently fuses Base's
`muladd(::Complex,::Complex,::Complex)`; see the divergence measurement
above). Compare the general-`beta` case with a tolerance, never `==`.

`@inline`, with the vectorized store body, for the reason given at the real
`store_tile!` (src/microkernels/simd.jl): an out-of-line store spills the
whole two-plane accumulator to the stack on every micro-tile. The scalar
fallback stays out of line, as the real path's does.
"""
@inline function store_tile!(
        destination::QSTile, acc::NTuple{NA, Vec{W, R}},
        alpha::T, beta::T, kernel::PlanarKernel{MR, NR, T, W}
    ) where {MR, NR, T, W, R, NA}
    m, n = _store_prologue!(destination, alpha, beta)
    (m == 0 || n == 0) && return destination

    if _complex_vector_eligible(destination, T)
        return _store_tile_planar_vector!(destination, acc, alpha, beta, kernel, m, n)
    end

    return _store_tile_planar!(destination, acc, alpha, beta, kernel, m, n)
end
