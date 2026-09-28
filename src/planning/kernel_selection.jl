# Kernel selection: which microkernel `plan_contract` builds when the caller
# does not name one.
#
# Everything is keyed on (element type `T`, method), where the method is
# `RealMethod()` for a real `T` and `PlanarMethod()` for a complex one
# (`_default_method`); `OneMMethod()` is never chosen here, only by naming the
# kernel. Per method there is
#
#   * a closed menu of `(MR, NR, W)` shapes (`kernel_shapes`), so the set of
#     compiled kernel and driver specializations stays bounded;
#   * a kernel type (`_kernel_type`), built from a menu shape by
#     `_kernel_from_shape`;
#   * a shape resolved from the detected hardware (`_derived_shape`): an
#     explicit override row, else the `MR = MV*W, NR = 6` rule where it has
#     been validated (`MV = 4` for the real method on Intel `:avx512`, `2`
#     elsewhere; `_rule_mv` / `_profile_mv`), else a conservative fitted
#     shape (`_fitted_shape`);
#   * four plan-time demotions: the real `MV = 4` shape steps down to its
#     `MV = 2` sibling when M is short enough that the taller tile pads more
#     (`_extent_shape`), and when C's unit-stride run along M would break
#     the tall slivers' vectorized store (`_store_shape`); to the fitted
#     shape when M cannot fill one register tile (`_default_shape` /
#     `_default_kernel`, src/planning/defaults.jl;
#     complex on `:avx512` goes to an FMAddSub shape instead,
#     `_small_m_shape`); and to a menu shape that keeps every register sliver
#     unit-stride in C (`_demote_for_run`).
#
# Everything here is a pure function of a `TargetProfile` and `T`. The
# engine's per-call entry points (`_default_kernel`, `default_blocking(kernel)`)
# live in src/planning/defaults.jl, which resolves these once per (profile,
# `T`) and caches the result: the `Val(profile.isa)` dispatch below is a
# dynamic one (the ISA is a runtime `Symbol`), too slow to repeat on every
# `plan_contract`.

# ----------------------------------------------------------------------------
# Methods, menus and kernel types
# ----------------------------------------------------------------------------

# The method the engine uses for `T`; `OneMMethod` is selected only by naming
# the kernel, since method ranking does not transfer between machines.
_default_method(::Type{<:Real}) = RealMethod()
_default_method(::Type{<:Complex}) = PlanarMethod()

const NR_DEFAULT = 6

# The conservative shape: a `(8, 6)` register tile in logical rows, with the
# lane width of one 256-bit register of `real(T)`. It is the default wherever
# no rule applies, and in every real menu.
_fallback_shape(::Type{T}) where {T} = (8, NR_DEFAULT, _default_lanewidth(real(T)))

# Menus. Complex `MR` counts logical (complex) rows and `W` real lanes. The 1m
# menus look "unaligned" (MR = 12 at W = 8) only because 1m runs a real
# microkernel of `2MR` rows, so it is `2MR` that must be a multiple of `W`.
# Each planar menu starts with the shape `_derived_shape` resolves to on
# `:avx512`; each real menu contains `_fallback_shape` and the AVX-512/AVX2
# rule shapes, and each 1m menu contains its AVX-512 rule shape.
#
# The last three entries of each planar menu are an `MV = 1` tile at each lane
# width the package compiles, so that `_fitted_shape` finds a fitting entry for
# any (lane count, register budget >= 16) pair. They are unmeasured and are not
# claimed to be good, only to fit.
#
# 2026-09-25: each 1m menu's LAST entry is its AVX2-native ("MR = 2W") rule
# shape, added because both menus previously held AVX-512-only (W=8/W=16)
# shapes -- 1m had never been measured on AVX2 hardware at a correctly-sized
# shape. Derived the same way as the AVX-512 entry (real-MR/2 at the real
# kernel's own AVX2 rule shape): `KERNEL_SHAPES_F64`'s AVX2 shape is
# `(8,6,4)` -> 1m `(4,6,4)`; `KERNEL_SHAPES_F32`'s AVX2 shape is `(16,6,8)` ->
# 1m `(8,6,8)`. Measured, job 7107477: `(4,6,4)` is the new ComplexF64 AVX2
# winner outright (see the AVX2 override block below); `(8,6,8)` is
# ComplexF32's best 1m shape but does not beat planar there, so it stays
# menu-only (reachable via an explicit `kernel = OneMMethod()`, never picked
# automatically).
#
# 2026-09-26: the `MV = 4` shapes `(32, 6, 8)` / `(64, 6, 16)` are the real
# AVX-512 rule shapes (see `_rule_mv`); the `MV = 2` shapes stay in the menus
# as their short-M step-down (`_extent_shape`) and as `_demote_for_run`
# targets.
const KERNEL_SHAPES_F64 = ((8, 6, 4), (16, 6, 8), (32, 6, 8))
const KERNEL_SHAPES_F32 = ((8, 6, 8), (32, 6, 16), (16, 6, 8), (64, 6, 16))
const KERNEL_SHAPES_C64_PLANAR = (
    (24, 3, 8), (16, 6, 8), (8, 8, 8), (4, 5, 4), (4, 6, 2), (2, 6, 2),
)
const KERNEL_SHAPES_C64_ONEM = ((12, 8, 8), (16, 6, 8), (8, 8, 8), (4, 6, 4))
const KERNEL_SHAPES_C32_PLANAR = (
    (48, 3, 16), (32, 6, 16), (16, 8, 16), (8, 5, 8), (8, 6, 4), (4, 6, 4),
)
const KERNEL_SHAPES_C32_ONEM = ((24, 8, 16), (32, 6, 16), (16, 8, 16), (8, 6, 8))

# FMAddSub (src/microkernels/fmaddsub.jl): not the default method, like 1m;
# auto-selected only by the AVX-512 small-M demotion (`_small_m_shape`).
# Its accumulator has 1m's layout and count, so the menus START from 1m's
# shapes (a same-shape head-to-head isolates the A-format/swap trade), plus the
# AVX2 `NR = 5` sibling of 1m's AVX2 rule shape, which fits AVX2's 16
# registers by `fmaddsub_register_pressure` (16) where `NR = 6` (18) does not:
# `-C znver2` codegen of `(4,6,4)`/`(8,6,8)` spills (25 stores / 22 reloads per
# K step, Julia 1.12.7) and `(4,5,4)`/`(8,5,8)` is clean
# (benchmark/probes/fmaddsub_codegen.jl, 2026-09-25). `NR = 6` stays for the
# same-shape head-to-head with 1m.
#
# Measured 2026-09-25, `benchmark/submit_fmaddsub.sh` (bench_complex_efficiency.jl
# arm 2, Julia 1.12.7, 21 reps, geomean GF/s over its 8 cases):
#
#   AVX2, Rome/znver2, job 7109276 (canary spread 1.27%): fmaddsub `4x5/W4` is
#   the ComplexF64 winner across all three methods, 32.36 GF/s against 1m
#   `4x6/W4` 31.10 (+4.1%) and planar `4x5/W4` 31.01 (+4.4%). Against planar it
#   wins every one of the 8 cases (+1% to +10%); against 1m it is mixed (+7..11%
#   at 64^3/shallow-K/small-N, -2% at 256^3/512^3/small-M). Same shape as 1m,
#   `4x6/W4`, it LOSES (23.93 GF/s) to the Cliff A spill above. ComplexF32: no
#   win -- fmaddsub `8x5/W8` 51.97 ties 1m `8x6/W8` 52.02 and trails planar
#   `8x5/W8` 54.95 (-5.4%), the unchanged champion.
#
#   AVX-512, Ice Lake-SP/icelake-server, job 7109277 (canary spread 0.95%): at
#   1m's own ComplexF64 shapes fmaddsub beats 1m by +14% (`12x8/W8`, 51.19 vs
#   44.88) and +22% (`8x8/W8`, 50.01 vs 40.87), every case >= +1% -- half the
#   packed-A bytes paying for the swap -- but the overall winner stays planar
#   `24x3/W8` (55.50; fmaddsub's best is 7.8% behind). ComplexF32: fmaddsub
#   `16x8/W16` 87.27 ties 1m `16x8/W16` 87.01; planar `48x3/W16` 93.55 wins.
#
# So: a ~4% AVX2 ComplexF64 win, and nothing elsewhere. Too small, and on one
# machine, to justify an auto-dispatch rule (see `ComplexMethod`'s docstring);
# the menus stay reachable by naming `FMAddSubKernel`, and no override row is
# added. The small-M case (2x, not 4%) is the exception: `_small_m_shape`.
const KERNEL_SHAPES_C64_FMADDSUB = ((12, 8, 8), (8, 8, 8), (4, 6, 4), (4, 5, 4))
const KERNEL_SHAPES_C32_FMADDSUB = ((24, 8, 16), (16, 8, 16), (8, 6, 8), (8, 5, 8))

"""
    kernel_shapes(T, method::ComplexMethod = _default_method(T)) -> NTuple{<:Any,NTuple{3,Int}}

The closed menu of `(MR, NR, W)` register shapes the engine may build for
element type `T` under `method`. Each method has its own menu because each has
its own packed A format and therefore its own register budget.
"""
kernel_shapes(::Type{T}) where {T} = kernel_shapes(T, _default_method(T))
kernel_shapes(::Type{Float64}, ::RealMethod) = KERNEL_SHAPES_F64
kernel_shapes(::Type{Float32}, ::RealMethod) = KERNEL_SHAPES_F32
kernel_shapes(::Type{ComplexF64}, ::PlanarMethod) = KERNEL_SHAPES_C64_PLANAR
kernel_shapes(::Type{ComplexF64}, ::OneMMethod) = KERNEL_SHAPES_C64_ONEM
kernel_shapes(::Type{ComplexF32}, ::PlanarMethod) = KERNEL_SHAPES_C32_PLANAR
kernel_shapes(::Type{ComplexF32}, ::OneMMethod) = KERNEL_SHAPES_C32_ONEM
kernel_shapes(::Type{ComplexF64}, ::FMAddSubMethod) = KERNEL_SHAPES_C64_FMADDSUB
kernel_shapes(::Type{ComplexF32}, ::FMAddSubMethod) = KERNEL_SHAPES_C32_FMADDSUB

# The kernel type implementing `method` for `T`, or `nothing` when there is none
# (exactly the pairs `kernel_shapes` has a menu for).
_kernel_type(::RealMethod, ::Type{<:Union{Float32, Float64}}) = SIMDKernel
_kernel_type(::PlanarMethod, ::Type{<:Union{ComplexF32, ComplexF64}}) = PlanarKernel
_kernel_type(::OneMMethod, ::Type{<:Union{ComplexF32, ComplexF64}}) = OneMKernel
_kernel_type(::FMAddSubMethod, ::Type{<:Union{ComplexF32, ComplexF64}}) = FMAddSubKernel
_kernel_type(::Any, ::Type) = nothing

"""
    _kernel_from_shape(shape, T, method = _default_method(T)) -> kernel

Build the `method` kernel for `T` at `shape`, which must be in
`kernel_shapes(T, method)`. Unrolled over the menu so that every branch builds
a concrete kernel from literal `Val`s (a plain loop would construct
`Val(shape[1])` dynamically and widen to `Any`); this costs one dynamic
dispatch per `plan_contract`, none per tile or K step. A shape outside the
menu, or a method with no kernel for `T`, throws rather than silently building
a different kernel than was asked for.
"""
_kernel_from_shape(shape::Tuple{Int, Int, Int}, ::Type{T}) where {T} =
    _kernel_from_shape(shape, T, _default_method(T))

@generated function _kernel_from_shape(
        shape::Tuple{Int, Int, Int}, ::Type{T}, method::M
    ) where {T, M}
    K = _kernel_type(M.instance, T)
    K === nothing && return :(_throw_no_kernel(shape, T, method))
    ex = :(_throw_shape_not_in_menu(shape, T, method))
    for (MR, NR, W) in reverse(kernel_shapes(T, M.instance))
        ex = :(shape === ($MR, $NR, $W) ? $K(Val($MR), Val($NR), T, Val($W)) : $ex)
    end
    return ex
end

"""
    _menu_val(shape, T, method) -> Val{shape}()

`Val(shape)` for a `shape` in `kernel_shapes(T, method)`, built as a literal
in an unrolled ladder over the menu (a plain `Val(shape)` would construct the
type at runtime), with the same two throws as `_kernel_from_shape`. The
result is a singleton, so it crosses `plan_contract`'s kernel barrier
(`_plan_with_kernel`, src/planning/plan.jl) without boxing, and the callee is
specialised on this one shape; building only menu shapes keeps that set of
specialisations closed.
"""
@generated function _menu_val(shape::Tuple{Int, Int, Int}, ::Type{T}, method::M) where {T, M}
    K = _kernel_type(M.instance, T)
    K === nothing && return :(_throw_no_kernel(shape, T, method))
    ex = :(_throw_shape_not_in_menu(shape, T, method))
    for (MR, NR, W) in reverse(kernel_shapes(T, M.instance))
        ex = :(shape === ($MR, $NR, $W) ? Val(($MR, $NR, $W)) : $ex)
    end
    return ex
end

@noinline _throw_no_kernel(shape, ::Type{T}, method) where {T} = throw(
    ArgumentError(
        "no microkernel is available for $T at shape $shape under $(method). " *
            "Pass an explicit `kernel = ...` to plan_contract to use a kernel this " *
            "engine does not pick itself."
    )
)

@noinline _throw_shape_not_in_menu(shape, ::Type{T}, method) where {T} = throw(
    ArgumentError(
        "shape $shape is not in the $(method) menu for $T, " *
            "$(kernel_shapes(T, method)); only menu shapes are compiled"
    )
)

# ----------------------------------------------------------------------------
# Shape resolution from the detected hardware
# ----------------------------------------------------------------------------

# Explicit per-ISA shapes, consulted first. Planar only. Empty for the real
# types: there the `MR = 2W` rule below is already the measured optimum, so a
# row would only pin the package to one machine's noise. 1m has none: it is
# never selected automatically.
_shape_override(key::Val, ::Type{T}) where {T} = _shape_override(key, T, _default_method(T))
_shape_override(::Val, ::Type, ::ComplexMethod) = nothing

# AVX-512, measured: the derived `MR = 2W, NR = 6` shape is the worst planar
# configuration (by 38-41%), and `24x3`/`48x3` win outright.
_shape_override(::Val{:avx512}, ::Type{ComplexF64}, ::PlanarMethod) = (24, 3, 8)
_shape_override(::Val{:avx512}, ::Type{ComplexF32}, ::PlanarMethod) = (48, 3, 16)

# NEON, measured on an Apple M3 Max by the sibling `tensorcontract-rs` project
# (planar winner `(MV, NR) = (2, 6)` for both precisions). The register-budget
# fit selects the same shapes; the rows pin them so a later menu edit cannot
# move them silently.
_shape_override(::Val{:neon}, ::Type{ComplexF64}, ::PlanarMethod) = (4, 6, 2)
_shape_override(::Val{:neon}, ::Type{ComplexF32}, ::PlanarMethod) = (8, 6, 4)

# AVX2, measured: `benchmark/bench_complex_efficiency.jl` arm 2 on a Rome
# (znver2) node (2026-09-24, job 7102205) ranks every menu shape for both
# planar and 1m; at the time, `(4, 5, 4)` won ComplexF64 outright (next best,
# 1m `8x8/W8`, was 1.397x slower) and `(8, 5, 8)` won ComplexF32 outright
# (next best, 1m `16x8/W16`, was 1.476x slower) -- confirming the
# register-budget reasoning these rows originally shipped with (at `MV = 1`,
# `NR = 6` costs all 16 of AVX2's vector registers (`2*6 + 2 + 2`), leaving
# none for address arithmetic, while `NR = 5` costs 14). Neither 1m entry
# available at the time was AVX2-native (both were AVX-512-sized, `W = 8`),
# so this did not yet measure 1m at a correctly-sized AVX2 shape.
#
# 2026-09-25, job 7107477 (same node class, canary spread 1.3%): with the
# AVX2-native 1m shapes `(4, 6, 4)`/`(8, 6, 8)` added to the menus (see the
# menu comment above), 1m `4x6/W4` WINS ComplexF64 outright, 6.3% faster than
# planar `4x5/W4` (the prior champion; geomean 32.95 vs 30.98 GF/s) --
# 1m sidesteps the planar broadcast:FMA problem at `MV = 1` entirely by
# reusing the real kernel's `accumulate`, which needs no scalar-to-vector
# broadcast at all. ComplexF32 does NOT flip: 1m `8x6/W8` (53.55 GF/s) is
# 4.3% slower than planar `8x5/W8` (54.81 GF/s, still champion) -- so the
# override below is added for ComplexF64 only; ComplexF32 keeps its planar
# override unchanged.
_shape_override(::Val{:avx2}, ::Type{ComplexF64}, ::PlanarMethod) = (4, 5, 4)
_shape_override(::Val{:avx2}, ::Type{ComplexF32}, ::PlanarMethod) = (8, 5, 8)

# AVX2, measured, job 7107477 (2026-09-25, see above): `(4, 6, 4)` is the
# correctly AVX2-sized 1m shape and the outright ComplexF64 AVX2 winner
# across both methods. Pinned here (not left to `_fitted_shape`, which would
# still hand 1m the AVX-512-sized `(12, 8, 8)` on AVX2) so a caller who
# explicitly opts into `kernel = OneMMethod()` gets it. This does not change
# automatic dispatch: `_default_method` still returns `PlanarMethod()` for
# complex `T`, so nothing selects 1m unless the caller names it.
_shape_override(::Val{:avx2}, ::Type{ComplexF64}, ::OneMMethod) = (4, 6, 4)

# Where the `MR = MV*W` rule is validated. Real: AVX-512 and AVX2 (on NEON it
# would pick MR = 4 on 128-bit lanes, not obviously better than the fallback).
# Complex: AVX-512 only -- planar holds separate real and imaginary
# accumulator planes, so on AVX2's 16 registers even `(MV, NR) = (1, 6)`
# leaves nothing spare (Cliff A, src/microkernels/planar.jl).
_rule_applies(::Val{:avx512}, ::RealMethod) = true
_rule_applies(::Val{:avx512}, ::Union{PlanarMethod, OneMMethod}) = true
_rule_applies(::Val{:avx2}, ::RealMethod) = true
_rule_applies(::Val, ::ComplexMethod) = false

# `W` is one vector register's worth of REAL lanes, `MR = MV * W` with `MV`
# A vectors (and accumulator rows) per output column, `NR = NR_DEFAULT`. The
# default `MV = 2` is the rule as it was validated for every method on every
# ISA; `_derived_shape` passes `_rule_mv`.
_rule_shape(vb::Int, ::Type{T}, mv::Int = 2) where {T} =
    (mv * (vb ÷ sizeof(real(T))), NR_DEFAULT, vb ÷ sizeof(real(T)))

# A vectors per column in the rule shape. `4` for the real method on
# `:avx512`, `2` everywhere else.
#
# Why 4 on AVX-512 (measured 2026-09-26, ccqlin038 / Cascade Lake, Julia
# 1.12.7, `perf stat`): the `MV = 2` real kernel `(16, 6, 8)` is FRONT-END
# bound, not FMA bound. Its K step is 2 A loads + 6 `vbroadcastsd` + 12
# register FMAs + 5 loop-overhead instructions = 25 fused-domain uops for 6
# FMA-cycles, i.e. 4.2 uops/cycle against Skylake's 4-wide allocation; it
# runs at 2.98 IPC and 23.8 flops/cycle (74% of the 32-flop peak) even with
# both packed panels L1-resident. Doubling `MV` doubles the FMAs per B
# broadcast and per loop iteration: `(32, 6, 8)` is 38 instructions per 12
# FMA-cycles and measures 29.1 flops/cycle (91%; OpenBLAS dgemm on the same
# core: 28.7). The register budget is what caps `MV` at 4: `MV*NR`
# accumulators + `MV` A vectors + 2 broadcast/scratch = 30 of AVX-512's 32
# registers, while AVX2's 16 hold exactly the `MV = 2` tile (12 + 2 + 2).
# Float32 behaves identically (`(32, 6, 16)` 180 -> `(64, 6, 16)` 218 GF/s,
# kernel-only). Engine, 3969^3 Float64 (same session, interleaved):
# `(16, 6, 8)` 73.8 -> `(32, 6, 8)` 87.1 GF/s, OpenBLAS 105.7.
#
# Complex methods keep `MV = 2`: planar holds two accumulator planes (its
# `(16, 6, 8)` already spills, see `planar_register_pressure`), and 1m runs a
# real kernel of `2 MR` rows over its own menu.
_rule_mv(::Val{:avx512}, ::RealMethod) = 4
_rule_mv(::Val, ::ComplexMethod) = 2

# `_rule_mv` for one host: `MV = 4` also needs a core where the `MV = 2` tile
# is front-end bound, which a double-pumped AVX-512 core is not (see above),
# so AMD's AVX-512 cores keep `MV = 2`. Measured on Genoa (znver4), PR #10
# head (MV = 4) against main (MV = 2), bench_to_suite jobs 7124412 / 7112089,
# QS time ratio: the large GEMMs MV = 4 exists for gain 1-3% at most
# (dim32/dim63_2_2_2, all layouts, 0.97-0.99), while mps_*_D64 (1.69-1.73),
# dim12/dim16_2_2_2_gemm_ready (1.24-1.33) and trg_plaquette_chi16/24
# (1.15-1.27) regress there and improve on Ice Lake in the same comparison
# (0.51-0.97). `znver5` (full-width 512-bit datapaths, unmeasured) keeps the
# pre-MV = 4 shape too: nothing measured says the taller tile pays there.
const _MV4_UNPROFITABLE_CPUS = ("znver4", "znver5")

_profile_mv(profile::TargetProfile, method) = _rule_mv(Val(profile.isa), method)
function _profile_mv(profile::TargetProfile, method::RealMethod)
    mv = _rule_mv(Val(profile.isa), method)
    return (mv == 4 && profile.cpu_name in _MV4_UNPROFITABLE_CPUS) ? 2 : mv
end

"""
    _derived_shape(profile::TargetProfile, T, method = _default_method(T)) -> (MR, NR, W)

The register shape for `T` under `method` on `profile`, in precedence order:
an explicit `_shape_override` row, then `_rule_shape` where `_rule_applies`
and the rule's shape is in the menu, then `_fitted_shape`. Always a member of
`kernel_shapes(T, method)`.
"""
_derived_shape(profile::TargetProfile, ::Type{T}) where {T} =
    _derived_shape(profile, T, _default_method(T))

function _derived_shape(profile::TargetProfile, ::Type{T}, method) where {T}
    key = Val(profile.isa)
    ovr = _shape_override(key, T, method)
    ovr === nothing || return ovr
    vb = profile.vector_bytes
    if _rule_applies(key, method) && vb > 0 && vb % sizeof(real(T)) == 0
        shape = _rule_shape(vb, T, _profile_mv(profile, method))
        shape in kernel_shapes(T, method) && return shape
    end
    return _fitted_shape(profile, T, method)
end

"""
    _planar_pressure(MR, NR, W) -> Int

Vector registers a planar kernel holds live per K step: `2*MV*NR`
accumulators (two planes) + `2*MV` A vectors (two planes) + 2 B broadcasts,
with `MV = MR ÷ W`.

A *necessary* condition only, not a predictor: spilling is not monotone in
this number (planar `(24,3,8)` at 26 is clean while `(8,8,8)` at 20 spills),
so it is used to *exclude* shapes that cannot possibly fit, never to rank the
ones that can.
"""
_planar_pressure(MR::Int, NR::Int, W::Int) = 2 * (MR ÷ W) * NR + 2 * (MR ÷ W) + 2

"""
    _fitted_shape(profile::TargetProfile, T, method) -> (MR, NR, W)

The shape used where no rule applies, and the target of extent demotion: the
smallest thing guaranteed to be constructible on this host.

Real: `_fallback_shape(T)`. Planar: the largest *menu* shape that fits the
host -- selected from the menu so that membership holds by construction --
under two necessary constraints: `W <= hardware lanes` (a `Vec{8,Float64}` on
128-bit NEON is emulated across four registers) and `_planar_pressure <=
nregisters` (16 when the register count is unknown, the conservative x86
baseline). Not `_fallback_shape`, whose pressure of 30 is over AVX2's 16. The
fitted shapes are unmeasured off `:avx512`; they are not claimed to be good,
only to run without spilling by the budget's own reckoning.
"""
_fitted_shape(::TargetProfile, ::Type{T}, ::RealMethod) where {T} = _fallback_shape(T)

# 1m's menus hold AVX-512 shapes only; its head is spill-free at every lane
# width it compiles, so it is the conservative choice.
_fitted_shape(::TargetProfile, ::Type{T}, method::OneMMethod) where {T} =
    first(kernel_shapes(T, method))

# FMAddSub: same reasoning as 1m -- the menu head, unmeasured.
_fitted_shape(::TargetProfile, ::Type{T}, method::FMAddSubMethod) where {T} =
    first(kernel_shapes(T, method))

function _fitted_shape(profile::TargetProfile, ::Type{T}, method::PlanarMethod) where {T}
    R = real(T)
    vb = profile.vector_bytes
    lanes = (vb > 0 && vb % sizeof(R) == 0) ? vb ÷ sizeof(R) : _default_lanewidth(R)
    budget = profile.nregisters > 0 ? profile.nregisters : 16
    best = nothing
    for shape in kernel_shapes(T, method)
        MR, NR, W = shape
        (W <= lanes && MR % W == 0) || continue
        _planar_pressure(MR, NR, W) <= budget || continue
        # Largest logical tile wins; ties by the wider vector.
        if best === nothing || (MR * NR, W) > (best[1] * best[2], best[3])
            best = shape
        end
    end
    # Unreachable for any budget >= 16 (see the menu comment); a correct but
    # slow kernel beats throwing here.
    return best === nothing ? last(kernel_shapes(T, method)) : best
end

# ----------------------------------------------------------------------------
# The default kernel for a profile, and the small-M demotion rule
# ----------------------------------------------------------------------------

# The default kernel for `T` on `profile`, uncached: what `_default_kernel(T)`
# (src/planning/defaults.jl) resolves for the detected host, and the form the
# tests use with synthetic profiles.
@noinline _kernel_for(profile::TargetProfile, ::Type{T}) where {T} =
    _kernel_from_shape(_derived_shape(profile, T), T, _default_method(T))

"""
    _extent_shape(profile, T, method, Qm) -> (MR, NR, W)

The derived shape, or -- real method only, and only when the derived shape is
the `MV = 4` rule shape -- its `MV = 2` sibling `(MR ÷ 2, NR, W)` when `Qm` is
shorter than one tall tile, or shorter than two and padded to strictly fewer
rows by the half-height tile. Always a member of `kernel_shapes(T, method)`.

The step-down rule is a padding comparison, not a throughput model: with
`MR = 32`, `Qm = 40` costs 64 padded rows against the half tile's 48 (1.33x
the work for a tile that is only ~1.2x faster per padded row, engine
measurement at large M), while `Qm = 56` pads to 64 either way and keeps the
tall tile. Above `2 MR` the tall tile's worst-case padding excess is `MR/2`
over `2 MR + 1` rows (< 1.2x) and it always wins, which is why the comparison
stops there. It never steps below `MV = 2`: an `MV = 1` real tile has half
the FMAs per broadcast and per loop iteration of the `MV = 2` tile and runs
at roughly half its rate, more than any padding it could save; `Qm < 2W`
still reaches the fitted shape through `_default_kernel`'s own demotion.
"""
_extent_shape(profile::TargetProfile, ::Type{T}, method, Qm::Int) where {T} =
    _extent_shape(_derived_shape(profile, T, method), T, method, Qm)

# The same rule applied to an already derived `shape`: the form the per-call
# path uses (`_default_shape`, src/planning/defaults.jl, passes the cached
# `ResolvedDefaults.shape`), so no `Val(profile.isa)` dispatch is repeated
# per plan. Pure and static: `kernel_shapes(T, method)` is a constant tuple.
@inline _extent_shape(shape::NTuple{3, Int}, ::Type{T}, method, Qm::Int) where {T} = shape

@inline function _extent_shape(shape::NTuple{3, Int}, ::Type{T}, method::RealMethod, Qm::Int) where {T}
    MR, NR, W = shape
    (Qm > 0 && MR == 4 * W && Qm < 2 * MR) || return shape
    half = (2 * W, NR, W)
    half in kernel_shapes(T, method) || return shape
    # Below one tall tile the half tile is taken unconditionally, padding tie
    # or not: keeping the tall shape there would hand `_default_kernel`'s
    # `Qm < mr` demotion the FITTED shape, two steps down instead of one
    # (measured at `Qm = 63`, Float32, K = N = 1024: fitted 58.6 GF/s against
    # the half tile's 94.6 and the tall tile's 109.3).
    Qm < MR && return half
    return cld(Qm, half[1]) * half[1] < cld(Qm, MR) * MR ? half : shape
end

"""
    _store_shape(shape, T, method, Qm, run) -> (MR, NR, W)

`shape`, or -- real method only, and only when `shape` is the `MV = 4` rule
shape -- its `MV = 2` sibling `(MR ÷ 2, NR, W)` when C's leading unit-stride
run along M (`run`, `_leading_unit_run`) does not make every tall register
sliver unit-stride (`Qm != run` and `run % MR != 0`) but is at least one half
tile long. Applied per orientation, BEFORE the swap decision
(`_candidate_mrs`, src/planning/plan.jl), and again to the chosen one, so the
swap and the run-length demotion (`_demote_for_run`) see the tile that will
actually run. Always a member of `kernel_shapes(T, method)`.

Why: a sliver that is not unit-stride in C takes the scattered store, whose
cost per tile is fixed while the kernel's work per tile grows with `Qk`, and
the `MV = 4` tile's 1.2x kernel rate does not pay for it below `Qk` ~ 200. The
measurement: 2026-09-28, ccqlin038 (Cascade Lake), Julia 1.12.7,
`benchmark/probes/probe_mv4_store_sweep.jl` (named kernels, no swap, no run
demotion; Qm = 1920-2048, Qn = 480-512; C = [a, n, b] so `run` = extent of
`a`), GF/s ratio `MV = 4` / `MV = 2`:

    Float64 (16 vs 32 rows)   Qk:  16    32    64   128   256   512  1024
      run = 16 (MV2 whole, MV4 broken)  0.31  0.46  0.59  0.68-0.76  0.95-0.98  0.89  0.95
      run = 24 (both broken)            0.55  0.64  0.77  0.88  1.03
      run = 48 (MV2 whole, MV4 half)    0.55  0.69  0.84  0.94  1.07  1.07  1.00
      run = 32, 64, 96 (both whole)     1.06-1.17 at Qk = 16..128
    Float32 (32 vs 64 rows)
      run = 32                          0.19  0.29  0.43  0.61  0.80  0.98
      run = 48 (both broken)            0.49  0.56  0.65  0.78  0.92  1.04
      run = 96                          0.39  0.51  0.69  0.85  1.00  1.05
      run = 64, 128 (both whole)        1.01-1.24 at every Qk (4..512)

Past the crossover (Qk ~ 200-500, and only where part of the tall slivers
is whole) the tall tile gains at most ~7%, and since `kc` caps a panel's
depth near there the ratio stops growing; below it the half tile is up to 5x
faster. So the rule has no `Qk` term. This is the regression the `MV = 4`
rule first shipped with: the swap decision (`_prefer_swap`) compared C's
16-long runs of `ccsd_t_2_dim16` / `ao2mo_2_dim16` against `mr = 32`,
declined the swap `MV = 2` had taken, and ran every tile through the
scattered store (2.2-2.6x slower end to end).

Below one half tile (`run < 2W`) the tall shape is kept: both tiles store
scattered there, and the tall one measured 0.88-1.00x at Qk <= 32 (where the
run demotion replaces either by the tile that divides `run`) and 1.05-1.16x
from Qk = 64 (Float64, run = 8).
"""
@inline _store_shape(shape::NTuple{3, Int}, ::Type{T}, method, Qm::Int, run::Int) where {T} = shape

@inline function _store_shape(shape::NTuple{3, Int}, ::Type{T}, method::RealMethod, Qm::Int, run::Int) where {T}
    MR, NR, W = shape
    (MR == 4 * W && Qm != run && run % MR != 0 && run >= 2 * W) || return shape
    half = (2 * W, NR, W)
    return half in kernel_shapes(T, method) ? half : shape
end

# Small-M demotion target for complex `T` on AVX-512: the native-width
# (`W == lanes`) FMAddSub menu shape that pads `Qm` least (`cld(Qm, MR) * MR`),
# ties by the larger `MR * NR` tile; `nothing` (keep the planar fitted shape)
# everywhere else. Reached only through `_default_kernel`'s extent demotion
# (src/planning/defaults.jl), i.e. when `Qm` is below the planar override's
# `MR` (24 for ComplexF64, 48 for ComplexF32).
#
# Split in two so the host-dependent half can be cached: `_small_m_candidates`
# is the native-width slice of the FMAddSub menu (a property of the profile
# and `T` alone, empty wherever the rule does not apply), and
# `_small_m_shape(candidates, Qm)` is the pure pick among them.
#
# Why: the planar fitted shape is the LARGEST fitting tile, `(16,6,8)` /
# `(32,6,16)` on AVX-512 -- the planar shape measured 38-41% worst there (see
# the override rows), and at ComplexF32 `Qm = 12` it still pads to 32 rows.
# Measured 2026-09-25, ccqlin038 (Cascade Lake, `:avx512`), Julia 1.13.0,
# `benchmark/bench_vs_openblas.jl` (21 reps, default blocking per kernel),
# GF/s default-before -> FMAddSub at the shape this rule picks [best 1m]:
#
#   ComplexF64 12x256x256: planar 16x6/W8 18.1 -> fmaddsub 12x8/W8 37.3 [1m 12x8 36.4]
#   ComplexF64 16x256x16:  planar 16x6/W8 21.5 -> fmaddsub  8x8/W8 42.1 [1m  8x8 37.9]
#   ComplexF32 12x256x256: planar 32x6/W16 20.7 -> fmaddsub 16x8/W16 61.7 [1m 16x8 56.4]
#   ComplexF32 16x256x16:  planar 32x6/W16 20.9 -> fmaddsub 16x8/W16 75.0 [1m 16x8 67.5]
#
# (OpenBLAS: 49.1 / 38.6 / 59.2 / 58.9.) The best planar shape at any size in
# the menu is 28.0 / 32.1 / 42.7 / 58.3, so the method switch, not just a
# smaller planar tile, is what closes the gap; FMAddSub beats 1m at 3 of 4
# cells (the fourth ties). Every larger-M shape in the sweep has
# `Qm >= mr(override)` and never reaches this: an interleaved old/new re-run
# of the whole sweep (all four dtypes, 3 ABAB rounds, canary spread 3.7%/7.1%)
# plans the identical kernel at the other 32 cells, and the four above land at
# 1.52-1.65x (ComplexF64) and 2.35-2.95x (ComplexF32) the old default. The
# rest of the range this rule covers, probed at `Mx256x256` the same way
# (canary spread 5.4%): ComplexF64 M = 1, 4, 8, 20, 23 all gain 1.91-1.93x
# (M >= 24 keeps planar `24x3`); ComplexF32 M = 1, 4, 8, 20, 23, 24, 32, 37,
# 47 all gain 1.54-2.53x (M = 23..47 reach 80-143% of OpenBLAS, against
# 45-79% before). Not extended to `:avx2` (where
# the planar override has `MR = 4`, so demotion needs `Qm < 4`; unmeasured)
# or to other ISAs.
_small_m_candidates(::Val, ::TargetProfile, ::Type) = NTuple{3, Int}[]
function _small_m_candidates(::Val{:avx512}, profile::TargetProfile, ::Type{T}) where {T <: Complex}
    lanes = profile.vector_bytes ÷ sizeof(real(T))
    return [shape for shape in kernel_shapes(T, FMAddSubMethod()) if shape[3] == lanes]
end

function _small_m_shape(candidates::Vector{NTuple{3, Int}}, Qm::Int)
    best = nothing
    for shape in candidates
        MR, NR, _ = shape
        key = (-(cld(Qm, MR) * MR), MR * NR)
        if best === nothing || key > best[1]
            best = (key, shape)
        end
    end
    return best === nothing ? nothing : best[2]
end

_small_m_shape(key::Val, profile::TargetProfile, ::Type{T}, Qm::Int) where {T} =
    _small_m_shape(_small_m_candidates(key, profile, T), Qm)

# Run-length-aware demotion, applied to whichever operand the M/N swap decision
# chose to feed M. The vectorized store needs EVERY register sliver unit-stride
# in C; given a leading unit-stride run of
# length `run` in C, that holds iff `Qm == run || run % mr(kernel) == 0` (not
# the weaker `mr <= run`: at run=20, mr=16 only 40% of slivers are
# contiguous). When the kernel fails this, demote to the LARGEST menu shape
# whose `mr` divides `run` (at run=16 for Float32, `(16,6,8)` beats
# `(8,6,8)`), which reuses an already-compiled specialization.
#
# The demotion wins by fixing the store, but it also changes the packing and
# microkernel cost, which grows with the contracted extent `Qk`; past
# `_RUN_DEMOTE_KMAX_*` that cost dominates and demoting loses, so the guard
# declines to demote there.
#
# Real element types only. Complex kernels now have a vectorized store too, so
# extending this is a deliberately deferred, unmeasured follow-up. The menu
# search below already goes through the kernel's own method, but lifting the
# guard would also need `plan_contract` to compute `run_m`/`run_n` for complex
# `T` (it passes `0` today) and a measured complex `Qk` cutoff.
#
# Order matters for cost: the cheap `Qm == run` / `run % mr == 0`
# short-circuits run before the O(Qm/mr) `_unbroken_fraction` scan, which is
# additionally skipped while `_RUN_DEMOTE_BROKEN_ENOUGH` is inert (`1.0`). Whenever a
# short-circuit holds the fraction is `1.0` anyway, so the order changes no
# decision.
function _demote_for_run(::Type{T}, kernel, run::Int, Qm::Int, Qk::Int) where {T}
    method = complex_method(kernel)
    target = _run_demotion_target(T, mr(kernel), method, run, Qm, Qk)
    return target === nothing ? kernel : _kernel_from_shape(target, T, method)
end

# The same rule on a `(shape, method)` pair, as `plan_contract` applies it
# before its kernel barrier (`_plan_with_kernel`, src/planning/plan.jl): the
# kernel built from `shape` has `mr == shape[1]` for every method.
@inline function _demote_shape_for_run(
        ::Type{T}, shape::NTuple{3, Int}, method, run::Int, Qm::Int, Qk::Int
    ) where {T}
    target = _run_demotion_target(T, shape[1], method, run, Qm, Qk)
    return target === nothing ? shape : target
end

# The menu shape `_demote_for_run` demotes a kernel of register height `mrk`
# (under `method`) to, or `nothing` to keep it.
function _run_demotion_target(::Type{T}, mrk::Int, method, run::Int, Qm::Int, Qk::Int) where {T}
    T <: Real || return nothing
    kmax = T === Float64 ? _RUN_DEMOTE_KMAX_F64 : _RUN_DEMOTE_KMAX_F32
    Qk > kmax && return nothing
    Qm == run && return nothing
    run % mrk == 0 && return nothing
    _RUN_DEMOTE_BROKEN_ENOUGH < 1.0 && _unbroken_fraction(Qm, run, mrk) > _RUN_DEMOTE_BROKEN_ENOUGH &&
        return nothing
    best = nothing
    for shape in kernel_shapes(T, method)
        m = shape[1]
        if run % m == 0 && (best === nothing || m > best[1])
            best = shape
        end
    end
    return best
end

# Deepest `Qk` at which run-length demotion still wins: it crosses over from a
# win to a loss between Qk=32 and 64 for Float64 and between 64 and 128 for
# Float32.
const _RUN_DEMOTE_KMAX_F64 = 32
const _RUN_DEMOTE_KMAX_F32 = 64

# Demotion is also skipped when more than this fraction of the register
# slivers already lie inside one run. Inert at `1.0`: the evidence conflicts
# between Float64 and Float32 and one shared threshold cannot satisfy both, so
# the `Qk` cutoff above is the only active guard. Known residual: on
# less-broken shapes demotion still fires, and can lose, for `Qk <= kmax(T)`.
const _RUN_DEMOTE_BROKEN_ENOUGH = 1.0

# Fraction of the `cld(Qm, mr)` register slivers `[s*mr, min((s+1)*mr, Qm))`
# that lie ENTIRELY inside one run of length `run`, i.e. `lo ÷ run == hi ÷
# run` for the sliver's first and last valid index. Exactly `1.0` iff
# `Qm == run || run % mr == 0`, the predicate `_demote_for_run` tests first.
# Precondition, not checked: `1 <= run <= Qm` (`_leading_unit_run` never
# returns more than `Qm`, and callers never pass `run == 0`).
function _unbroken_fraction(Qm::Int, run::Int, mr::Int)::Float64
    nslivers = cld(Qm, mr)
    nslivers == 0 && return 1.0
    whole = 0
    for s in 0:(nslivers - 1)
        lo = s * mr
        hi = min((s + 1) * mr, Qm) - 1
        whole += lo ÷ run == hi ÷ run
    end
    return whole / nslivers
end
