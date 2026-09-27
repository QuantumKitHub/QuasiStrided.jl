# Cache-blocking factors for the driver's five-loop nest. `Blocking` is a
# plain, always-valid (>=1 per field) user-facing value; `plan_contract`
# rounds/clamps it into the *effective* blocking stored on `ContractPlan`.

"""
    Blocking(mc::Int, kc::Int, nc::Int)

Cache-blocking factors for the macro-blocking driver: `mc` (M block extent,
loop 3), `kc` (K block/panel depth, loop 4), `nc` (N block extent, loop 5).
All three fields must be `>= 1`; the constructor throws `ArgumentError`
otherwise.
"""
struct Blocking
    mc::Int
    kc::Int
    nc::Int

    function Blocking(mc::Int, kc::Int, nc::Int)
        mc >= 1 || throw(ArgumentError("Blocking requires mc >= 1, got mc = $mc"))
        kc >= 1 || throw(ArgumentError("Blocking requires kc >= 1, got kc = $kc"))
        nc >= 1 || throw(ArgumentError("Blocking requires nc >= 1, got nc = $nc"))
        return new(mc, kc, nc)
    end
end

# `default_blocking(kernel)`, the form `plan_contract` consumes, lives in
# src/planning/defaults.jl: it reads the real row below through the
# per-(profile, eltype) cache rather than re-deriving it on every plan.

# Complex blocking is the real row divided by the packed reals per element of
# each operand, so every method gets the same packed BYTE budget rather than
# the same element count. Deriving it from `sizeof(T)` instead would hand 1m
# (four reals per packed A element, planar two) double the L2 footprint and
# rig any planar-vs-1m comparison; this way 1m's `mc` comes out at exactly
# half planar's.
@inline _scale_blocking(base::Blocking, m::ComplexMethod) =
    Blocking(max(1, base.mc ÷ a_reals(m)), base.kc, max(1, base.nc ÷ b_reals(m)))

# The real rows, unscaled.
default_blocking(v::Val, ::Type{T}, ::RealMethod) where {T} = default_blocking(v, T)

# `T` is deliberately unconstrained rather than `T <: Complex`: with the bound,
# this method and the `RealMethod` one above are mutually ambiguous at
# `(Val, Type{<:Complex}, RealMethod)`. Unbounded, the `RealMethod` method is
# strictly more specific and wins.
default_blocking(v::Val, ::Type{T}, m::ComplexMethod) where {T} =
    _scale_blocking(default_blocking(v, real(T)), m)

# The fallback, for a host whose L1d or L2 size is undetected. These are the
# former measured AVX-512 row (Cascade Lake, 2026-09-11). Against the previous
# fallback (64, 128, 768) / (96, 384, 1152), which sat on the small-`kc` cliff,
# they score 1.033 vs 1.070 on Rome at the (8, 6) shape, 1.013 vs 1.029 on
# Genoa, 1.016 vs 1.048 on Ice Lake-SP and 1.071 vs 1.090 on Cascade Lake
# (Float64; same benchmark and normalization as `_modelled_blocking` below).
_fallback_blocking(::Type{Float64}) = Blocking(128, 256, 768)
_fallback_blocking(::Type{Float32}) = Blocking(96, 768, 1152)
_fallback_blocking(::Type{T}, m::ComplexMethod = PlanarMethod()) where {T <: Complex} =
    _scale_blocking(_fallback_blocking(real(T)), m)
default_blocking(::Val, ::Type{T}) where {T} = _fallback_blocking(T)

# Every ISA takes the model; the `Val` key is kept so a future measured row
# can override one ISA without touching the others.
default_blocking(::Val, ::Type{T}) where {T <: Union{Float32, Float64}} =
    _real_blocking_row(target_profile(), T)

# The real row for `T` on `profile`: the cache model where the geometry is
# detected, the fallback constants otherwise. The one definition behind both
# `default_blocking(::Val, T)` above and the per-(profile, eltype) cache in
# src/planning/defaults.jl, so the two cannot disagree.
_real_blocking_row(profile::TargetProfile, ::Type{T}) where {T <: Union{Float32, Float64}} =
    something(_modelled_blocking(profile, T), _fallback_blocking(T))

# --- analytical real row, from the detected cache geometry -------------------
#
# Byte accounting of `_execute_nest!` (src/execution/execute.jl): per (jc, pc)
# the packed B panel is `nc*kc` elements, per (jc, pc, ic) the packed A block is
# `mc*kc`, and loop 2 (jr) is outside loop 1 (ir), so one `NR x kc` B sliver is
# reused by every A sliver of the block while those stream past it. Hence
#
#     NR*kc*S  <= L1/2              the reused B sliver, half of L1
#     mc*kc*S  <= L2core/2          the A block, reused by every B sliver
#     nc*kc*S  <= L2core + L3core   the B panel, reused by every A block
#
# with `S = sizeof(T)`, rounded down to MR/NR multiples. Shared levels are
# divided by the cores sharing them (`sharing ÷ l1d.sharing`, L1d being private
# to one core's SMT siblings): on a shared node a single-threaded contraction
# cannot count on another core's slice. With no L3, `L3core` is 0.
#
# The `nc` bound is deliberately the whole per-core capacity (non-inclusive L3,
# as on every machine measured), not half the L3: an oversized B panel only
# re-streams B at `S/(2*mc)` bytes per flop, while an undersized one repacks A
# once per `jc` block. Measured (bench_blocking_model.jl, 2026-09-25, jobs
# 7107932-4 on Rome/Genoa/Ice Lake-SP): `nc <= 192` costs 3-6% on every
# machine, `nc` in 768..6144 is within ~3% of the best even where the B panel
# is 8x the core's L3 share; the half-L3 bound gave Ice Lake `nc = 192`.
#
# Measured against the fallback and the grid (same benchmark, jobs 7108316-8
# and a local ccqlin038 run; geomean over 9 shapes of time normalized per
# shape by the best of 157 points): model / fallback, Float64, Rome 1.028 /
# 1.033, Genoa 1.009 / 1.013, Ice Lake-SP 1.029 / 1.016, Cascade Lake 1.102 /
# 1.071, Apple M3 (NEON, macOS: no L3 reported, so `nc` comes from the L2
# alone) 1.020 / 1.015 -- on the plateau everywhere, and never more than 3%
# from the constants either way.
#
# Two refinements measured and NOT adopted (jobs 7109030-2, 2026-09-25,
# per-shape paired ratios against the same run's repeat noise, whose middle
# half spans 0.998-1.003): `kc` from L1 associativity, the B sliver taking
# (ways-1)÷2 ways (median ratio 1.000-1.002, middle half 0.995-1.020), and
# splitting K into equal blocks under `kc` on non-power-of-two K (median
# 0.999-1.000, middle half 0.995-1.010). Blocking is at its noise floor here.
#
# Real rows only; complex rows go through `_scale_blocking` as for every other
# row. `nothing` when L1d or L2 is undetected.
function _modelled_blocking(profile::TargetProfile, ::Type{T}, MR::Int, NR::Int) where {T}
    l1, l2, l3 = profile.l1d, profile.l2, profile.l3
    (l1.bytes > 0 && l2.bytes > 0) || return nothing
    smt = max(1, l1.sharing)
    core_share(c) = c.bytes ÷ max(1, c.sharing ÷ smt)
    l2core = core_share(l2)
    l3core = l3.bytes > 0 ? core_share(l3) : 0
    S = sizeof(T)
    kc = max(1, (l1.bytes ÷ 2) ÷ (NR * S))
    mc = max(MR, ((l2core ÷ 2) ÷ (kc * S)) ÷ MR * MR)
    nc = max(NR, ((l2core + l3core) ÷ (kc * S)) ÷ NR * NR)
    return Blocking(mc, kc, nc)
end

# At the real kernel shape the engine derives for `T` on `profile`.
function _modelled_blocking(profile::TargetProfile, ::Type{T}) where {T <: Real}
    MR, NR, _ = _derived_shape(profile, T)
    return _modelled_blocking(profile, T, MR, NR)
end

# For a bare scalar type: the fallback row, independent of the host.
default_blocking(::Type{Float64}) = _fallback_blocking(Float64)
default_blocking(::Type{Float32}) = _fallback_blocking(Float32)

# Smallest multiple of `n` that is >= `x` (`x >= 0`, `n >= 1`).
@inline _roundup(x::Int, n::Int) = cld(x, n) * n
