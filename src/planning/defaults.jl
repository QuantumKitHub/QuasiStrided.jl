# The engine's per-host defaults -- the register shape, the small-M demotion
# targets and the cache-blocking row for an element type -- resolved ONCE per
# (detected profile, element type) and cached, so that `plan_contract` reads
# them instead of re-deriving them on every call.
#
# Why a cache at all: everything in src/planning/kernel_selection.jl and
# src/planning/blocking.jl is keyed on `Val(profile.isa)`, and `profile.isa`
# is a runtime `Symbol`, so each of those lookups is a dynamic dispatch (a
# `Val` type built at runtime, then `jl_apply_generic`). Chained -- override
# row, rule, fitted shape, small-M rule, the blocking model, each behind its
# own `Val` -- they dominated the per-call floor. Measured on ccqlin038
# (Cascade Lake, Julia 1.12.7) before this file existed:
# `_default_kernel(ComplexF64, 8, 8)` took 1.2 us and `plan_contract` called
# it twice, `default_blocking(kernel)` another 0.3-0.4 us; the same two for
# `Float64` were 75 ns and 95 ns. Against a whole 8x8x8 ComplexF64
# contraction of 5.5 us (StridedBLAS: 0.4 us), planning was 3.6 us of it.
# With the cache the two reads are a slot load and a profile comparison.

"""
    ResolvedDefaults

What `plan_contract` needs from the hardware profile for one element type,
resolved by [`_resolve_defaults`](@ref) and cached per (profile, eltype) by
[`_resolved_defaults`](@ref): the default register `shape`
(`_derived_shape`), the `fitted` shape the off-AVX-512 small-M demotion falls
to (`_fitted_shape`), the native-width FMAddSub `small_m` candidates of the
AVX-512 complex small-M rule (`_small_m_candidates`, empty where the rule does
not apply), the unscaled `real_row` of the cache-blocking model for
`real(T)` (`_real_blocking_row`), and the core's private L2 share `l2_core`
that the K-order cost model reads (`_l2_core_bytes`, src/planning/labels.jl).
`profile` is the profile these were derived
from and is the cache key: a cached entry serves exactly while it is `===`
the current [`target_profile`](@ref). Every field but the candidate list is
isbits, so reads are type-stable and allocation-free.
"""
struct ResolvedDefaults
    profile::TargetProfile
    shape::NTuple{3, Int}
    fitted::NTuple{3, Int}
    small_m::Vector{NTuple{3, Int}}
    real_row::Blocking
    l2_core::Int
end

"""
    _resolve_defaults(profile::TargetProfile, T) -> ResolvedDefaults

Derive every [`ResolvedDefaults`](@ref) field for `T` on `profile` through
the pure selection functions of src/planning/kernel_selection.jl and
src/planning/blocking.jl. Uncached; `_resolved_defaults` is the cached entry.
"""
function _resolve_defaults(profile::TargetProfile, ::Type{T}) where {T}
    method = _default_method(T)
    # First, so an element type with no menu throws from `kernel_shapes` here,
    # as `_kernel_for` did, rather than from a later field.
    shape = _derived_shape(profile, T, method)
    fitted = _fitted_shape(profile, T, method)
    small_m = _small_m_candidates(Val(profile.isa), profile, T)
    real_row = _real_blocking_row(profile, real(T))
    l2_core = _l2_core_bytes(profile)
    return ResolvedDefaults(profile, shape, fitted, small_m, real_row, l2_core)
end

# One slot per supported element type, so the lookup is a method dispatch on
# `T` rather than a `Dict` probe. `nothing` for any other `T`: those take the
# uncached path and fail in `_resolve_defaults` exactly as they did before
# (no shape menu for the type). A slot is filled lazily, on the first plan
# for its type after the profile last changed.
const _DEFAULTS_F64 = Ref{Union{Nothing, ResolvedDefaults}}(nothing)
const _DEFAULTS_F32 = Ref{Union{Nothing, ResolvedDefaults}}(nothing)
const _DEFAULTS_C64 = Ref{Union{Nothing, ResolvedDefaults}}(nothing)
const _DEFAULTS_C32 = Ref{Union{Nothing, ResolvedDefaults}}(nothing)
_defaults_slot(::Type{Float64}) = _DEFAULTS_F64
_defaults_slot(::Type{Float32}) = _DEFAULTS_F32
_defaults_slot(::Type{ComplexF64}) = _DEFAULTS_C64
_defaults_slot(::Type{ComplexF32}) = _DEFAULTS_C32
_defaults_slot(::Type) = nothing

"""
    _resolved_defaults(T) -> ResolvedDefaults

The cached [`ResolvedDefaults`](@ref) of `T` for the current
[`target_profile`](@ref), (re)derived when the slot is empty or holds an
entry for a different profile -- which is how a profile installed after load
(`_init_target!`, or a test writing `_TARGET[]` directly, as
test/forced_isa_runner.jl does) takes effect without an explicit
invalidation hook. The comparison is `===` on the immutable `TargetProfile`,
i.e. field-wise, so two detections of the same host share an entry.

Not locked: a racing refill computes the same immutable value and the slot
write is a single pointer store, so the worst case is redundant work.
"""
@inline function _resolved_defaults(::Type{T}) where {T}
    slot = _defaults_slot(T)
    profile = target_profile()
    slot === nothing && return _resolve_defaults(profile, T)
    cached = slot[]
    (cached !== nothing && cached.profile === profile) && return cached
    return _refill_defaults!(slot, profile, T)
end

@noinline function _refill_defaults!(slot, profile::TargetProfile, ::Type{T}) where {T}
    fresh = _resolve_defaults(profile, T)
    slot[] = fresh
    return fresh
end

# ----------------------------------------------------------------------------
# The engine's default kernel, and plan-time demotion
# ----------------------------------------------------------------------------

# `_kernel_for(target_profile(), T)`, read through the cache.
_default_kernel(::Type{T}) where {T} =
    _kernel_from_shape(_resolved_defaults(T).shape, T, _default_method(T))

# Extent-aware choice, used only when the caller did not name a kernel: a
# contraction whose M extent cannot fill one register tile pads every
# micro-tile away, so demote to the fitted shape; before that, a real `MV = 4`
# shape steps down to its `MV = 2` sibling where the taller tile would pad
# more (`_extent_shape`, src/planning/kernel_selection.jl, applied to the
# cached shape). Both key on padding waste, a countable quantity known at plan
# time, not on a cache estimate. `shape[1]` is `mr` of the kernel
# `_kernel_from_shape` builds at `shape`.
#
# Returns the `(shape, method)` pair as plain values rather than the kernel:
# `plan_contract` carries the shape across its kernel barrier as a singleton
# `Val` (`_plan_with_kernel`, src/planning/plan.jl), so the choice is made
# without ever holding a menu-wide kernel Union (ten members for ComplexF64).
# `_default_kernel` below builds the kernel from the same pair.
@inline function _default_shape(::Type{T}, Qm::Int, Qn::Int) where {T}
    d = _resolved_defaults(T)
    method = _default_method(T)
    shape = _extent_shape(d.shape, T, method, Qm)
    (Qm > 0 && Qm < shape[1]) || return (shape, method)
    # The FMAddSub small-M demotion is complex-only (`_small_m_candidates` is
    # empty for a real `T`); saying so statically keeps a real `T`'s method a
    # concrete `RealMethod` rather than a Union with `FMAddSubMethod`.
    T <: Complex || return (d.fitted, method)
    small = _small_m_shape(d.small_m, Qm)
    small === nothing || return (small, FMAddSubMethod())
    return (d.fitted, method)
end

# `@noinline` deliberately: the return type is the Union of every menu kernel
# of `T`, and inlining that Union into a caller would only widen the caller's
# inference for no gain.
@noinline function _default_kernel(::Type{T}, Qm::Int, Qn::Int) where {T}
    shape, method = _default_shape(T, Qm, Qn)
    return _kernel_from_shape(shape, T, method)
end

# ----------------------------------------------------------------------------
# default_blocking(kernel)
# ----------------------------------------------------------------------------

"""
    default_blocking(kernel) -> Blocking

Cache-blocking factors for `kernel` on this host: an analytical model of the
detected cache geometry ([`target_profile`](@ref), `_modelled_blocking`) for
the real row at the engine's real register shape, scaled per complex method
by packed reals per element. With L1d or L2 undetected, fixed fallback
constants instead. Measured grids are wide plateaus (1.7-17% best-to-worst
over `bench_driver.jl`'s 36 points, 4-24% over 157) whose one cliff is small
`kc`, so no row is a sharp optimum. `plan_contract` rounds `mc`/`nc` to
`mr`/`nr` multiples and clamps all three to the contraction's extents.

Identical to `default_blocking(Val(target_profile().isa), scalartype(kernel),
complex_method(kernel))`, read through the per-(profile, eltype) cache: the
real row for `real(scalartype(kernel))` comes from [`_resolved_defaults`](@ref)
and is scaled here by the kernel's own method (`_scale_blocking`, the identity
for a real kernel).
"""
function default_blocking(kernel)
    T = scalartype(kernel)
    return _scale_blocking(_resolved_defaults(T).real_row, complex_method(kernel))
end
