# The engine's per-host defaults for an element type, resolved once per
# (detected profile, eltype) and cached: every lookup in kernel_selection.jl and
# blocking.jl is keyed on `Val(profile.isa)`, a dynamic dispatch far too slow to
# repeat on every `plan_contract`.

# The default `shape`, the `fitted` small-M demotion shape, the AVX-512 complex
# small-M FMAddSub candidates, the unscaled real blocking row and the core's L2
# share, all derived from `profile`, which is the cache key.
struct ResolvedDefaults
    profile::TargetProfile
    shape::NTuple{3, Int}
    fitted::NTuple{3, Int}
    small_m::Vector{NTuple{3, Int}}
    real_row::Blocking
    l2_core::Int
end

function _resolve_defaults(profile::TargetProfile, ::Type{T}) where {T}
    method = _default_method(T)
    # First, so an element type with no menu throws from here.
    shape = _derived_shape(profile, T, method)
    fitted = _fitted_shape(profile, T, method)
    small_m = _small_m_candidates(Val(profile.isa), profile, T)
    real_row = _real_blocking_row(profile, real(T))
    l2_core = _l2_core_bytes(profile)
    return ResolvedDefaults(profile, shape, fitted, small_m, real_row, l2_core)
end

# One slot per supported element type (a method dispatch, not a `Dict` probe);
# other types take the uncached path and fail there.
const _DEFAULTS_F64 = Ref{Union{Nothing, ResolvedDefaults}}(nothing)
const _DEFAULTS_F32 = Ref{Union{Nothing, ResolvedDefaults}}(nothing)
const _DEFAULTS_C64 = Ref{Union{Nothing, ResolvedDefaults}}(nothing)
const _DEFAULTS_C32 = Ref{Union{Nothing, ResolvedDefaults}}(nothing)
_defaults_slot(::Type{Float64}) = _DEFAULTS_F64
_defaults_slot(::Type{Float32}) = _DEFAULTS_F32
_defaults_slot(::Type{ComplexF64}) = _DEFAULTS_C64
_defaults_slot(::Type{ComplexF32}) = _DEFAULTS_C32
_defaults_slot(::Type) = nothing

# Refilled whenever the slot's profile is not `===` the current one, so a
# profile installed after load (as test/forced_isa_runner.jl does) takes effect
# without an invalidation hook. Unlocked: a racing refill stores the same value.
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

_default_kernel(::Type{T}) where {T} =
    _kernel_from_shape(_resolved_defaults(T).shape, T, _default_method(T))

# The automatic `(shape, method)` for extents `m_length`/`n_length` and C's
# unit-stride run along M (`run = m_length`: no layout known). Returned as plain
# values so `plan_contract` never holds a menu-wide kernel Union. The extent and
# store step-downs apply first; then an `m_length` that cannot fill one tile
# demotes to the fitted shape, or on AVX-512 complex to FMAddSub.
@inline function _default_shape(::Type{T}, m_length::Int, n_length::Int, run::Int = m_length) where {T}
    d = _resolved_defaults(T)
    method = _default_method(T)
    shape = _store_shape(_extent_shape(d.shape, T, method, m_length), T, method, m_length, run)
    (m_length > 0 && m_length < shape[1]) || return (shape, method)
    # Static, so a real `T`'s method stays a concrete `RealMethod`.
    T <: Complex || return (d.fitted, method)
    small = _small_m_shape(d.small_m, m_length)
    small === nothing || return (small, FMAddSubMethod())
    return (d.fitted, method)
end

# The automatic `(shape, method)` under `method`, `_default_method(T, TA, TB)`.
@inline _default_shape(::Type{T}, ::KernelMethod, m_length::Int, n_length::Int, run::Int) where {T} =
    _default_shape(T, m_length, n_length, run)

# The real default shape of the real problem, mapped: the real extent, store and
# small-M demotions carry over, on the real type's cached defaults.
@inline function _default_shape(::Type{T}, method::_MixedMethod, m_length::Int, n_length::Int, run::Int) where {T}
    shape, _ = _default_shape(real(T), _real_problem(method, m_length, n_length, run)...)
    return (_mixed_shape(method, shape), method)
end

# `@noinline`: the return type is the Union of `T`'s menu kernels.
@noinline function _default_kernel(::Type{T}, m_length::Int, n_length::Int) where {T}
    shape, method = _default_shape(T, m_length, n_length)
    return _kernel_from_shape(shape, T, method)
end

"""
    default_blocking(kernel) -> Blocking

Cache-blocking factors for `kernel` on this host: an analytical model of the
detected cache geometry ([`target_profile`](@ref)), or fixed constants where
L1d or L2 is undetected, scaled for complex methods by packed reals per
element. `plan_contract` rounds `m_block`/`n_block` to multiples of the tile
size and clamps all three to the contraction's extents.
"""
function default_blocking(kernel)
    T = scalartype(kernel)
    return _scale_blocking(_resolved_defaults(T).real_row, complex_method(kernel))
end
