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

# The core's private L2 share, or 1 MB when undetected.
function l2_core_bytes(profile::TargetProfile)
    profile.l2.bytes > 0 || return 1 << 20
    return core_bytes(profile, profile.l2)
end

# Through the per-eltype cache; an eltype without a slot fails later in planning.
@inline l2_core_bytes(::Type{T}) where {T} =
    _defaults_slot(T) === nothing ? l2_core_bytes(target_profile()) : _resolved_defaults(T).l2_core

function _resolve_defaults(profile::TargetProfile, ::Type{T}) where {T}
    method = default_method(T)
    # First, so an element type with no menu throws from here.
    shape = derived_shape(profile, T, method)
    fitted = fitted_shape(profile, T, method)
    small_m = small_m_candidates(Val(profile.isa), profile, T)
    real_row = _real_blocking_row(profile, real(T))
    l2_core = l2_core_bytes(profile)
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
    kernel_from_shape(_resolved_defaults(T).shape, T, default_method(T))

# `@noinline`: the return type is the Union of `T`'s menu kernels.
@noinline function _default_kernel(::Type{T}, m_length::Int, n_length::Int) where {T}
    shape, method = select_shape(T, default_method(T), m_length, 0, m_length)
    return kernel_from_shape(shape, T, method)
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
    return _scale_blocking(_resolved_defaults(T).real_row, KernelMethod(kernel))
end
