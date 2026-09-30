# Kernel selection: the microkernel `plan_contract` builds when the caller does
# not name one. Everything here is a pure function of a `TargetProfile` and `T`;
# src/planning/defaults.jl caches the host-dependent results per eltype, since
# the `Val(profile.isa)` dispatch below is dynamic.

# `OneMMethod`/`FMAddSubMethod` are chosen only by naming the kernel (FMAddSub
# also by the AVX-512 small-M demotion, `_small_m_shape`).
_default_method(::Type{<:Real}) = RealMethod()
_default_method(::Type{<:Complex}) = PlanarMethod()

const _MixedMethod = Union{ComplexRealMethod, RealComplexMethod}

# The method for compute type `T` and (oriented) operand eltypes: a mixed-domain
# method when exactly one operand is real.
_default_method(::Type{T}, ::Type{TA}, ::Type{TB}) where {T, TA, TB} =
    _default_method(T)
_default_method(::Type{T}, ::Type{<:Complex}, ::Type{<:Real}) where {T <: Complex} =
    ComplexRealMethod()
_default_method(::Type{T}, ::Type{<:Real}, ::Type{<:Complex}) where {T <: Complex} =
    RealComplexMethod()

const NR_DEFAULT = 6

# An (8, 6) tile at one 256-bit register's lane width: the shape wherever no rule applies.
_fallback_shape(::Type{T}) where {T} = (8, NR_DEFAULT, _default_lanewidth(real(T)))

# Closed menus of `(MR, NR, W)` shapes, so the compiled specializations stay
# bounded. Complex `MR` counts complex rows and `W` real lanes; 1m runs a real
# kernel of `2MR` rows, so only `2MR` must be a multiple of `W`. The trailing
# `MV = 1` planar entries guarantee `_fitted_shape` a fit on any register file
# of >= 16 registers.
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
# FMAddSub starts from 1m's shapes (same accumulator layout); the `NR = 5` AVX2
# entries fit 16 registers, where `NR = 6` spills.
const KERNEL_SHAPES_C64_FMADDSUB = ((12, 8, 8), (8, 8, 8), (4, 6, 4), (4, 5, 4))
const KERNEL_SHAPES_C32_FMADDSUB = ((24, 8, 16), (16, 8, 16), (8, 6, 8), (8, 5, 8))

"""
    kernel_shapes(T, method::ComplexMethod = _default_method(T)) -> NTuple{<:Any,NTuple{3,Int}}

The closed menu of `(MR, NR, W)` register shapes the engine may build for
element type `T` under `method`.
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
# The mixed menus are the real menu of `real(T)`, mapped by `_mixed_shape`.
kernel_shapes(::Type{T}, method::_MixedMethod) where {T <: Union{ComplexF32, ComplexF64}} =
    map(s -> _mixed_shape(method, s), kernel_shapes(real(T), RealMethod()))

# A real inner shape to the mixed kernel's `(MR, NR, W)`, and back: complex A
# rows and complex B columns each span two reals of the inner kernel.
_mixed_shape(::ComplexRealMethod, (MR, NR, W)::NTuple{3, Int}) = (MR ÷ 2, NR, W)
_mixed_shape(::RealComplexMethod, (MR, NR, W)::NTuple{3, Int}) = (MR, NR ÷ 2, W)
_real_shape(::ComplexRealMethod, (MR, NR, W)::NTuple{3, Int}) = (2 * MR, NR, W)
_real_shape(::RealComplexMethod, (MR, NR, W)::NTuple{3, Int}) = (MR, 2 * NR, W)

# The real problem the inner kernel sees: extents and C's unit-stride M run in reals.
_real_problem(::ComplexRealMethod, m_length::Int, n_length::Int, run::Int) = (2 * m_length, n_length, 2 * run)
_real_problem(::RealComplexMethod, m_length::Int, n_length::Int, run::Int) = (m_length, 2 * n_length, run)

# The kernel type implementing `method` for `T`, or `nothing` (exactly the
# pairs with a menu).
_kernel_type(::RealMethod, ::Type{<:Union{Float32, Float64}}) = SIMDKernel
_kernel_type(::PlanarMethod, ::Type{<:Union{ComplexF32, ComplexF64}}) = PlanarKernel
_kernel_type(::OneMMethod, ::Type{<:Union{ComplexF32, ComplexF64}}) = OneMKernel
_kernel_type(::FMAddSubMethod, ::Type{<:Union{ComplexF32, ComplexF64}}) = FMAddSubKernel
_kernel_type(::ComplexRealMethod, ::Type{<:Union{ComplexF32, ComplexF64}}) = ComplexRealKernel
_kernel_type(::RealComplexMethod, ::Type{<:Union{ComplexF32, ComplexF64}}) = RealComplexKernel
_kernel_type(::Any, ::Type) = nothing

# An unrolled `shape === menu[i] ? ... :` ladder over the menu whose branches
# are built from literals, so each is concrete; a shape outside the menu throws.
# Each branch builds the kernel, or `Val(shape)` when `valonly`.
function _menu_ladder(::Type{T}, method, valonly::Bool) where {T}
    K = _kernel_type(method, T)
    K === nothing && return :(_throw_no_kernel(shape, T, method))
    ex = :(_throw_shape_not_in_menu(shape, T, method))
    for (MR, NR, W) in reverse(kernel_shapes(T, method))
        arm = valonly ? :(Val(($MR, $NR, $W))) : :($K(Val($MR), Val($NR), T, Val($W)))
        ex = :(shape === ($MR, $NR, $W) ? $arm : $ex)
    end
    return ex
end

# The `method` kernel for `T` at a menu `shape` (one dynamic dispatch per plan).
_kernel_from_shape(shape::Tuple{Int, Int, Int}, ::Type{T}) where {T} =
    _kernel_from_shape(shape, T, _default_method(T))
@generated _kernel_from_shape(shape::Tuple{Int, Int, Int}, ::Type{T}, method::M) where {T, M} =
    _menu_ladder(T, M.instance, false)

# `Val(shape)` for a menu `shape`: a singleton, so it crosses `plan_contract`'s
# kernel barrier without boxing.
@generated _menu_val(shape::Tuple{Int, Int, Int}, ::Type{T}, method::M) where {T, M} =
    _menu_ladder(T, M.instance, true)

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

# Per-ISA shapes, consulted first. None for real types: the rule below is the optimum.
_shape_override(key::Val, ::Type{T}) where {T} = _shape_override(key, T, _default_method(T))
_shape_override(::Val, ::Type, ::ComplexMethod) = nothing
# On AVX-512 the rule's planar `MR = 2W, NR = 6` spills; `NR = 3` tiles do not.
_shape_override(::Val{:avx512}, ::Type{ComplexF64}, ::PlanarMethod) = (24, 3, 8)
_shape_override(::Val{:avx512}, ::Type{ComplexF32}, ::PlanarMethod) = (48, 3, 16)
# What the register fit selects on NEON anyway; pinned against menu edits.
_shape_override(::Val{:neon}, ::Type{ComplexF64}, ::PlanarMethod) = (4, 6, 2)
_shape_override(::Val{:neon}, ::Type{ComplexF32}, ::PlanarMethod) = (8, 6, 4)
# AVX2 has 16 registers: at `MV = 1` planar `NR = 6` needs all 16, `NR = 5` needs 14.
_shape_override(::Val{:avx2}, ::Type{ComplexF64}, ::PlanarMethod) = (4, 5, 4)
_shape_override(::Val{:avx2}, ::Type{ComplexF32}, ::PlanarMethod) = (8, 5, 8)
# The AVX2-sized 1m shape, for a caller naming `OneMMethod()` (the fit would
# hand it an AVX-512 shape).
_shape_override(::Val{:avx2}, ::Type{ComplexF64}, ::OneMMethod) = (4, 6, 4)

# Where the `MR = MV*W` rule applies. Not on NEON (2W on 128-bit lanes is no
# better than the fallback), nor complex off AVX-512 (planar's two accumulator
# planes leave AVX2's 16 registers nothing spare).
_rule_applies(::Val{:avx512}, ::RealMethod) = true
_rule_applies(::Val{:avx512}, ::Union{PlanarMethod, OneMMethod}) = true
_rule_applies(::Val{:avx2}, ::RealMethod) = true
_rule_applies(::Val, ::ComplexMethod) = false

# `W` real lanes per register, `MV` A vectors per column.
_rule_shape(vb::Int, ::Type{T}, mv::Int) where {T} =
    (mv * (vb ÷ sizeof(real(T))), NR_DEFAULT, vb ÷ sizeof(real(T)))

# MV = 4 for the real kernel on AVX-512: the MV = 2 tile is front-end bound on
# cores with 2 FMA ports, and 4*6 accumulators + 4 A vectors + 2 still fit 32
# registers. Complex methods keep MV = 2 (planar already spills there).
_rule_mv(::Val{:avx512}, ::RealMethod) = 4
_rule_mv(::Val, ::ComplexMethod) = 2

# AMD's AVX-512 cores double-pump 512-bit FMAs, so their MV = 2 tile is not
# front-end bound and the taller tile only adds edge and store cost.
const _MV4_UNPROFITABLE_CPUS = ("znver4", "znver5")

_profile_mv(profile::TargetProfile, method) = _rule_mv(Val(profile.isa), method)
function _profile_mv(profile::TargetProfile, method::RealMethod)
    mv = _rule_mv(Val(profile.isa), method)
    return (mv == 4 && profile.cpu_name in _MV4_UNPROFITABLE_CPUS) ? 2 : mv
end

# The register shape for `T` under `method` on `profile`: an override row, else
# the rule where it applies (and lands in the menu), else `_fitted_shape`.
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

# Vector registers a planar kernel holds live per K step (two planes of
# accumulators and A vectors, plus 2 broadcasts). Spilling is not monotone in
# it, so it only excludes shapes, never ranks them.
_planar_pressure(MR::Int, NR::Int, W::Int) = 2 * (MR ÷ W) * NR + 2 * (MR ÷ W) + 2

# The shape where no rule applies, and the small-M demotion target. Planar: the
# largest menu shape with `W <= lanes` whose pressure fits the register file
# (16 when unknown). 1m/FMAddSub: the menu head.
_fitted_shape(::TargetProfile, ::Type{T}, ::RealMethod) where {T} = _fallback_shape(T)
_fitted_shape(::TargetProfile, ::Type{T}, method::Union{OneMMethod, FMAddSubMethod}) where {T} =
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
    # Unreachable for any budget >= 16; a slow kernel beats throwing.
    return best === nothing ? last(kernel_shapes(T, method)) : best
end

# ----------------------------------------------------------------------------
# The default kernel for a profile, and the plan-time demotions
# ----------------------------------------------------------------------------

# The default kernel for `T` on `profile`, uncached.
@noinline _kernel_for(profile::TargetProfile, ::Type{T}) where {T} =
    _kernel_from_shape(_derived_shape(profile, T), T, _default_method(T))

# The real MV = 4 shape steps down to its MV = 2 sibling when `m_length` is
# below one tall tile, or below two and the half tile pads to fewer rows. Above
# `2MR` the tall tile's padding excess is under 1.2x, its speed advantage.
@inline _extent_shape(shape::NTuple{3, Int}, ::Type{T}, method, m_length::Int) where {T} = shape

@inline function _extent_shape(shape::NTuple{3, Int}, ::Type{T}, method::RealMethod, m_length::Int) where {T}
    MR, NR, W = shape
    (m_length > 0 && MR == 4 * W && m_length < 2 * MR) || return shape
    half = (2 * W, NR, W)
    half in kernel_shapes(T, method) || return shape
    # Below one tall tile, always: the tall shape would fall to the fitted
    # shape (two steps down) in `_default_shape`.
    m_length < MR && return half
    return cld(m_length, half[1]) * half[1] < cld(m_length, MR) * MR ? half : shape
end

# The real MV = 4 shape steps down to MV = 2 when C's unit-stride run along M
# (`run`) can't fill the tall tile's slivers but fills a half tile: a sliver
# that is not unit-stride in C takes the scattered store, which costs more than
# the tall tile gains. Below one half tile both scatter, and the tall one stays.
@inline _store_shape(shape::NTuple{3, Int}, ::Type{T}, method, m_length::Int, run::Int) where {T} = shape

@inline function _store_shape(shape::NTuple{3, Int}, ::Type{T}, method::RealMethod, m_length::Int, run::Int) where {T}
    MR, NR, W = shape
    (MR == 4 * W && m_length != run && run % MR != 0 && run >= 2 * W) || return shape
    half = (2 * W, NR, W)
    return half in kernel_shapes(T, method) ? half : shape
end

# Small-M demotion for complex `T` on AVX-512, where the planar fitted shape is
# the spilling `MR = 2W` tile: the native-width FMAddSub shape that pads
# `m_length` least, ties by the larger tile. `_small_m_candidates` is the
# cached, host-dependent half (empty where the rule does not apply).
_small_m_candidates(::Val, ::TargetProfile, ::Type) = NTuple{3, Int}[]
function _small_m_candidates(::Val{:avx512}, profile::TargetProfile, ::Type{T}) where {T <: Complex}
    lanes = profile.vector_bytes ÷ sizeof(real(T))
    return [shape for shape in kernel_shapes(T, FMAddSubMethod()) if shape[3] == lanes]
end

function _small_m_shape(candidates::Vector{NTuple{3, Int}}, m_length::Int)
    best = nothing
    for shape in candidates
        MR, NR, _ = shape
        key = (-(cld(m_length, MR) * MR), MR * NR)
        if best === nothing || key > best[1]
            best = (key, shape)
        end
    end
    return best === nothing ? nothing : best[2]
end

# Run-length demotion (real `T` only): the vectorized store needs every
# register sliver unit-stride in C, i.e. `m_length == run || run % m_tile == 0`
# (not `m_tile <= run`). Otherwise demote to the largest menu shape whose
# `m_tile` divides `run`, unless `k_length` is deep enough that the smaller
# kernel's cost dominates.
@inline function _demote_shape_for_run(
        ::Type{T}, shape::NTuple{3, Int}, method, run::Int, m_length::Int, k_length::Int
    ) where {T}
    target = _run_demotion_target(T, shape[1], method, run, m_length, k_length)
    return target === nothing ? shape : target
end

@inline function _demote_shape_for_run(
        ::Type{T}, shape::NTuple{3, Int}, method::_MixedMethod, run::Int, m_length::Int, k_length::Int
    ) where {T}
    m, _, r = _real_problem(method, m_length, 0, run)
    real_shape = _demote_shape_for_run(real(T), _real_shape(method, shape), RealMethod(), r, m, k_length)
    return _mixed_shape(method, real_shape)
end

function _run_demotion_target(::Type{T}, m_tile::Int, method, run::Int, m_length::Int, k_length::Int) where {T}
    T <: Real || return nothing
    kmax = T === Float64 ? _RUN_DEMOTE_KMAX_F64 : _RUN_DEMOTE_KMAX_F32
    k_length > kmax && return nothing
    m_length == run && return nothing
    run % m_tile == 0 && return nothing
    best = nothing
    for shape in kernel_shapes(T, method)
        m = shape[1]
        if run % m == 0 && (best === nothing || m > best[1])
            best = shape
        end
    end
    return best
end

# Deepest `k_length` at which run-length demotion still wins.
const _RUN_DEMOTE_KMAX_F64 = 32
const _RUN_DEMOTE_KMAX_F32 = 64
