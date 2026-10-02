# Kernel selection: the microkernel `plan_contract` builds when the caller does
# not name one. Everything here is a pure function of a `TargetProfile` and `T`;
# src/planning/defaults.jl caches the host-dependent results per eltype, since
# the `Val(profile.isa)` dispatch below is dynamic.

# `OneMMethod`/`FMAddSubMethod` are chosen only by naming the kernel (FMAddSub
# also by the AVX-512 small-M demotion, `small_m_shape`).
default_method(::Type{<:Real}) = RealMethod()
default_method(::Type{<:Complex}) = PlanarMethod()

const MixedMethod = Union{ComplexRealMethod, RealComplexMethod}

# The method for compute type `T` and (oriented) operand eltypes: a mixed-domain
# method when exactly one operand is real.
default_method(::Type{T}, ::Type{TA}, ::Type{TB}) where {T, TA, TB} =
    default_method(T)
default_method(::Type{T}, ::Type{<:Complex}, ::Type{<:Real}) where {T <: Complex} =
    ComplexRealMethod()
default_method(::Type{T}, ::Type{<:Real}, ::Type{<:Complex}) where {T <: Complex} =
    RealComplexMethod()

const NR_DEFAULT = 6

# An (8, 6) tile at one 256-bit register's lane width: the shape wherever no rule applies.
fallback_shape(::Type{T}) where {T} = (8, NR_DEFAULT, default_lanewidth(real(T)))

# Closed menus of `(MR, NR, W)` shapes, so the compiled specializations stay
# bounded. Complex `MR` counts complex rows and `W` real lanes; 1m runs a real
# kernel of `2MR` rows, so only `2MR` must be a multiple of `W`. The trailing
# `MV = 1` planar entries guarantee `fitted_shape` a fit on any register file
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
    kernel_shapes(T, method::KernelMethod = default_method(T)) -> NTuple{<:Any,NTuple{3,Int}}

The closed menu of `(MR, NR, W)` register shapes the engine may build for
element type `T` under `method`.
"""
kernel_shapes(::Type{T}) where {T} = kernel_shapes(T, default_method(T))
kernel_shapes(::Type{Float64}, ::RealMethod) = KERNEL_SHAPES_F64
kernel_shapes(::Type{Float32}, ::RealMethod) = KERNEL_SHAPES_F32
kernel_shapes(::Type{ComplexF64}, ::PlanarMethod) = KERNEL_SHAPES_C64_PLANAR
kernel_shapes(::Type{ComplexF64}, ::OneMMethod) = KERNEL_SHAPES_C64_ONEM
kernel_shapes(::Type{ComplexF32}, ::PlanarMethod) = KERNEL_SHAPES_C32_PLANAR
kernel_shapes(::Type{ComplexF32}, ::OneMMethod) = KERNEL_SHAPES_C32_ONEM
kernel_shapes(::Type{ComplexF64}, ::FMAddSubMethod) = KERNEL_SHAPES_C64_FMADDSUB
kernel_shapes(::Type{ComplexF32}, ::FMAddSubMethod) = KERNEL_SHAPES_C32_FMADDSUB
# The mixed menus are the real menu of `real(T)`, mapped by `mixed_shape`.
kernel_shapes(::Type{T}, method::MixedMethod) where {T <: Union{ComplexF32, ComplexF64}} =
    map(s -> mixed_shape(method, s), kernel_shapes(real(T), RealMethod()))

# A real inner shape to the mixed kernel's `(MR, NR, W)`: complex A rows and
# complex B columns each span two reals of the inner kernel.
mixed_shape(::ComplexRealMethod, (MR, NR, W)::NTuple{3, Int}) = (MR ÷ 2, NR, W)
mixed_shape(::RealComplexMethod, (MR, NR, W)::NTuple{3, Int}) = (MR, NR ÷ 2, W)

# The real problem the inner kernel sees: `m_length` and C's unit-stride M run in reals.
real_problem(::ComplexRealMethod, m_length::Int, run::Int) = (2 * m_length, 2 * run)
real_problem(::RealComplexMethod, m_length::Int, run::Int) = (m_length, run)

# The kernel type implementing `method` for `T`, or `nothing` (exactly the
# pairs with a menu).
kernel_type(::RealMethod, ::Type{<:Union{Float32, Float64}}) = SIMDKernel
kernel_type(::PlanarMethod, ::Type{<:Union{ComplexF32, ComplexF64}}) = PlanarKernel
kernel_type(::OneMMethod, ::Type{<:Union{ComplexF32, ComplexF64}}) = OneMKernel
kernel_type(::FMAddSubMethod, ::Type{<:Union{ComplexF32, ComplexF64}}) = FMAddSubKernel
kernel_type(::ComplexRealMethod, ::Type{<:Union{ComplexF32, ComplexF64}}) = ComplexRealKernel
kernel_type(::RealComplexMethod, ::Type{<:Union{ComplexF32, ComplexF64}}) = RealComplexKernel
kernel_type(::Any, ::Type) = nothing

# An unrolled `shape === menu[i] ? ... :` ladder over the menu whose branches
# are built from literals, so each is concrete; a shape outside the menu throws.
# Each branch builds the kernel, or `Val(shape)` when `valonly`.
function menu_ladder(::Type{T}, method, valonly::Bool) where {T}
    K = kernel_type(method, T)
    K === nothing && return :(throw_no_kernel(shape, T, method))
    ex = :(throw_shape_not_in_menu(shape, T, method))
    for (MR, NR, W) in reverse(kernel_shapes(T, method))
        arm = valonly ? :(Val(($MR, $NR, $W))) : :($K(Val($MR), Val($NR), T, Val($W)))
        ex = :(shape === ($MR, $NR, $W) ? $arm : $ex)
    end
    return ex
end

# The `method` kernel for `T` at a menu `shape` (one dynamic dispatch per plan).
kernel_from_shape(shape::Tuple{Int, Int, Int}, ::Type{T}) where {T} =
    kernel_from_shape(shape, T, default_method(T))
@generated kernel_from_shape(shape::Tuple{Int, Int, Int}, ::Type{T}, method::M) where {T, M} =
    menu_ladder(T, M.instance, false)

# `Val(shape)` for a menu `shape`: a singleton, so it crosses `plan_contract`'s
# kernel barrier without boxing.
@generated menu_val(shape::Tuple{Int, Int, Int}, ::Type{T}, method::M) where {T, M} =
    menu_ladder(T, M.instance, true)

@noinline throw_no_kernel(shape, ::Type{T}, method) where {T} = throw(
    ArgumentError(
        "no microkernel is available for $T at shape $shape under $(method). " *
            "Pass an explicit `kernel = ...` to plan_contract to use a kernel this " *
            "engine does not pick itself."
    )
)

@noinline throw_shape_not_in_menu(shape, ::Type{T}, method) where {T} = throw(
    ArgumentError(
        "shape $shape is not in the $(method) menu for $T, " *
            "$(kernel_shapes(T, method)); only menu shapes are compiled"
    )
)

# ----------------------------------------------------------------------------
# Shape resolution from the detected hardware
# ----------------------------------------------------------------------------

# Per-ISA shapes, consulted first. None for real types: the rule below is the optimum.
shape_override(key::Val, ::Type{T}) where {T} = shape_override(key, T, default_method(T))
shape_override(::Val, ::Type, ::KernelMethod) = nothing
# On AVX-512 the rule's planar `MR = 2W, NR = 6` spills; `NR = 3` tiles do not.
shape_override(::Val{:avx512}, ::Type{ComplexF64}, ::PlanarMethod) = (24, 3, 8)
shape_override(::Val{:avx512}, ::Type{ComplexF32}, ::PlanarMethod) = (48, 3, 16)
# AVX2 has 16 registers: at `MV = 1` planar `NR = 6` needs all 16, `NR = 5` needs 14.
shape_override(::Val{:avx2}, ::Type{ComplexF64}, ::PlanarMethod) = (4, 5, 4)
shape_override(::Val{:avx2}, ::Type{ComplexF32}, ::PlanarMethod) = (8, 5, 8)
# The AVX2-sized 1m shape, for a caller naming `OneMMethod()` (the fit would
# hand it an AVX-512 shape).
shape_override(::Val{:avx2}, ::Type{ComplexF64}, ::OneMMethod) = (4, 6, 4)

# Where the `MR = MV*W` rule applies. Not on NEON (2W on 128-bit lanes is no
# better than the fallback), nor complex off AVX-512 (planar's two accumulator
# planes leave AVX2's 16 registers nothing spare).
rule_applies(::Val{:avx512}, ::RealMethod) = true
rule_applies(::Val{:avx512}, ::Union{PlanarMethod, OneMMethod}) = true
rule_applies(::Val{:avx2}, ::RealMethod) = true
rule_applies(::Val, ::KernelMethod) = false

# `W` real lanes per register, `MV` A vectors per column.
rule_shape(vb::Int, ::Type{T}, mv::Int) where {T} =
    (mv * (vb ÷ sizeof(real(T))), NR_DEFAULT, vb ÷ sizeof(real(T)))

# MV = 4 for the real kernel on AVX-512: the MV = 2 tile is front-end bound on
# cores with 2 FMA ports, and 4*6 accumulators + 4 A vectors + 2 still fit 32
# registers. Complex methods keep MV = 2 (planar already spills there).
rule_mv(::Val{:avx512}, ::RealMethod) = 4
rule_mv(::Val, ::KernelMethod) = 2

# AMD's AVX-512 cores double-pump 512-bit FMAs, so their MV = 2 tile is not
# front-end bound and the taller tile only adds edge and store cost.
const _MV4_UNPROFITABLE_CPUS = ("znver4", "znver5")

profile_mv(profile::TargetProfile, method) = rule_mv(Val(profile.isa), method)
function profile_mv(profile::TargetProfile, method::RealMethod)
    mv = rule_mv(Val(profile.isa), method)
    return (mv == 4 && profile.cpu_name in _MV4_UNPROFITABLE_CPUS) ? 2 : mv
end

# The register shape for `T` under `method` on `profile`: an override row, else
# the rule where it applies (and lands in the menu), else `fitted_shape`.
derived_shape(profile::TargetProfile, ::Type{T}) where {T} =
    derived_shape(profile, T, default_method(T))

function derived_shape(profile::TargetProfile, ::Type{T}, method) where {T}
    key = Val(profile.isa)
    ovr = shape_override(key, T, method)
    ovr === nothing || return ovr
    vb = profile.vector_bytes
    if rule_applies(key, method) && vb > 0
        shape = rule_shape(vb, T, profile_mv(profile, method))
        shape in kernel_shapes(T, method) && return shape
    end
    return fitted_shape(profile, T, method)
end

# Vector registers a planar kernel holds live per K step (two planes of
# accumulators and A vectors, plus 2 broadcasts). Spilling is not monotone in
# it, so it only excludes shapes, never ranks them.
planar_pressure(MR::Int, NR::Int, W::Int) = 2 * (MR ÷ W) * NR + 2 * (MR ÷ W) + 2

# The shape where no rule applies, and the small-M demotion target. Planar: the
# largest menu shape with `W <= lanes` whose pressure fits the register file
# (16 when unknown). 1m/FMAddSub: the menu head.
fitted_shape(::TargetProfile, ::Type{T}, ::RealMethod) where {T} = fallback_shape(T)
fitted_shape(::TargetProfile, ::Type{T}, method::Union{OneMMethod, FMAddSubMethod}) where {T} =
    first(kernel_shapes(T, method))

function fitted_shape(profile::TargetProfile, ::Type{T}, method::PlanarMethod) where {T}
    R = real(T)
    vb = profile.vector_bytes
    lanes = vb > 0 ? vb ÷ sizeof(R) : default_lanewidth(R)
    budget = profile.nregisters > 0 ? profile.nregisters : 16
    best = nothing
    for shape in kernel_shapes(T, method)
        MR, NR, W = shape
        (W <= lanes && MR % W == 0) || continue
        planar_pressure(MR, NR, W) <= budget || continue
        # Largest logical tile wins; ties by the wider vector.
        if best === nothing || (MR * NR, W) > (best[1] * best[2], best[3])
            best = shape
        end
    end
    # Unreachable for any budget >= 16; a slow kernel beats throwing.
    return best === nothing ? last(kernel_shapes(T, method)) : best
end

# ----------------------------------------------------------------------------
# The automatic shape for a contraction
# ----------------------------------------------------------------------------

"""
    select_shape(T, method, m_length, k_length, run) -> (shape, method)

The automatic register shape `(MR, NR, W)`, and the method it runs under, for
compute type `T` under `method = default_method(T, TA, TB)`, extents
`m_length`/`k_length` and C's unit-stride run along M (`run`; `m_length` when
no layout is known). The steps, in order:

 1. The host's shape for `T` (`derived_shape`, cached in `ResolvedDefaults`).
 2. Extent (real): the MV = 4 shape steps down to MV = 2 where `m_length`
    pads less (`extent_shape`).
 3. Small M: an `m_length` below one tile takes the fitted shape, or for
    complex `T` on AVX-512 the FMAddSub shape that pads least (`small_m_shape`).
 4. Run (real): the shape shrinks until C's run fills its slivers (`fit_to_run`).

A mixed method runs the steps on the real problem of `real(T)` and maps the
shape. Plain values, so `plan_contract` never holds a menu-wide kernel Union.
"""
@inline function select_shape(::Type{T}, method, m_length::Int, k_length::Int, run::Int) where {T}
    d = _resolved_defaults(T)
    shape = extent_shape(d.shape, T, method, m_length)
    if m_length > 0 && m_length < shape[1]
        # Static, so a real `T`'s method stays concrete.
        if T <: Complex
            small = small_m_shape(d.small_m, m_length)
            small === nothing || return (small, FMAddSubMethod())
        end
        shape = d.fitted
    end
    return (fit_to_run(shape, T, method, m_length, k_length, run), method)
end

@inline function select_shape(::Type{T}, method::MixedMethod, m_length::Int, k_length::Int, run::Int) where {T}
    m, r = real_problem(method, m_length, run)
    shape, _ = select_shape(real(T), RealMethod(), m, k_length, r)
    return (mixed_shape(method, shape), method)
end

# The real MV = 4 shape steps down to its MV = 2 sibling when `m_length` is
# below one tall tile, or below two and the half tile pads to fewer rows. Above
# `2MR` the tall tile's padding excess is under 1.2x, its speed advantage.
@inline extent_shape(shape::NTuple{3, Int}, ::Type{T}, method, m_length::Int) where {T} = shape

@inline function extent_shape(shape::NTuple{3, Int}, ::Type{T}, method::RealMethod, m_length::Int) where {T}
    MR, NR, W = shape
    (m_length > 0 && MR == 4 * W && m_length < 2 * MR) || return shape
    half = (2 * W, NR, W)
    half in kernel_shapes(T, method) || return shape
    # Below one tall tile, always: the tall shape would fall to the fitted
    # shape (two steps down) at the small-M step.
    m_length < MR && return half
    return cld(m_length, half[1]) * half[1] < cld(m_length, MR) * MR ? half : shape
end

# Small-M demotion for complex `T` on AVX-512, where the planar fitted shape is
# the spilling `MR = 2W` tile: the native-width FMAddSub shape that pads
# `m_length` least, ties by the larger tile. `small_m_candidates` is the
# cached, host-dependent half (empty where the rule does not apply).
small_m_candidates(::Val, ::TargetProfile, ::Type) = NTuple{3, Int}[]
function small_m_candidates(::Val{:avx512}, profile::TargetProfile, ::Type{T}) where {T <: Complex}
    lanes = profile.vector_bytes ÷ sizeof(real(T))
    return [shape for shape in kernel_shapes(T, FMAddSubMethod()) if shape[3] == lanes]
end

function small_m_shape(candidates::Vector{NTuple{3, Int}}, m_length::Int)
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

# A register sliver takes the vectorized store only if it is unit-stride in C:
# `m_length == run || run % MR == 0` (not `MR <= run`); otherwise it takes the
# scattered one. Real `T` only, two steps in this order:
#  - the MV = 4 shape steps down to MV = 2 when the run fills a half tile but
#    not the tall tile's slivers (below one half tile both scatter, and the tall
#    one stays);
#  - at a shallow `k_length`, where the smaller kernel's cost does not dominate,
#    the largest menu shape whose `MR` divides the run.
@inline fit_to_run(shape::NTuple{3, Int}, ::Type{T}, method, m_length::Int, k_length::Int, run::Int) where {T} = shape

@inline function fit_to_run(shape::NTuple{3, Int}, ::Type{T}, method::RealMethod, m_length::Int, k_length::Int, run::Int) where {T}
    m_length == run && return shape
    MR, NR, W = shape
    half = (2 * W, NR, W)
    if MR == 4 * W && run % MR != 0 && run >= 2 * W && half in kernel_shapes(T, method)
        shape = half
    end
    kmax = T === Float64 ? _RUN_DEMOTE_KMAX_F64 : _RUN_DEMOTE_KMAX_F32
    (k_length > kmax || run % shape[1] == 0) && return shape
    best = nothing
    for s in kernel_shapes(T, method)
        run % s[1] == 0 && (best === nothing || s[1] > best[1]) && (best = s)
    end
    return something(best, shape)
end

# Deepest `k_length` at which the divisor step still wins.
const _RUN_DEMOTE_KMAX_F64 = 32
const _RUN_DEMOTE_KMAX_F32 = 64
