# Kernel selection: the microkernel `plan_contract` builds when the caller does
# not name one, a pure function of a `TargetProfile` and `T`.

# Kernel types, unparameterised, name the complex-arithmetic schemes here.
# `OneMKernel` is chosen only by naming the kernel; `FMAddSubKernel` also on
# AVX2 and by the AVX-512 small-M demotion (`select_shape`).
default_kernel_type(::Type{<:Real}) = SIMDKernel
default_kernel_type(::Type{<:Complex}) = PlanarKernel

const MixedKernel = Union{Type{<:ComplexRealKernel}, Type{<:RealComplexKernel}}

# The kernel type for compute type `T` and (oriented) operand eltypes: a
# mixed-domain kernel when exactly one operand is real.
default_kernel_type(::Type{T}, ::Type{TA}, ::Type{TB}) where {T, TA, TB} =
    default_kernel_type(T)
default_kernel_type(::Type{T}, ::Type{<:Complex}, ::Type{<:Real}) where {T <: Complex} =
    ComplexRealKernel
default_kernel_type(::Type{T}, ::Type{<:Real}, ::Type{<:Complex}) where {T <: Complex} =
    RealComplexKernel

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
# FMAddSub starts from 1m's shapes (same accumulator layout).
const KERNEL_SHAPES_C64_FMADDSUB = ((12, 8, 8), (8, 8, 8), (4, 6, 4))
const KERNEL_SHAPES_C32_FMADDSUB = ((24, 8, 16), (16, 8, 16), (8, 6, 8))

"""
    kernel_shapes(T, K = default_kernel_type(T)) -> NTuple{<:Any,NTuple{3,Int}}

The closed menu of `(MR, NR, W)` register shapes the engine may build for
element type `T` with kernel type `K` (empty where it builds none).
"""
kernel_shapes(::Type{T}) where {T} = kernel_shapes(T, default_kernel_type(T))
kernel_shapes(::Type, ::Type) = ()
kernel_shapes(::Type{Float64}, ::Type{<:SIMDKernel}) = KERNEL_SHAPES_F64
kernel_shapes(::Type{Float32}, ::Type{<:SIMDKernel}) = KERNEL_SHAPES_F32
kernel_shapes(::Type{ComplexF64}, ::Type{<:PlanarKernel}) = KERNEL_SHAPES_C64_PLANAR
kernel_shapes(::Type{ComplexF64}, ::Type{<:OneMKernel}) = KERNEL_SHAPES_C64_ONEM
kernel_shapes(::Type{ComplexF32}, ::Type{<:PlanarKernel}) = KERNEL_SHAPES_C32_PLANAR
kernel_shapes(::Type{ComplexF32}, ::Type{<:OneMKernel}) = KERNEL_SHAPES_C32_ONEM
kernel_shapes(::Type{ComplexF64}, ::Type{<:FMAddSubKernel}) = KERNEL_SHAPES_C64_FMADDSUB
kernel_shapes(::Type{ComplexF32}, ::Type{<:FMAddSubKernel}) = KERNEL_SHAPES_C32_FMADDSUB
# The mixed menus are the real menu of `real(T)`, mapped by `mixed_shape`.
kernel_shapes(::Type{T}, K::MixedKernel) where {T <: Union{ComplexF32, ComplexF64}} =
    map(s -> mixed_shape(K, s), kernel_shapes(real(T), SIMDKernel))

# A real inner shape to the mixed kernel's `(MR, NR, W)`: complex A rows and
# complex B columns each span two reals of the inner kernel.
mixed_shape(::Type{<:ComplexRealKernel}, (MR, NR, W)::NTuple{3, Int}) = (MR ÷ 2, NR, W)
mixed_shape(::Type{<:RealComplexKernel}, (MR, NR, W)::NTuple{3, Int}) = (MR, NR ÷ 2, W)

# The real problem the inner kernel sees: `m_length` and C's unit-stride M run in reals.
real_problem(::Type{<:ComplexRealKernel}, m_length::Int, run::Int) = (2 * m_length, 2 * run)
real_problem(::Type{<:RealComplexKernel}, m_length::Int, run::Int) = (m_length, run)

# An unrolled `shape === menu[i] ? ... :` ladder over the menu whose branches
# are built from literals, so each is concrete; a shape outside the menu throws.
# Each branch builds the kernel, or a `Val` of its concrete type when `valonly`.
function menu_ladder(::Type{T}, ::Type{K}, valonly::Bool) where {T, K}
    menu = kernel_shapes(T, K)
    isempty(menu) && return :(throw_no_kernel(shape, T, K))
    ex = :(throw_shape_not_in_menu(shape, T, K))
    for (MR, NR, W) in reverse(menu)
        arm = valonly ? :(Val{$(K{MR, NR, T, W})}()) : :(K(Val($MR), Val($NR), T, Val($W)))
        ex = :(shape === ($MR, $NR, $W) ? $arm : $ex)
    end
    return ex
end

# The kernel of type `K` for `T` at a menu `shape` (one dynamic dispatch per plan).
kernel_from_shape(shape::Tuple{Int, Int, Int}, ::Type{T}) where {T} =
    kernel_from_shape(shape, T, default_kernel_type(T))
@generated kernel_from_shape(shape::Tuple{Int, Int, Int}, ::Type{T}, ::Type{K}) where {T, K} =
    menu_ladder(T, K, false)
kernel_from_shape(shape::Tuple{Int, Int, Int}, ::Type{T}, ::Val{K}) where {T, K} =
    kernel_from_shape(shape, T, K)

# The concrete kernel type at a menu `shape`, as a `Val`: a singleton crosses
# `plan_contract`'s kernel barrier without boxing, and dispatch on it hits the
# fast method cache, which a bare `Type` argument misses.
@generated menu_val(shape::Tuple{Int, Int, Int}, ::Type{T}, ::Type{K}) where {T, K} =
    menu_ladder(T, K, true)

@noinline throw_no_kernel(shape, ::Type{T}, K) where {T} = throw(
    ArgumentError(
        "no microkernel is available for $T at shape $shape with $(K). " *
            "Pass an explicit `kernel = ...` to plan_contract to use a kernel this " *
            "engine does not pick itself."
    )
)

@noinline throw_shape_not_in_menu(shape, ::Type{T}, K) where {T} = throw(
    ArgumentError(
        "shape $shape is not in the $(K) menu for $T, " *
            "$(kernel_shapes(T, K)); only menu shapes are compiled"
    )
)

# ----------------------------------------------------------------------------
# Shape resolution from the detected hardware
# ----------------------------------------------------------------------------

# Per-ISA shapes, consulted first. None for real types: the rule below is the optimum.
shape_override(isa::Symbol, ::Type{T}) where {T} = shape_override(isa, T, default_kernel_type(T))
function shape_override(isa::Symbol, ::Type{T}, ::Type{K}) where {T, K}
    if K <: PlanarKernel && T === ComplexF64
        # On AVX-512 the rule's planar `MR = 2W, NR = 6` spills; `NR = 3` tiles
        # do not. AVX2 has 16 registers: at `MV = 1` planar `NR = 6` needs all
        # 16, `NR = 5` needs 14.
        isa === :avx512 && return (24, 3, 8)
        isa === :avx2 && return (4, 5, 4)
    elseif K <: PlanarKernel && T === ComplexF32
        isa === :avx512 && return (48, 3, 16)
        isa === :avx2 && return (8, 5, 8)
    elseif K <: Union{OneMKernel, FMAddSubKernel}
        # The AVX2-sized lane-pair shapes (the fit would hand them an AVX-512 one).
        isa === :avx2 && return T === ComplexF64 ? (4, 6, 4) : (8, 6, 8)
    end
    return nothing
end

# Where the `MR = MV*W` rule applies. Not on NEON (2W on 128-bit lanes is no
# better than the fallback), nor complex off AVX-512 (planar's two accumulator
# planes leave AVX2's 16 registers nothing spare).
rule_applies(isa::Symbol, ::Type{K}) where {K} =
    isa === :avx512 ? K <: Union{SIMDKernel, PlanarKernel, OneMKernel} : isa === :avx2 && K <: SIMDKernel

# `W` real lanes per register, `MV` A vectors per column.
rule_shape(vb::Int, ::Type{T}, mv::Int) where {T} =
    (mv * (vb ÷ sizeof(real(T))), NR_DEFAULT, vb ÷ sizeof(real(T)))

# MV = 4 for the real kernel on AVX-512: the MV = 2 tile is front-end bound on
# cores with 2 FMA ports, and 4*6 accumulators + 4 A vectors + 2 still fit 32
# registers. Complex kernels keep MV = 2 (planar already spills there).
rule_mv(isa::Symbol, ::Type{K}) where {K} = isa === :avx512 && K <: SIMDKernel ? 4 : 2

# Where 512-bit FMAs are double-pumped the MV = 2 tile is not front-end bound,
# and the taller tile only adds edge and store cost.
function profile_mv(profile::TargetProfile, ::Type{K}) where {K}
    mv = rule_mv(profile.isa, K)
    return (mv == 4 && profile.double_pumped) ? 2 : mv
end

# The register shape for `T` with kernel type `K` on `profile`: an override
# row, else the rule where it applies (and lands in the menu), else `fitted_shape`.
derived_shape(profile::TargetProfile, ::Type{T}) where {T} =
    derived_shape(profile, T, default_kernel_type(T))

function derived_shape(profile::TargetProfile, ::Type{T}, ::Type{K}) where {T, K}
    ovr = shape_override(profile.isa, T, K)
    ovr === nothing || return ovr
    vb = profile.vector_bytes
    if rule_applies(profile.isa, K) && vb > 0
        shape = rule_shape(vb, T, profile_mv(profile, K))
        shape in kernel_shapes(T, K) && return shape
    end
    return fitted_shape(profile, T, K)
end

# Vector registers a planar kernel holds live per K step (two planes of
# accumulators and A vectors, plus 2 broadcasts). Spilling is not monotone in
# it, so it only excludes shapes, never ranks them.
planar_pressure(MR::Int, NR::Int, W::Int) = 2 * (MR ÷ W) * NR + 2 * (MR ÷ W) + 2

# The shape where no rule applies, and the small-M demotion target. Planar: the
# largest menu shape with `W <= lanes` whose pressure fits the register file
# (16 when unknown). 1m/FMAddSub: the menu head.
fitted_shape(::TargetProfile, ::Type{T}, ::Type{<:SIMDKernel}) where {T} = fallback_shape(T)
fitted_shape(::TargetProfile, ::Type{T}, ::Type{K}) where {T, K <: Union{OneMKernel, FMAddSubKernel}} =
    first(kernel_shapes(T, K))

function fitted_shape(profile::TargetProfile, ::Type{T}, ::Type{K}) where {T, K <: PlanarKernel}
    lanes = vector_lanes(profile, real(T))
    budget = profile.nregisters > 0 ? profile.nregisters : 16
    best = nothing
    for shape in kernel_shapes(T, K)
        MR, NR, W = shape
        (W <= lanes && MR % W == 0) || continue
        planar_pressure(MR, NR, W) <= budget || continue
        # Largest logical tile wins; ties by the wider vector.
        if best === nothing || (MR * NR, W) > (best[1] * best[2], best[3])
            best = shape
        end
    end
    # Unreachable for any budget >= 16; a slow kernel beats throwing.
    return best === nothing ? last(kernel_shapes(T, K)) : best
end

# ----------------------------------------------------------------------------
# The automatic shape for a contraction
# ----------------------------------------------------------------------------

"""
    select_shape(T, K, m_length, k_length, run) -> (shape, Val(kernel_type))

The automatic register shape `(MR, NR, W)`, and the kernel type it is built
with, for compute type `T` and kernel type `K = default_kernel_type(T, TA, TB)`,
extents `m_length`/`k_length` and C's unit-stride run along M (`run`;
`m_length` when no layout is known). The steps, in order:

 1. The host's shape for `T` (`derived_shape`); for complex `T` on AVX2 the
    FMAddSub one, where C's rows take its vector store.
 2. Extent (real): the MV = 4 shape steps down to MV = 2 where `m_length`
    pads less (`extent_shape`).
 3. Small M: an `m_length` below one tile takes the fitted shape, or for
    complex `T` on AVX-512 the FMAddSub shape that pads least (`small_m_shape`).
 4. Run (real): the shape shrinks until C's run fills its slivers (`fit_to_run`).

A mixed kernel runs the steps on the real problem of `real(T)` and maps the
shape. Plain values, so `plan_contract` never holds a menu-wide kernel Union;
the kernel type as a `Val`, since inference widens a type inside a returned
tuple to `UnionAll`.
"""
@inline function select_shape(::Type{T}, ::Type{K}, m_length::Int, k_length::Int, run::Int) where {T, K}
    profile = target_profile()
    if K <: PlanarKernel && profile.isa === :avx2
        # FMAddSub's one accumulator plane fits `NR = 6` with a register to spare
        # (planar's two spill), but its scalar store is the slower one.
        shape = derived_shape(profile, T, FMAddSubKernel)
        (run == m_length || run % shape[1] == 0) && return (shape, Val(FMAddSubKernel))
    end
    shape = extent_shape(derived_shape(profile, T, K), T, K, m_length)
    if m_length > 0 && m_length < shape[1]
        # Static, so a real `T`'s kernel type stays concrete.
        if T <: Complex
            small = small_m_shape(profile, T, m_length)
            small === nothing || return (small, Val(FMAddSubKernel))
        end
        shape = fitted_shape(profile, T, K)
    end
    return (fit_to_run(shape, T, K, m_length, k_length, run), Val(K))
end

@inline function select_shape(::Type{T}, K::MixedKernel, m_length::Int, k_length::Int, run::Int) where {T}
    m, r = real_problem(K, m_length, run)
    shape, _ = select_shape(real(T), SIMDKernel, m, k_length, r)
    return (mixed_shape(K, shape), Val(K))
end

# The real MV = 4 shape steps down to its MV = 2 sibling when `m_length` is
# below one tall tile, or below two and the half tile pads to fewer rows. Above
# `2MR` the tall tile's padding excess is under 1.2x, its speed advantage.
@inline extent_shape(shape::NTuple{3, Int}, ::Type{T}, ::Type, m_length::Int) where {T} = shape

@inline function extent_shape(shape::NTuple{3, Int}, ::Type{T}, ::Type{K}, m_length::Int) where {T, K <: SIMDKernel}
    MR, NR, W = shape
    (m_length > 0 && MR == 4 * W && m_length < 2 * MR) || return shape
    half = (2 * W, NR, W)
    half in kernel_shapes(T, K) || return shape
    # Below one tall tile, always: the tall shape would fall to the fitted
    # shape (two steps down) at the small-M step.
    m_length < MR && return half
    return cld(m_length, half[1]) * half[1] < cld(m_length, MR) * MR ? half : shape
end

# Small-M demotion for complex `T` on AVX-512, where the planar fitted shape is
# the spilling `MR = 2W` tile: the native-width FMAddSub shape that pads
# `m_length` least, ties by the larger tile.
function small_m_shape(profile::TargetProfile, ::Type{T}, m_length::Int) where {T}
    profile.isa === :avx512 || return nothing
    lanes = vector_lanes(profile, real(T))
    best = nothing
    for shape in kernel_shapes(T, FMAddSubKernel)
        MR, NR, W = shape
        W == lanes || continue
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
@inline fit_to_run(shape::NTuple{3, Int}, ::Type{T}, ::Type, m_length::Int, k_length::Int, run::Int) where {T} = shape

@inline function fit_to_run(shape::NTuple{3, Int}, ::Type{T}, ::Type{K}, m_length::Int, k_length::Int, run::Int) where {T, K <: SIMDKernel}
    m_length == run && return shape
    MR, NR, W = shape
    half = (2 * W, NR, W)
    if MR == 4 * W && run % MR != 0 && run >= 2 * W && half in kernel_shapes(T, K)
        shape = half
    end
    kmax = T === Float64 ? RUN_DEMOTE_KMAX_F64 : RUN_DEMOTE_KMAX_F32
    (k_length > kmax || run % shape[1] == 0) && return shape
    best = nothing
    for s in kernel_shapes(T, K)
        run % s[1] == 0 && (best === nothing || s[1] > best[1]) && (best = s)
    end
    return something(best, shape)
end

# Deepest `k_length` at which the divisor step still wins.
const RUN_DEMOTE_KMAX_F64 = 32
const RUN_DEMOTE_KMAX_F32 = 64
