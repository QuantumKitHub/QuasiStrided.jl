# What every microkernel shares. Each kernel file implements only
# `zero_accumulator`, `accumulate` and `store_tile!`, with one contract:
#   * `accumulate(kernel, acc, packed_a, packed_b, kc)` adds `kc` K steps to
#     `acc`; `kc == 0` returns `acc` without reading the panels.
#   * `store_tile!(dest, acc, alpha, beta, kernel)` writes `alpha*acc + beta*C`
#     over the valid rectangle only: `alpha == 0` never reads `acc`, `beta == 0`
#     never reads old `C`, padding lanes are never read.
# The vector kernels match `ScalarKernel` only to within rounding (FMA grouping
# differs), never bitwise.

# Kernels wrapping a descriptor as `.descriptor`; the accessors forward once here.
abstract type DescriptorKernel{MR, NR, T} end

# Which complex-arithmetic method a kernel implements, as singletons so blocking
# and the shape menus dispatch on it. No method ranking is hardcoded: planar is
# the default, 1m and fmaddsub are used only when named (plus fmaddsub for the
# AVX-512 small-M demotion, `_small_m_shape` in src/planning/kernel_selection.jl).
abstract type ComplexMethod end
struct RealMethod <: ComplexMethod end      # what a real kernel reports
struct PlanarMethod <: ComplexMethod end    # split re/im planes, 4 real FMAs per MAC
struct OneMMethod <: ComplexMethod end      # Van Zee's 1m: a real 2mr x nr kernel
struct FMAddSubMethod <: ComplexMethod end  # interleaved A, x86 `vfmaddsub`

# Reals per packed A / B element. Blocking divides the real `mc`/`nc` by these,
# so every method gets the same packed byte budget.
a_reals(::RealMethod) = 1
b_reals(::RealMethod) = 1
a_reals(::PlanarMethod) = 2
b_reals(::PlanarMethod) = 2
a_reals(::OneMMethod) = 4
b_reals(::OneMMethod) = 2
a_reals(::FMAddSubMethod) = 2
b_reals(::FMAddSubMethod) = 2

# Accumulator planes held live: planar keeps separate re/im planes.
accumulator_planes(::RealMethod) = 1
accumulator_planes(::PlanarMethod) = 2
accumulator_planes(::OneMMethod) = 1
accumulator_planes(::FMAddSubMethod) = 1

complex_method(::Any) = RealMethod()

realtype(k::DescriptorKernel) = realtype(k.descriptor)
packed_a_per_k(k::DescriptorKernel) = packed_a_per_k(k.descriptor)
packed_b_per_k(k::DescriptorKernel) = packed_b_per_k(k.descriptor)
a_format(k::DescriptorKernel) = a_format(k.descriptor)
b_format(k::DescriptorKernel) = b_format(k.descriptor)
mr(k::DescriptorKernel) = mr(k.descriptor)
nr(k::DescriptorKernel) = nr(k.descriptor)
scalartype(k::DescriptorKernel) = scalartype(k.descriptor)
packed_a_offset(k::DescriptorKernel, i::Int, p::Int) = packed_a_offset(k.descriptor, i, p)
packed_b_offset(k::DescriptorKernel, j::Int, p::Int) = packed_b_offset(k.descriptor, j, p)
packed_a_length(k::DescriptorKernel, kc::Int) = packed_a_length(k.descriptor, kc)
packed_b_length(k::DescriptorKernel, kc::Int) = packed_b_length(k.descriptor, kc)
# Only these resolve for a complex descriptor; the single-plane offsets above
# are a MethodError there, by design.
@inline packed_a_plane_offset(k::DescriptorKernel, plane::Int, i::Int, p::Int) =
    packed_a_plane_offset(k.descriptor, plane, i, p)
@inline packed_b_plane_offset(k::DescriptorKernel, plane::Int, j::Int, p::Int) =
    packed_b_plane_offset(k.descriptor, plane, j, p)

# GUARDRAIL: every forwarded argument needs its OWN bound type parameter; an
# unbound one makes the call dynamically dispatched and allocating on every
# pack. `V` is unconstrained so a `PackedPanel` forwards too.
pack_a!(
    packed::V, source::QSTile, kernel::K, transform::F
) where {V, MR, NR, T, K <: DescriptorKernel{MR, NR, T}, F} =
    pack_a!(packed, source, kernel.descriptor, transform)
pack_b!(
    packed::V, source::QSTile, kernel::K, transform::F
) where {V, MR, NR, T, K <: DescriptorKernel{MR, NR, T}, F} =
    pack_b!(packed, source, kernel.descriptor, transform)
@inline unsafe_pack_a!(
    packed::V, source::QSTile, kernel::K, transform::F
) where {V, MR, NR, T, K <: DescriptorKernel{MR, NR, T}, F} =
    unsafe_pack_a!(packed, source, kernel.descriptor, transform)
@inline unsafe_pack_b!(
    packed::V, source::QSTile, kernel::K, transform::F
) where {V, MR, NR, T, K <: DescriptorKernel{MR, NR, T}, F} =
    unsafe_pack_b!(packed, source, kernel.descriptor, transform)

# Constructor check: `rows` reals per sliver must split into whole `W`-vectors.
# `even`: the lane-pair kernels keep one element's re/im in adjacent lanes, and
# an odd `W` can divide `2MR` while splitting an element across two vectors --
# a wrong answer, not an error, hence rejected here.
@inline function _check_vector_shape(name, rows::Int, W, even::Bool = false)
    W isa Int && W > 0 ||
        throw(ArgumentError("$name requires an Int vector width W > 0, got W = $W"))
    even && isodd(W) &&
        throw(ArgumentError("$name requires an even vector width W, got W = $W"))
    mod(rows, W) == 0 || throw(
        ArgumentError("$name requires $rows reals per sliver to be a multiple of W = $W")
    )
    return nothing
end

# Generator-time check of an accumulator tuple against the kernel's shape.
function _check_acc(f::Symbol, R, T, NA::Int, want::Int)
    R === real(T) || throw(ArgumentError("$f: accumulator lane type $R is not real($T)"))
    NA == want || throw(ArgumentError("$f: accumulator length $NA, expected $want"))
    return nothing
end

# Default `SIMD.Vec` lane count: one 256-bit register.
_default_lanewidth(::Type{Float64}) = 4
_default_lanewidth(::Type{Float32}) = 8

# `beta == 0` writes zeros without reading `C`; `beta == 1` is a no-op.
function scale_tile!(destination::QSTile, beta::T) where {T}
    m = nrows(destination)
    n = ncols(destination)
    (m == 0 || n == 0) && return destination
    if isone(beta)
        return destination
    elseif iszero(beta)
        @inbounds for j in 0:(n - 1), i in 0:(m - 1)
            tile_store!(destination, i, j, zero(T))
        end
    else
        @inbounds for j in 0:(n - 1), i in 0:(m - 1)
            tile_store!(destination, i, j, tile_load(destination, i, j) * beta)
        end
    end
    return destination
end

# `C = alpha*r + beta*C` at one element. Ternaries, so `beta == 0` never reads C.
@inline _axpby_tile!(dest, i::Int, j::Int, alpha, r, beta) = tile_store!(
    dest, i, j,
    iszero(beta) ? alpha * r :
        isone(beta) ? muladd(alpha, r, tile_load(dest, i, j)) :
        muladd(alpha, r, beta * tile_load(dest, i, j))
)

@inline _axpby_at!(storage, idx::Int, alpha, r, beta) = @inbounds storage[idx] =
    iszero(beta) ? alpha * r :
    isone(beta) ? muladd(alpha, r, storage[idx]) :
    muladd(alpha, r, beta * storage[idx])

# The `(m, n)` to store over, or `(0, 0)` when already done (empty destination,
# or `alpha == 0` handled by `scale_tile!`).
@inline function _store_prologue!(destination::QSTile, alpha, beta)
    m = nrows(destination)
    n = ncols(destination)
    if m == 0 || n == 0
        return (0, 0)
    elseif iszero(alpha)
        scale_tile!(destination, beta)
        return (0, 0)
    end
    return (m, n)
end

# GUARDRAIL: every throw reachable from the per-tile prologue sits behind a
# `@noinline` helper. An inline interpolated message pulls `print_to_string`, a
# GC frame and ~1 KB of stack into the hot tile function.
@noinline _throw_packed_short(which::Symbol, got::Int, need::Int, kc::Int) = throw(
    DimensionMismatch(
        "execute_tile!: packed_$which has length $got, " *
            "need at least packed_$(which)_length(kernel, kc=$kc) = $need"
    )
)
@noinline _throw_tile_extent(which::Symbol, got::Int, limit::Int) = throw(
    ArgumentError(
        "destination valid $which extent $got exceeds $(which === :row ? "mr" : "nr")(kernel) = $limit"
    )
)
@noinline _throw_negative_kc(where::Symbol, kc::Int) =
    throw(ArgumentError("$where requires kc >= 0, got kc = $kc"))

# `execute_tile!`'s validation, in order. Returns `(run, alphaT, betaT)`;
# `run == false` means the call is finished. `Val(false)` drops the storage
# bounds check at compile time (`unsafe_execute_tile!` only).
# GUARDRAIL: `@inline` and one bound type parameter per argument (hot path).
@inline function _execute_tile_prologue!(
        kernel::K, destination::QSTile, packed_a::PA, packed_b::PB,
        kc::Int, alpha, beta, ::Val{BOUNDS}
    ) where {MR, NR, T, K <: DescriptorKernel{MR, NR, T}, PA, PB, BOUNDS}
    m = nrows(destination)
    n = ncols(destination)
    m <= MR || _throw_tile_extent(:row, m, MR)
    n <= NR || _throw_tile_extent(:column, n, NR)
    kc >= 0 || _throw_negative_kc(:execute_tile!, kc)

    alphaT = convert(T, alpha)
    betaT = convert(T, beta)

    (m == 0 || n == 0) && return (false, alphaT, betaT)

    BOUNDS && checked_tile_storage_bounds(destination)

    if kc == 0 || iszero(alphaT)
        scale_tile!(destination, betaT)
        return (false, alphaT, betaT)
    end

    need_a = packed_a_length(kernel, kc)
    length(packed_a) >= need_a || _throw_packed_short(:a, length(packed_a), need_a, kc)
    need_b = packed_b_length(kernel, kc)
    length(packed_b) >= need_b || _throw_packed_short(:b, length(packed_b), need_b, kc)

    return (true, alphaT, betaT)
end

# One checked K panel: `zero_accumulator`, `accumulate`, `store_tile!`.
function execute_tile!(
        kernel::K, destination::QSTile, packed_a::PA, packed_b::PB,
        kc::Int, alpha, beta
    ) where {MR, NR, T, K <: DescriptorKernel{MR, NR, T}, PA, PB}
    run, alphaT, betaT = _execute_tile_prologue!(
        kernel, destination, packed_a, packed_b, kc, alpha, beta, Val(true)
    )
    run || return destination
    acc = accumulate(kernel, zero_accumulator(kernel), packed_a, packed_b, kc)
    return store_tile!(destination, acc, alphaT, betaT, kernel)
end

# `execute_tile!` without `checked_tile_storage_bounds(destination)`: an
# out-of-range destination is a silent out-of-bounds WRITE. The callers
# (`_execute_nest!`, `_execute_half_packed_nest!`) check the union of a whole
# macro block's tiles once with `checked_span_bounds`, which is equivalent
# because the check only compares range extremes.
@inline function unsafe_execute_tile!(
        kernel::K, destination::QSTile, packed_a::PA, packed_b::PB,
        kc::Int, alpha, beta
    ) where {K, PA, PB}
    run, alphaT, betaT = _execute_tile_prologue!(
        kernel, destination, packed_a, packed_b, kc, alpha, beta, Val(false)
    )
    run || return destination
    acc = accumulate(kernel, zero_accumulator(kernel), packed_a, packed_b, kc)
    return store_tile!(destination, acc, alphaT, betaT, kernel)
end
