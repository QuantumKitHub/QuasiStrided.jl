# What every microkernel shares: the `DescriptorKernel` supertype and its
# forwarding, the complex-method traits, constructor checks, the axpby element
# helpers, the validation prologues of `execute_tile!`/`store_tile!`, and the
# one generic `execute_tile!`. Each kernel file implements only
# `zero_accumulator`, `accumulate` and `store_tile!`.

# Shared supertype for kernels wrapping a KernelDescriptor as `.descriptor`;
# lets mr/nr/scalartype/packed_*/pack_a!/pack_b! forward once for all of them.
abstract type DescriptorKernel{MR, NR, T} end

# ----------------------------------------------------------------------------
# Complex methods
# ----------------------------------------------------------------------------

"""
    ComplexMethod

Which complex-arithmetic method a kernel implements. Singleton types, so
`default_blocking` and the shape menus dispatch on them without a runtime
branch.

**No auto-dispatch rule is derived from any measurement**: the ranking of
methods differs from machine to machine.
[`PlanarMethod`](@ref) is the default; [`OneMMethod`](@ref) is
selected only by naming the kernel. One narrow exception: on `:avx512`, when
M is too small to fill the planar override's register tile, the extent
demotion picks an [`FMAddSubMethod`](@ref) shape (`_small_m_shape` in
src/planning/kernel_selection.jl, measured 2x there).
"""
abstract type ComplexMethod end

"""
    RealMethod()

Not a complex method: what `complex_method` reports for a real kernel, so that
`default_blocking` and the shape menus are total without a `T <: Complex` guard
at every call site.
"""
struct RealMethod <: ComplexMethod end

"""
    PlanarMethod()

Split-complex: both operands [`PlanarFormat`](@ref), driven by a genuinely
complex microkernel that issues four real FMAs per (A-vector, B-scalar) pair on
data already in the right lanes -- no shuffles, no `fmaddsub`. The default.
"""
struct PlanarMethod <: ComplexMethod end

"""
    OneMMethod()

Van Zee's induced 1m: one *real* microkernel of shape `2mr x nr` run over
`2*kc` real steps, fed by [`OneEFormat`](@ref) A and [`PlanarFormat`](@ref) B.
The kernel body is the real `SIMDKernel`'s, reused verbatim -- so a
planar-vs-1m measurement compares two methods, not two hand-written kernels.
"""
struct OneMMethod <: ComplexMethod end

"""
    FMAddSubMethod()

Interleaved complex accumulation with x86 `vfmaddsub`: A packed in
[`InterleavedFormat`](@ref) (`Complex{T}`'s native `[re, im, ...]` order),
B in [`PlanarFormat`](@ref) (broadcast `re`/`im` scalars), and ONE
interleaved accumulator plane, updated per (A-vector, B-column) pair by two
chained fmaddsub ops on `a` and its in-register pair-swap. Same FMA count as
planar and 1m; see src/microkernels/fmaddsub.jl. Selected by naming
[`FMAddSubKernel`](@ref), and automatically only by the AVX-512 small-M
demotion (`_small_m_shape`, src/planning/kernel_selection.jl).
"""
struct FMAddSubMethod <: ComplexMethod end

"""
    a_reals(::ComplexMethod) -> Int
    b_reals(::ComplexMethod) -> Int

Reals per element in the packed A / B panel under this method. These are what
`default_blocking` divides the measured real `mc`/`nc` by, so every method gets
the *same packed byte budget* rather than the same element count -- 1m's `mc`
halving is derived from this, never tabulated.
"""
a_reals(::RealMethod) = 1
b_reals(::RealMethod) = 1
a_reals(::PlanarMethod) = 2
b_reals(::PlanarMethod) = 2
a_reals(::OneMMethod) = 4
b_reals(::OneMMethod) = 2
a_reals(::FMAddSubMethod) = 2
b_reals(::FMAddSubMethod) = 2

"""
    accumulator_planes(::ComplexMethod) -> Int

Accumulator planes the microkernel holds live. Planar keeps separate real and
imaginary planes (so its register budget is twice a real kernel's at the same
`(mr, nr)`); 1m keeps one plane over a doubled real row count. Used by the
register-budget assertion, which must not assume the real kernel's shape.
"""
accumulator_planes(::RealMethod) = 1
accumulator_planes(::PlanarMethod) = 2
accumulator_planes(::OneMMethod) = 1
accumulator_planes(::FMAddSubMethod) = 1

"""
    complex_method(kernel) -> ComplexMethod

Which complex method a kernel implements; [`RealMethod`](@ref) for every kernel
that has not said otherwise, which keeps `default_blocking` and the shape menus
total.
"""
complex_method(::Any) = RealMethod()

# ----------------------------------------------------------------------------
# DescriptorKernel forwarding to the wrapped `Descriptor` (src/packing/format.jl)
# ----------------------------------------------------------------------------

realtype(k::DescriptorKernel) = realtype(k.descriptor)
packed_a_per_k(k::DescriptorKernel) = packed_a_per_k(k.descriptor)
packed_b_per_k(k::DescriptorKernel) = packed_b_per_k(k.descriptor)
a_format(k::DescriptorKernel) = a_format(k.descriptor)
b_format(k::DescriptorKernel) = b_format(k.descriptor)

# Shared forwarding for any DescriptorKernel (ScalarKernel, SIMDKernel, ...).
mr(k::DescriptorKernel) = mr(k.descriptor)
nr(k::DescriptorKernel) = nr(k.descriptor)
scalartype(k::DescriptorKernel) = scalartype(k.descriptor)
packed_a_offset(k::DescriptorKernel, i::Int, p::Int) = packed_a_offset(k.descriptor, i, p)
packed_b_offset(k::DescriptorKernel, j::Int, p::Int) = packed_b_offset(k.descriptor, j, p)
packed_a_length(k::DescriptorKernel, kc::Int) = packed_a_length(k.descriptor, kc)
packed_b_length(k::DescriptorKernel, kc::Int) = packed_b_length(k.descriptor, kc)

# Plane-offset forwarding. The single-plane `packed_a_offset`/`packed_b_offset`
# above only resolve for a real descriptor: a complex kernel asked for one gets
# the intended MethodError.
@inline packed_a_plane_offset(k::DescriptorKernel, plane::Int, i::Int, p::Int) =
    packed_a_plane_offset(k.descriptor, plane, i, p)
@inline packed_b_plane_offset(k::DescriptorKernel, plane::Int, j::Int, p::Int) =
    packed_b_plane_offset(k.descriptor, plane, j, p)

# pack_a!/pack_b! dispatch on a bare `Descriptor`; forward any wrapper.
#
# GUARDRAIL: every forwarded argument needs its OWN bound type parameter (`V`,
# `K`, `F`). Leaving one unbound makes the call dynamically dispatched and
# allocating on every pack. `V` is unconstrained rather than
# `<: AbstractVector{T}` so that a `PackedPanel` (src/packing/panel.jl) forwards too;
# `T` comes from `K` instead.
pack_a!(
    packed::V, source::QSTile, kernel::K, transform::F
) where {V, MR, NR, T, K <: DescriptorKernel{MR, NR, T}, F} =
    pack_a!(packed, source, kernel.descriptor, transform)
pack_b!(
    packed::V, source::QSTile, kernel::K, transform::F
) where {V, MR, NR, T, K <: DescriptorKernel{MR, NR, T}, F} =
    pack_b!(packed, source, kernel.descriptor, transform)

# Same forwarding for the bounds-check-skipping siblings (src/packing/pack.jl), with
# the same per-argument type parameters for the same reason. `@inline`, like
# the functions they forward to: these are the driver's per-sliver entry points.
@inline unsafe_pack_a!(
    packed::V, source::QSTile, kernel::K, transform::F
) where {V, MR, NR, T, K <: DescriptorKernel{MR, NR, T}, F} =
    unsafe_pack_a!(packed, source, kernel.descriptor, transform)
@inline unsafe_pack_b!(
    packed::V, source::QSTile, kernel::K, transform::F
) where {V, MR, NR, T, K <: DescriptorKernel{MR, NR, T}, F} =
    unsafe_pack_b!(packed, source, kernel.descriptor, transform)

# Lane-width checks shared by SIMDKernel/PlanarKernel/OneMKernel. `name` only
# names the type in the message; construction-time only, never hot.
@inline function _check_lanewidth(name, W)
    W isa Int && W > 0 ||
        throw(ArgumentError("$name requires an Int vector width W > 0, got W = $W"))
    return nothing
end

@inline function _check_mr_multiple(name, MR::Int, W::Int)
    mod(MR, W) == 0 || throw(
        ArgumentError(
            "$name requires mr(kernel) = $MR to be a multiple of " *
                "the vector width W = $W"
        )
    )
    return nothing
end

"""
    scale_tile!(destination::QSTile{T}, beta::T) -> destination

Apply `C[i,j] = beta * C[i,j]` over the destination's valid rectangle.
`beta == 0` writes `zero(T)` without reading old `C`; `beta == 1` is a
no-op; an empty destination is a no-op in every branch.
"""
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

# `C = alpha*r + beta*C` at one element, in the two forms the kernels need.
# Ternaries, so `beta == 0` never reads the old value (contract pinned by the
# nonfinite-poisoning tests).
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

# ----------------------------------------------------------------------------
# Validation prologues of store_tile! and execute_tile!
# ----------------------------------------------------------------------------

# `store_tile!`'s alpha/beta preamble: an empty destination is a no-op, and
# `alpha == 0` degenerates to `scale_tile!`. Returns the `(m, n)` extent to
# store over, or `(0, 0)` when the store is already finished.
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

@noinline _throw_packed_short(which::Symbol, got::Int, need::Int, kc::Int) = throw(
    DimensionMismatch(
        "execute_tile!: packed_$which has length $got, " *
            "need at least packed_$(which)_length(kernel, kc=$kc) = $need"
    )
)

# GUARDRAIL: every throw reachable from the per-tile prologue lives behind
# one of these `@noinline` helpers, never as an inline
# `throw(ArgumentError("...$m..."))`. An inline interpolated message pulls
# `print_to_string` -- and with it a GC frame and ~1 KB of stack -- into the
# hot tile function, which is entered once per micro-tile (measured 2026-09-26,
# ccqlin038: the K=1 outer-product suite cases spend ~37 ns per 16x6 Float64
# tile of pure fixed cost, of which this frame setup and the out-of-line store
# call below are the two avoidable parts).
@noinline _throw_tile_extent(which::Symbol, got::Int, limit::Int) = throw(
    ArgumentError(
        "destination valid $which extent $got exceeds $(which === :row ? "mr" : "nr")(kernel) = $limit"
    )
)
@noinline _throw_negative_kc(where::Symbol, kc::Int) =
    throw(ArgumentError("$where requires kc >= 0, got kc = $kc"))

# `execute_tile!`'s validation sequence, in this order:
# destination extent vs. kernel shape, `kc >= 0`, the alpha/beta converts, the
# empty short-circuit, storage bounds BEFORE any `@inbounds` path,
# the `kc == 0 || alpha == 0` beta-only branch, then both packed capacities.
# Returns `(run, alphaT, betaT)`; `run == false` means the call is finished and
# the caller returns `destination` untouched.
#
# GUARDRAIL: `@inline`, and one bound type parameter per argument, for the
# reason spelled out at `pack_a!` above. This is on the hot path.
@inline _execute_tile_prologue!(
    kernel::K, destination::QSTile, packed_a::PA, packed_b::PB,
    kc::Int, alpha, beta
) where {MR, NR, T, K <: DescriptorKernel{MR, NR, T}, PA, PB} =
    _execute_tile_prologue!(
    kernel, destination, packed_a, packed_b, kc, alpha, beta, Val(true)
)

# `BOUNDS` is a compile-time flag: at `Val(false)` the
# `checked_tile_storage_bounds` call folds away. Only `unsafe_execute_tile!`
# passes `Val(false)`.
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

"""
    _default_lanewidth(::Type{T}) -> Int

Default `SIMD.Vec` lane count for `T` (one 256-bit register's worth): 4 for
`Float64`, 8 for `Float32`. Not hardware-detected or tuned.
"""
_default_lanewidth(::Type{Float64}) = 4
_default_lanewidth(::Type{Float32}) = 8

"""
    execute_tile!(kernel, destination::QSTile, packed_a, packed_b, kc::Int, alpha, beta) -> destination

Checked composition of `zero_accumulator`, `accumulate` and `store_tile!` for
a single K-panel call (multi-panel accumulation is the driver's job), shared
by every `DescriptorKernel`. Validation order and short-circuits are
`_execute_tile_prologue!`'s: `kc == 0` or `alpha == 0` scales by `beta` only,
without reading `packed_a`/`packed_b`. The vector kernels match
`ScalarKernel` only to within a tolerance (FMA grouping differs), never
bitwise.
"""
function execute_tile!(
        kernel::K, destination::QSTile, packed_a::PA, packed_b::PB,
        kc::Int, alpha, beta
    ) where {MR, NR, T, K <: DescriptorKernel{MR, NR, T}, PA, PB}
    run, alphaT, betaT =
        _execute_tile_prologue!(kernel, destination, packed_a, packed_b, kc, alpha, beta)
    run || return destination
    acc = accumulate(kernel, zero_accumulator(kernel), packed_a, packed_b, kc)
    return store_tile!(destination, acc, alphaT, betaT, kernel)
end

"""
    unsafe_execute_tile!(kernel, destination::QSTile, packed_a, packed_b, kc::Int, alpha, beta) -> destination

[`execute_tile!`](@ref) **without** the
`checked_tile_storage_bounds(destination)` call.

PRECONDITION, which the caller must have established: every address
`destination` can write -- `destination.base + row_offset(i) + col_offset(j)`
for `0 <= i < nrows(destination)`, `0 <= j < ncols(destination)` -- lies in
`0:length(destination.storage)-1`. Violating it is an out-of-bounds WRITE
through an `@inbounds`/pointer path, not an exception; this is the sharpest
edge in the package, because the checked path is what stands between a bad
`AxisGroup` and silent memory corruption.

Everything else `_execute_tile_prologue!` validates is still validated: the
destination extent against the kernel's `(MR, NR)`, `kc >= 0`, both packed
capacities, and the empty / `kc == 0` / `alpha == 0` short-circuits.

The callers in this package are `_execute_nest!` (src/execution/execute.jl), which
validates the union of an entire (ic, jc) macro block's micro-tiles in one
[`checked_span_bounds`](@ref) call before running any of them, and
`_execute_half_packed_nest!` (src/execution/halfpack.jl), which does the same
per jc block for its single M sliver. That is an
exactly equivalent test: the block's micro-tiles are the full cross product of
its M-sliver row sets and N-sliver column sets, those sets partition the
block's two offset buffers, and the check only compares range extremes -- so
the block check passes iff every per-tile check would have.
"""
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
