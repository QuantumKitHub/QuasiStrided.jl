# The microkernel contract, the complex-arithmetic methods, and the validation
# and store helpers every kernel shares.

"""
    Microkernel{MR, NR, T}

A register-tile microkernel: computes an `MR x NR` tile of `C`, element type
`T`, from one packed A sliver and one packed B sliver of a K block. A kernel
wraps a `Descriptor` (tile size, element type, packing formats) as
`.descriptor` and implements [`zero_accumulator`](@ref), [`add_tile`](@ref) and
[`store_tile!`](@ref); [`execute_tile!`](@ref) composes them. The vector
kernels match `ScalarKernel` only to within rounding (FMA grouping differs),
never bitwise.
"""
abstract type Microkernel{MR, NR, T} end

"""
    KernelMethod(kernel)

How a kernel maps its multiply-adds onto real arithmetic, as singletons so
blocking and the shape menus dispatch on it:

  - `RealMethod`: a real kernel.
  - `PlanarMethod`: split re/im planes, 4 real FMAs per complex MAC.
  - `OneMMethod`: Van Zee's 1m, a real `2MR x NR` kernel.
  - `FMAddSubMethod`: interleaved A, x86 `vfmaddsub`.
  - `ComplexRealMethod`: complex A, real B; the real kernel on `2MR` rows.
  - `RealComplexMethod`: real A, complex B; the real kernel on `2NR` columns.

No method ranking is hardcoded: planar is the default, 1m and fmaddsub are used
only when named (plus fmaddsub for the AVX-512 small-M demotion,
`small_m_shape` in src/planning/kernel_selection.jl).
"""
abstract type KernelMethod end
struct RealMethod <: KernelMethod end
struct PlanarMethod <: KernelMethod end
struct OneMMethod <: KernelMethod end
struct FMAddSubMethod <: KernelMethod end
struct ComplexRealMethod <: KernelMethod end
struct RealComplexMethod <: KernelMethod end

# The (A, B) packing formats of every kernel of a method.
pack_formats(::RealMethod) = (RealFormat(), RealFormat())
pack_formats(::PlanarMethod) = (PlanarFormat(), PlanarFormat())
pack_formats(::OneMMethod) = (OneEFormat(), PlanarFormat())
pack_formats(::FMAddSubMethod) = (InterleavedFormat(), PlanarFormat())
pack_formats(::ComplexRealMethod) = (InterleavedFormat(), RealFormat())
pack_formats(::RealComplexMethod) = (RealFormat(), InterleavedFormat())

# Accumulator planes held live: planar keeps separate re/im planes.
accumulator_planes(::KernelMethod) = 1
accumulator_planes(::PlanarMethod) = 2

realtype(k::Microkernel) = realtype(k.descriptor)
sliver_width(k::Microkernel) = sliver_width(k.descriptor)
a_format(k::Microkernel) = a_format(k.descriptor)
b_format(k::Microkernel) = b_format(k.descriptor)
tile_size(k::Microkernel) = tile_size(k.descriptor)
scalartype(k::Microkernel) = scalartype(k.descriptor)
@inline sliver_spec(k::Microkernel, i::Int) = sliver_spec(k.descriptor, i)
@inline packed_a_offset(k::Microkernel, i::Int, p::Int, plane::Int = 0) =
    packed_a_offset(k.descriptor, i, p, plane)
@inline packed_b_offset(k::Microkernel, j::Int, p::Int, plane::Int = 0) =
    packed_b_offset(k.descriptor, j, p, plane)
packed_a_length(k::Microkernel, k_block_length::Int) = packed_a_length(k.descriptor, k_block_length)
packed_b_length(k::Microkernel, k_block_length::Int) = packed_b_length(k.descriptor, k_block_length)

"""
    zero_accumulator(kernel::Microkernel) -> acc

The kernel's zero register tile, an immutable value.
"""
function zero_accumulator end

"""
    add_tile(kernel::Microkernel, acc, packed_a, packed_b, k_block_length) -> acc

Add the products of `k_block_length` K steps of the packed A and B slivers to
`acc` and return the result. `k_block_length == 0` returns `acc` without
reading the panels; a negative one throws an `ArgumentError`.
"""
function add_tile end

"""
    store_tile!(destination::Tile, acc, alpha, beta, kernel::Microkernel) -> destination

Write `alpha * acc + beta * C` over the valid rectangle `size(destination)`
only: `alpha == 0` never reads `acc`, `beta == 0` never reads old `C`, and
padding lanes of `acc` are never read. `alpha` and `beta` are of the kernel's
element type.
"""
function store_tile! end

# Constructor check: `rows` reals per sliver must split into whole `W`-vectors.
# `even`: the lane-pair kernels keep one element's re/im in adjacent lanes, and
# an odd `W` can divide `2MR` while splitting an element across two vectors --
# a wrong answer, not an error, hence rejected here.
@inline function check_vector_shape(name, rows::Int, W, even::Bool = false)
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
function check_acc(f::Symbol, R, T, NA::Int, want::Int)
    R === real(T) || throw(ArgumentError("$f: accumulator lane type $R is not real($T)"))
    NA == want || throw(ArgumentError("$f: accumulator length $NA, expected $want"))
    return nothing
end

# Default `SIMD.Vec` lane count: one 256-bit register.
default_lanewidth(::Type{Float64}) = 4
default_lanewidth(::Type{Float32}) = 8

# `beta == 0` writes zeros without reading `C`; `beta == 1` is a no-op.
function scale_tile!(destination::Tile, beta::T) where {T}
    m, n = size(destination)
    (m == 0 || n == 0) && return destination
    if isone(beta)
        return destination
    elseif iszero(beta)
        @inbounds for j in 1:n, i in 1:m
            destination[i, j] = zero(T)
        end
    else
        @inbounds for j in 1:n, i in 1:m
            destination[i, j] = convert(T, destination[i, j]) * beta
        end
    end
    return destination
end

# `C = alpha*r + beta*C` at one element. Ternaries, so `beta == 0` never reads C.
@inline axpby_tile!(dest, i::Int, j::Int, alpha, r, beta::T) where {T} = @inbounds dest[i, j] =
    iszero(beta) ? alpha * r :
    isone(beta) ? muladd(alpha, r, convert(T, dest[i, j])) :
    muladd(alpha, r, beta * convert(T, dest[i, j]))

@inline axpby_at!(storage, idx::Int, alpha, r, beta::T) where {T} = @inbounds storage[idx] =
    iszero(beta) ? alpha * r :
    isone(beta) ? muladd(alpha, r, convert(T, storage[idx])) :
    muladd(alpha, r, beta * convert(T, storage[idx]))

# The `(m, n)` to store over, or `(0, 0)` when already done (empty destination,
# or `alpha == 0` handled by `scale_tile!`).
@inline function store_prologue!(destination::Tile, alpha, beta)
    m, n = size(destination)
    if m == 0 || n == 0
        return (0, 0)
    elseif iszero(alpha)
        scale_tile!(destination, beta)
        return (0, 0)
    end
    return (m, n)
end

# GUARDRAIL: every throw reachable from `execute_tile!` sits behind a
# `@noinline` helper. An inline interpolated message pulls `print_to_string`, a
# GC frame and ~1 KB of stack into the hot tile function.
@noinline throw_packed_short(which::Symbol, got::Int, need::Int, k_block_length::Int) = throw(
    DimensionMismatch(
        "execute_tile!: packed_$which has length $got, " *
            "need at least packed_$(which)_length(kernel, k_block_length=$k_block_length) = $need"
    )
)
@noinline throw_tile_extent(which::Symbol, got::Int, limit::Int) = throw(
    ArgumentError(
        "destination valid $which extent $got exceeds tile_size(kernel, $(which === :row ? 1 : 2)) = $limit"
    )
)
@noinline throw_negative_k_block_length(where::Symbol, k_block_length::Int) =
    throw(ArgumentError("$where requires k_block_length >= 0, got k_block_length = $k_block_length"))

# The kernels read only `PackedPanel`s; a `DenseVector` is borrowed as one.
function add_tile(
        kernel::K, acc::A, packed_a::PA, packed_b::PB, k_block_length::Int
    ) where {K <: Microkernel, A, PA <: DenseVector, PB <: DenseVector}
    return GC.@preserve packed_a packed_b add_tile(
        kernel, acc, packed_panel(packed_a, 1, length(packed_a)),
        packed_panel(packed_b, 1, length(packed_b)), k_block_length
    )
end

"""
    execute_tile!(kernel::Microkernel, destination::Tile, packed_a, packed_b,
                  k_block_length, alpha, beta) -> destination

One K block of one register tile: `zero_accumulator`, `add_tile`,
`store_tile!`, i.e. `C = alpha * A * B + beta * C` over `destination`'s valid
rectangle. Validates, in order: the destination extents against
`tile_size(kernel)` (`ArgumentError`), `k_block_length >= 0`
(`ArgumentError`), `alpha` and `beta` converting to the kernel's element type;
then returns at once for an empty destination; checks the
destination's storage bounds (`BoundsError`); applies only `beta` when
`k_block_length == 0` or `alpha == 0`, without reading the panels; and checks
the panel lengths (`DimensionMismatch`).

Under `@inbounds` (the call inlined) the storage bounds check is skipped, so an
out-of-range destination is a silent out-of-bounds write; every other check
stays.
"""
@inline function execute_tile!(
        kernel::K, destination::Tile, packed_a::PA, packed_b::PB,
        k_block_length::Int, alpha, beta
    ) where {MR, NR, T, K <: Microkernel{MR, NR, T}, PA, PB}
    m, n = size(destination)
    m <= MR || throw_tile_extent(:row, m, MR)
    n <= NR || throw_tile_extent(:column, n, NR)
    k_block_length >= 0 || throw_negative_k_block_length(:execute_tile!, k_block_length)
    alphaT = convert(T, alpha)
    betaT = convert(T, beta)
    (m == 0 || n == 0) && return destination
    @boundscheck checked_tile_storage_bounds(destination)
    if k_block_length == 0 || iszero(alphaT)
        scale_tile!(destination, betaT)
        return destination
    end

    need_a = packed_a_length(kernel, k_block_length)
    length(packed_a) >= need_a || throw_packed_short(:a, length(packed_a), need_a, k_block_length)
    need_b = packed_b_length(kernel, k_block_length)
    length(packed_b) >= need_b || throw_packed_short(:b, length(packed_b), need_b, k_block_length)

    acc = add_tile(kernel, zero_accumulator(kernel), packed_a, packed_b, k_block_length)
    return store_tile!(destination, acc, alphaT, betaT, kernel)
end
