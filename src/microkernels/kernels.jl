# The kernel types and their type-level facts, ahead of every implementation: a
# `@generated` function's generator only sees methods defined before it.

"""
    ScalarKernel(::Val{MR}, ::Val{NR}, ::Type{T})

Scalar reference microkernel for register tile `(MR, NR)` and element type `T`.
Its accumulator is an `NTuple{MR * NR, T}` holding the tile column-major.
"""
struct ScalarKernel{MR, NR, T} <: Microkernel{MR, NR, T}
    descriptor::RealDescriptor{MR, NR, T}
end

ScalarKernel(::Val{MR}, ::Val{NR}, ::Type{T}) where {MR, NR, T} =
    ScalarKernel(Descriptor(Val(MR), Val(NR), T))

"""
    VectorKernel{MR, NR, T, W}

An explicit-SIMD microkernel over `SIMD.Vec{W, real(T)}` lanes. Its
accumulator is an immutable tuple of vectors in one of three layouts
([`AccumulatorLayout`](@ref)), which fixes the stores; each kernel supplies
its K step, `accumulate_step`, and the generic `add_tile` loops it
`k_steps(kernel, k_block_length)` times.
"""
abstract type VectorKernel{MR, NR, T, W} <: Microkernel{MR, NR, T} end

"""
    AccumulatorLayout(kernel::VectorKernel)

How the accumulator holds the tile, as `(MR ÷ rows) * NR` vectors of `rows`
consecutive tile rows each, column-major (`rows_per_vector`):

  - `RealLayout()`: one real per lane (`SIMDKernel`).
  - `SplitLayout()`: separate re and im vectors (`PlanarKernel`: a real and an
    imaginary plane in one flat tuple; `RealComplexKernel`: the real kernel's
    columns `2j-1` and `2j`).
  - `LanePairLayout()`: lanes `2u-1` and `2u` are re and im of one row
    (1m, fmaddsub, `ComplexRealKernel`).
"""
abstract type AccumulatorLayout end
struct RealLayout <: AccumulatorLayout end
struct SplitLayout <: AccumulatorLayout end
struct LanePairLayout <: AccumulatorLayout end

# The checks of every vector kernel's constructor; the descriptor has checked
# the tile size and the element type.
function check_vector_kernel(::Type{K}, MR, W) where {K <: VectorKernel}
    pairs = AccumulatorLayout(K) isa LanePairLayout
    return check_vector_shape(nameof(K), pairs ? 2 * MR : MR, W, pairs)
end

"""
    SIMDKernel(::Val{MR}, ::Val{NR}, ::Type{T}[, ::Val{W}])

Explicit-SIMD real microkernel over `SIMD.Vec{W,T}` lanes. `MR` must be a
multiple of `W` (default: one 256-bit register, 4 for `Float64`, 8 for `Float32`).
"""
struct SIMDKernel{MR, NR, T, W} <: VectorKernel{MR, NR, T, W}
    descriptor::RealDescriptor{MR, NR, T}
    function SIMDKernel{MR, NR, T, W}(descriptor::RealDescriptor{MR, NR, T}) where {MR, NR, T, W}
        check_vector_kernel(SIMDKernel, MR, W)
        return new{MR, NR, T, W}(descriptor)
    end
end

"""
    PlanarKernel(::Val{MR}, ::Val{NR}, ::Type{T}[, ::Val{W}])

Split-complex microkernel for `T = ComplexF32/ComplexF64` over
`SIMD.Vec{W,real(T)}` lanes. `MR`/`NR` are complex extents; `MR` must be a
multiple of `W`, which counts reals.
"""
struct PlanarKernel{MR, NR, T, W} <: VectorKernel{MR, NR, T, W}
    descriptor::Descriptor{MR, NR, T, PlanarFormat, PlanarFormat}
    function PlanarKernel{MR, NR, T, W}(
            descriptor::Descriptor{MR, NR, T, PlanarFormat, PlanarFormat}
        ) where {MR, NR, T, W}
        check_vector_kernel(PlanarKernel, MR, W)
        return new{MR, NR, T, W}(descriptor)
    end
end

"""
    OneMKernel(::Val{MR}, ::Val{NR}, ::Type{T}[, ::Val{W}])

Van Zee's induced 1m complex microkernel: the real `SIMDKernel` of `2MR x NR`
over `SIMD.Vec{W,real(T)}` lanes, run verbatim over `2 * k_block_length` real
K steps against 1e-packed A and planar B. At logical K step `p`, `OneEFormat`
A holds `(re_0, im_0, re_1, im_1, ...)` then `(-im_0, re_0, -im_1, re_1, ...)`,
and planar B `re_0..` then `im_0..`, so accumulator real row `2i-1` is the
real and `2i` the imaginary part of complex row `i`. `2MR` must be a multiple
of `W`, and `W` must be even. Used only when named in
`plan_contract(...; kernel = ...)`.
"""
struct OneMKernel{MR, NR, T, W} <: VectorKernel{MR, NR, T, W}
    descriptor::Descriptor{MR, NR, T, OneEFormat, PlanarFormat}
    function OneMKernel{MR, NR, T, W}(
            descriptor::Descriptor{MR, NR, T, OneEFormat, PlanarFormat}
        ) where {MR, NR, T, W}
        check_vector_kernel(OneMKernel, MR, W)
        return new{MR, NR, T, W}(descriptor)
    end
end

"""
    FMAddSubKernel(::Val{MR}, ::Val{NR}, ::Type{T}[, ::Val{W}])

Interleaved-accumulator complex microkernel using x86 `vfmaddsub`, over
`SIMD.Vec{W,real(T)}` lanes with `InterleavedFormat` A and `PlanarFormat` B.
`2MR` must be a multiple of `W`, and `W` must be even. The default for small-M
complex contractions on AVX-512.
"""
struct FMAddSubKernel{MR, NR, T, W} <: VectorKernel{MR, NR, T, W}
    descriptor::Descriptor{MR, NR, T, InterleavedFormat, PlanarFormat}
    function FMAddSubKernel{MR, NR, T, W}(
            descriptor::Descriptor{MR, NR, T, InterleavedFormat, PlanarFormat}
        ) where {MR, NR, T, W}
        check_vector_kernel(FMAddSubKernel, MR, W)
        return new{MR, NR, T, W}(descriptor)
    end
end

"""
    ComplexRealKernel(::Val{MR}, ::Val{NR}, ::Type{T}[, ::Val{W}])

Mixed-domain microkernel for complex `A` times real `B`, computing in complex
`T`. Interleaved A is a real `2MR x k_block_length` panel, so this is the real
`SIMDKernel` of `2MR x NR` over `SIMD.Vec{W,real(T)}` lanes, run verbatim at 2
real FMAs per complex MAC; the accumulator is 1m's lane-pair layout. `2MR`
must be a multiple of `W`, and `W` must be even.
"""
struct ComplexRealKernel{MR, NR, T, W} <: VectorKernel{MR, NR, T, W}
    descriptor::Descriptor{MR, NR, T, InterleavedFormat, RealFormat}
    function ComplexRealKernel{MR, NR, T, W}(
            descriptor::Descriptor{MR, NR, T, InterleavedFormat, RealFormat}
        ) where {MR, NR, T, W}
        check_vector_kernel(ComplexRealKernel, MR, W)
        return new{MR, NR, T, W}(descriptor)
    end
end

"""
    RealComplexKernel(::Val{MR}, ::Val{NR}, ::Type{T}[, ::Val{W}])

Mixed-domain microkernel for real `A` times complex `B`, computing in complex
`T`. Interleaved B is a real `k_block_length x 2NR` panel, so this is the real
`SIMDKernel` of `MR x 2NR` over `SIMD.Vec{W,real(T)}` lanes, run verbatim at 2
real FMAs per complex MAC; accumulator column `2j-1` holds the real and `2j`
the imaginary part of column `j`. `MR` must be a multiple of `W`.
"""
struct RealComplexKernel{MR, NR, T, W} <: VectorKernel{MR, NR, T, W}
    descriptor::Descriptor{MR, NR, T, RealFormat, InterleavedFormat}
    function RealComplexKernel{MR, NR, T, W}(
            descriptor::Descriptor{MR, NR, T, RealFormat, InterleavedFormat}
        ) where {MR, NR, T, W}
        check_vector_kernel(RealComplexKernel, MR, W)
        return new{MR, NR, T, W}(descriptor)
    end
end

"""
    pack_formats(K::Type{<:Microkernel}) -> (a_format, b_format)

The packing formats of every kernel of type `K`. A kernel type is also the
shape-free name of its complex-arithmetic scheme, on which blocking and the
shape menus dispatch:

  - `SIMDKernel` (and `ScalarKernel`): real.
  - `PlanarKernel`: split re/im planes, 4 real FMAs per complex MAC.
  - `OneMKernel`: Van Zee's 1m, a real `2MR x NR` kernel.
  - `FMAddSubKernel`: interleaved A, x86 `vfmaddsub`.
  - `ComplexRealKernel`: complex A, real B; the real kernel on `2MR` rows.
  - `RealComplexKernel`: real A, complex B; the real kernel on `2NR` columns.

No ranking is hardcoded: planar is the complex default, 1m and fmaddsub are
used only when named (plus fmaddsub for the AVX-512 small-M demotion,
`small_m_shape` in src/planning/kernel_selection.jl).
"""
pack_formats(::Type{<:Union{ScalarKernel, SIMDKernel}}) = (RealFormat(), RealFormat())
pack_formats(::Type{<:PlanarKernel}) = (PlanarFormat(), PlanarFormat())
pack_formats(::Type{<:OneMKernel}) = (OneEFormat(), PlanarFormat())
pack_formats(::Type{<:FMAddSubKernel}) = (InterleavedFormat(), PlanarFormat())
pack_formats(::Type{<:ComplexRealKernel}) = (InterleavedFormat(), RealFormat())
pack_formats(::Type{<:RealComplexKernel}) = (RealFormat(), InterleavedFormat())

"""
    reads_b_by_element(K::Type{<:Microkernel}) -> Bool

Whether the K step of every kernel of type `K` reads B only through
`b_scalar`/`b_complex`, so that any B source implementing those (such as
`UnpackedBView`, B read in place) can stand in for its packed B panel.
False otherwise: `ScalarKernel` loads from the panel directly, and 1m and
`RealComplexKernel` run a real kernel over the packed panel's reals.
"""
reads_b_by_element(::Type{<:Microkernel}) = false
reads_b_by_element(::Type{<:Union{SIMDKernel, PlanarKernel, FMAddSubKernel, ComplexRealKernel}}) = true

# Accumulator planes held live: planar keeps separate re/im planes.
accumulator_planes(::Type{<:Microkernel}) = 1
accumulator_planes(::Type{<:PlanarKernel}) = 2

AccumulatorLayout(::Type{<:SIMDKernel}) = RealLayout()
AccumulatorLayout(::Type{<:Union{PlanarKernel, RealComplexKernel}}) = SplitLayout()
AccumulatorLayout(::Type{<:Union{OneMKernel, FMAddSubKernel, ComplexRealKernel}}) = LanePairLayout()
AccumulatorLayout(kernel::VectorKernel) = AccumulatorLayout(typeof(kernel))

rows_per_vector(::AccumulatorLayout, W::Int) = W
rows_per_vector(::LanePairLayout, W::Int) = W ÷ 2
accumulator_length(::AccumulatorLayout, MV::Int, NR::Int) = MV * NR
accumulator_length(::SplitLayout, MV::Int, NR::Int) = 2 * MV * NR
# The accumulator vector of row block `v` (zero-based) of column `j`.
acc_index(MV::Int, v::Int, j::Int) = v + MV * (j - 1) + 1

(::Type{K})(::Val{MR}, ::Val{NR}, ::Type{T}, ::Val{W}) where {K <: VectorKernel, MR, NR, T, W} =
    K{MR, NR, T, W}(Descriptor(Val(MR), Val(NR), T, pack_formats(K)...))
(::Type{K})() where {MR, NR, T, W, K <: VectorKernel{MR, NR, T, W}} =
    K(Descriptor(Val(MR), Val(NR), T, pack_formats(K)...))
(::Type{K})(::Val{MR}, ::Val{NR}, ::Type{T}) where {K <: VectorKernel, MR, NR, T} =
    K(Val(MR), Val(NR), T, Val(default_lanewidth(real(T))))

lanewidth(::VectorKernel{MR, NR, T, W}) where {MR, NR, T, W} = W

k_steps(::VectorKernel, k_block_length::Int) = k_block_length
k_steps(::OneMKernel, k_block_length::Int) = 2 * k_block_length

# The real kernel whose K step a kernel without one of its own runs.
inner(::OneMKernel{MR, NR, T, W}) where {MR, NR, T, W} = SIMDKernel(Val(2 * MR), Val(NR), real(T), Val(W))
inner(::ComplexRealKernel{MR, NR, T, W}) where {MR, NR, T, W} = SIMDKernel(Val(2 * MR), Val(NR), real(T), Val(W))
inner(::RealComplexKernel{MR, NR, T, W}) where {MR, NR, T, W} = SIMDKernel(Val(MR), Val(2 * NR), real(T), Val(W))
