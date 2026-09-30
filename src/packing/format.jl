# The contract between packing and the microkernels: the packed-panel formats
# and the descriptor fixing a kernel's register tile, element type and panel
# formats. Complex panels are "N planes of real(T)"; every format is addressed by
#
#     p * MR * reals_per_element  +  plane * MR  +  i
#
# which at RealFormat (one real per element, plane 0) reduces to `i + MR*p`.

abstract type PackFormat end

# One real per element.
struct RealFormat <: PackFormat end
# BLIS "1r": per K step all real parts, then all imaginary parts.
struct PlanarFormat <: PackFormat end
# BLIS "1e": per K step the real 2x2 block [[re, -im], [im, re]], stored as two
# real K steps of 2*MR reals (`[re, im, ...]` then `[-im, re, ...]`).
struct OneEFormat <: PackFormat end
# `Complex{T}`'s native `[re, im, ...]` order: 1e's first region alone (the
# fmaddsub kernel derives the second in registers).
struct InterleavedFormat <: PackFormat end

reals_per_element(::RealFormat) = 1
reals_per_element(::PlanarFormat) = 2
reals_per_element(::InterleavedFormat) = 2
reals_per_element(::OneEFormat) = 4

# `MR`/`NR` are logical extents (complex elements for a complex `T`); every
# length and offset below takes the logical `k_block_length` and counts reals.
struct Descriptor{MR, NR, T, FA <: PackFormat, FB <: PackFormat}
    function Descriptor{MR, NR, T, FA, FB}() where {MR, NR, T, FA, FB}
        _check_descriptor(MR, NR, T, FA, FB)
        return new{MR, NR, T, FA, FB}()
    end
end

const KernelDescriptor{MR, NR, T} = Descriptor{MR, NR, T, RealFormat, RealFormat}
const ComplexKernelDescriptor{MR, NR, T <: Complex, FA, FB} = Descriptor{MR, NR, T, FA, FB}

KernelDescriptor(::Val{MR}, ::Val{NR}, ::Type{T}) where {MR, NR, T} = KernelDescriptor{MR, NR, T}()

function ComplexKernelDescriptor(
        ::Val{MR}, ::Val{NR}, ::Type{T}, ::FA, ::FB
    ) where {MR, NR, T, FA <: PackFormat, FB <: PackFormat}
    return Descriptor{MR, NR, T, FA, FB}()
end

function _check_descriptor(MR, NR, T, FA, FB)
    real_formats = FA === RealFormat && FB === RealFormat
    name = real_formats ? "KernelDescriptor" : "ComplexKernelDescriptor"
    MR isa Int && NR isa Int ||
        throw(ArgumentError("$name requires Int type parameters MR, NR"))
    MR > 0 || throw(ArgumentError("$name requires MR > 0, got MR = $MR"))
    NR > 0 || throw(ArgumentError("$name requires NR > 0, got NR = $NR"))
    if real_formats
        T === Float32 || T === Float64 ||
            throw(ArgumentError("KernelDescriptor requires T ∈ (Float32, Float64), got $T"))
    else
        # A real operand of a complex `T` only in the mixed-domain pairings.
        mixed = (FA, FB) === (InterleavedFormat, RealFormat) ||
            (FA, FB) === (RealFormat, InterleavedFormat)
        !mixed && (FA === RealFormat || FB === RealFormat) && throw(
            ArgumentError(
                "ComplexKernelDescriptor requires complex formats on both operands, " *
                    "or (InterleavedFormat, RealFormat) / (RealFormat, InterleavedFormat), got ($FA, $FB)"
            )
        )
        T === ComplexF32 || T === ComplexF64 || throw(
            ArgumentError("ComplexKernelDescriptor requires T in (ComplexF32, ComplexF64), got $T")
        )
    end
    return nothing
end

tile_size(::Descriptor{MR, NR}) where {MR, NR} = (MR, NR)
scalartype(::Descriptor{MR, NR, T}) where {MR, NR, T} = T
a_format(::Descriptor{MR, NR, T, FA, FB}) where {MR, NR, T, FA, FB} = FA()
b_format(::Descriptor{MR, NR, T, FA, FB}) where {MR, NR, T, FA, FB} = FB()
# Eltype of the packed buffers and `Vec` lanes: not `scalartype` for complex T.
realtype(::Descriptor{MR, NR, T}) where {MR, NR, T} = real(T)

# Reals per A / B sliver per logical K step.
sliver_widths(d::Descriptor{MR, NR}) where {MR, NR} =
    (MR * reals_per_element(a_format(d)), NR * reals_per_element(b_format(d)))

packed_a_length(d::Descriptor, k_block_length::Int) = sliver_widths(d)[1] * k_block_length
packed_b_length(d::Descriptor, k_block_length::Int) = sliver_widths(d)[2] * k_block_length

# Zero-based offsets, in reals. For 1e and interleaved, `i`/`j` runs over reals
# (0:2MR-1) and 1e's second region is `plane == 2`.
@inline packed_a_plane_offset(d::Descriptor{MR}, plane::Int, i::Int, p::Int) where {MR} =
    p * sliver_widths(d)[1] + plane * MR + i
@inline packed_b_plane_offset(d::Descriptor{MR, NR}, plane::Int, j::Int, p::Int) where {MR, NR} =
    p * sliver_widths(d)[2] + plane * NR + j

# Real descriptors only: a complex element has no single offset.
packed_a_offset(kernel::KernelDescriptor{MR}, i::Int, p::Int) where {MR} = i + MR * p
packed_b_offset(kernel::KernelDescriptor{MR, NR}, j::Int, p::Int) where {MR, NR} = j + NR * p
