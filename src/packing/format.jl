# The contract between packing and the microkernels: the packed-panel formats
# and the descriptor fixing a kernel's register tile, element type and panel
# formats. Complex panels are "N planes of real(T)". A sliver of register
# extent `R` (`MR` for A, `NR` for B) puts lane `i` of plane `plane` at K step
# `p` (one-based `i`, `p`) at zero-based offset, in reals,
#
#     (p - 1) * R * reals_per_element  +  plane * R  +  (i - 1)

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
        check_descriptor(MR, NR, T, FA, FB)
        return new{MR, NR, T, FA, FB}()
    end
end

const RealDescriptor{MR, NR, T} = Descriptor{MR, NR, T, RealFormat, RealFormat}

Descriptor(
    ::Val{MR}, ::Val{NR}, ::Type{T}, a_format::PackFormat = RealFormat(), b_format::PackFormat = RealFormat()
) where {MR, NR, T} = Descriptor{MR, NR, T, typeof(a_format), typeof(b_format)}()

function check_descriptor(MR, NR, T, FA, FB)
    real_formats = FA === RealFormat && FB === RealFormat
    MR isa Int && NR isa Int ||
        throw(ArgumentError("Descriptor requires Int type parameters MR, NR"))
    MR > 0 || throw(ArgumentError("Descriptor requires MR > 0, got MR = $MR"))
    NR > 0 || throw(ArgumentError("Descriptor requires NR > 0, got NR = $NR"))
    if real_formats
        T === Float32 || T === Float64 ||
            throw(ArgumentError("Descriptor with real formats requires T ∈ (Float32, Float64), got $T"))
    else
        # A real operand of a complex `T` only in the mixed-domain pairings.
        mixed = (FA, FB) === (InterleavedFormat, RealFormat) ||
            (FA, FB) === (RealFormat, InterleavedFormat)
        !mixed && (FA === RealFormat || FB === RealFormat) && throw(
            ArgumentError(
                "Descriptor requires complex formats on both operands, " *
                    "or (InterleavedFormat, RealFormat) / (RealFormat, InterleavedFormat), got ($FA, $FB)"
            )
        )
        T === ComplexF32 || T === ComplexF64 || throw(
            ArgumentError("Descriptor with complex formats requires T in (ComplexF32, ComplexF64), got $T")
        )
    end
    return nothing
end

tile_size(::Descriptor{MR, NR}) where {MR, NR} = (MR, NR)
tile_size(k, i::Int) = tile_size(k)[i]
scalartype(::Descriptor{MR, NR, T}) where {MR, NR, T} = T
a_format(::Descriptor{MR, NR, T, FA, FB}) where {MR, NR, T, FA, FB} = FA()
b_format(::Descriptor{MR, NR, T, FA, FB}) where {MR, NR, T, FA, FB} = FB()
# Eltype of the packed buffers and `Vec` lanes: not `scalartype` for complex T.
realtype(::Descriptor{MR, NR, T}) where {MR, NR, T} = real(T)

# Reals per A / B sliver per logical K step.
sliver_width(d::Descriptor{MR, NR}) where {MR, NR} =
    (MR * reals_per_element(a_format(d)), NR * reals_per_element(b_format(d)))
sliver_width(k, i::Int) = sliver_width(k)[i]

packed_a_length(d::Descriptor, k_block_length::Int) = sliver_width(d, 1) * k_block_length
packed_b_length(d::Descriptor, k_block_length::Int) = sliver_width(d, 2) * k_block_length

# Zero-based offsets, in reals, of one-based coordinates. For 1e and
# interleaved, `i`/`j` runs over reals (1:2MR) and 1e's second region is
# `plane == 2`.
@inline packed_a_offset(d::Descriptor{MR}, i::Int, p::Int, plane::Int = 0) where {MR} =
    (p - 1) * sliver_width(d, 1) + plane * MR + (i - 1)
@inline packed_b_offset(d::Descriptor{MR, NR}, j::Int, p::Int, plane::Int = 0) where {MR, NR} =
    (p - 1) * sliver_width(d, 2) + plane * NR + (j - 1)
