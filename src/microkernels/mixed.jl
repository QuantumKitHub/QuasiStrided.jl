# Mixed-domain kernels (BLIS's mixed-domain method): a complex operand times a
# real one is a real product once the complex operand is viewed as real, so
# both run the real `SIMDKernel` verbatim, at 2 real FMAs per complex MAC.
#   * complex A x real B: interleaved A is a real `2MR x kc` panel; the
#     accumulator is 1m/fmaddsub's interleaved layout.
#   * real A x complex B: interleaved B is a real `kc x 2NR` panel; accumulator
#     column `2j` holds the real and `2j+1` the imaginary part of column `j`.

using SIMD: Vec

"""
    ComplexRealKernel(::Val{MR}, ::Val{NR}, ::Type{T}[, ::Val{W}])

Mixed-domain microkernel for complex `A` times real `B`, computing in complex
`T`: a real `SIMDKernel` of `2MR x NR` over `SIMD.Vec{W,real(T)}` lanes. `2MR`
must be a multiple of `W`, and `W` must be even.
"""
struct ComplexRealKernel{MR, NR, T, W, KI <: SIMDKernel} <: DescriptorKernel{MR, NR, T}
    descriptor::ComplexKernelDescriptor{MR, NR, T, InterleavedFormat, RealFormat}
    inner::KI

    function ComplexRealKernel{MR, NR, T, W, KI}(
            descriptor::ComplexKernelDescriptor{MR, NR, T, InterleavedFormat, RealFormat},
            inner::KI
        ) where {MR, NR, T, W, KI <: SIMDKernel}
        _check_vector_shape("ComplexRealKernel", 2 * MR, W, true)
        KI === SIMDKernel{2 * MR, NR, real(T), W} || throw(
            ArgumentError("ComplexRealKernel's inner kernel must be SIMDKernel{$(2 * MR),$NR,$(real(T)),$W}, got $KI")
        )
        return new{MR, NR, T, W, KI}(descriptor, inner)
    end
end

"""
    RealComplexKernel(::Val{MR}, ::Val{NR}, ::Type{T}[, ::Val{W}])

Mixed-domain microkernel for real `A` times complex `B`, computing in complex
`T`: a real `SIMDKernel` of `MR x 2NR` over `SIMD.Vec{W,real(T)}` lanes. `MR`
must be a multiple of `W`.
"""
struct RealComplexKernel{MR, NR, T, W, KI <: SIMDKernel} <: DescriptorKernel{MR, NR, T}
    descriptor::ComplexKernelDescriptor{MR, NR, T, RealFormat, InterleavedFormat}
    inner::KI

    function RealComplexKernel{MR, NR, T, W, KI}(
            descriptor::ComplexKernelDescriptor{MR, NR, T, RealFormat, InterleavedFormat},
            inner::KI
        ) where {MR, NR, T, W, KI <: SIMDKernel}
        _check_vector_shape("RealComplexKernel", MR, W)
        KI === SIMDKernel{MR, 2 * NR, real(T), W} || throw(
            ArgumentError("RealComplexKernel's inner kernel must be SIMDKernel{$MR,$(2 * NR),$(real(T)),$W}, got $KI")
        )
        return new{MR, NR, T, W, KI}(descriptor, inner)
    end
end

const _MixedKernel = Union{ComplexRealKernel, RealComplexKernel}

function ComplexRealKernel(::Val{MR}, ::Val{NR}, ::Type{T}, ::Val{W}) where {MR, NR, T, W}
    T <: Complex ||
        throw(ArgumentError("ComplexRealKernel requires a complex element type, got $T"))
    descriptor = ComplexKernelDescriptor(Val(MR), Val(NR), T, InterleavedFormat(), RealFormat())
    inner = SIMDKernel(Val(2 * MR), Val(NR), real(T), Val(W))
    return ComplexRealKernel{MR, NR, T, W, typeof(inner)}(descriptor, inner)
end

function RealComplexKernel(::Val{MR}, ::Val{NR}, ::Type{T}, ::Val{W}) where {MR, NR, T, W}
    T <: Complex ||
        throw(ArgumentError("RealComplexKernel requires a complex element type, got $T"))
    descriptor = ComplexKernelDescriptor(Val(MR), Val(NR), T, RealFormat(), InterleavedFormat())
    inner = SIMDKernel(Val(MR), Val(2 * NR), real(T), Val(W))
    return RealComplexKernel{MR, NR, T, W, typeof(inner)}(descriptor, inner)
end

function ComplexRealKernel(::Val{MR}, ::Val{NR}, ::Type{T}) where {MR, NR, T}
    T <: Complex ||
        throw(ArgumentError("ComplexRealKernel requires a complex element type, got $T"))
    return ComplexRealKernel(Val(MR), Val(NR), T, Val(_default_lanewidth(real(T))))
end
function RealComplexKernel(::Val{MR}, ::Val{NR}, ::Type{T}) where {MR, NR, T}
    T <: Complex ||
        throw(ArgumentError("RealComplexKernel requires a complex element type, got $T"))
    return RealComplexKernel(Val(MR), Val(NR), T, Val(_default_lanewidth(real(T))))
end

complex_method(::ComplexRealKernel) = ComplexRealMethod()
complex_method(::RealComplexKernel) = RealComplexMethod()
lanewidth(kernel::_MixedKernel) = lanewidth(kernel.inner)

zero_accumulator(kernel::_MixedKernel) = zero_accumulator(kernel.inner)

@inline Base.accumulate(
    kernel::K, acc::NTuple{NV, Vec{W, R}}, packed_a::PA, packed_b::PB, kc::Int
) where {K <: _MixedKernel, NV, W, R, PA, PB} =
    accumulate(kernel.inner, acc, packed_a, packed_b, kc)

# Inlined or not as the fmaddsub and planar stores each reuses.
function store_tile!(
        destination::QSTile, acc::NTuple{NV, Vec{W, R}},
        alpha::T, beta::T, kernel::ComplexRealKernel{MR, NR, T, W}
    ) where {MR, NR, T, W, R, NV}
    m, n = _store_prologue!(destination, alpha, beta)
    (m == 0 || n == 0) && return destination

    if _complex_vector_eligible(destination, T)
        return _store_tile_fmaddsub_vector!(destination, acc, alpha, beta, kernel, m, n)
    end

    return _store_tile_lanepair!(destination, acc, alpha, beta, kernel, m, n)
end

@inline function store_tile!(
        destination::QSTile, acc::NTuple{NV, Vec{W, R}},
        alpha::T, beta::T, kernel::RealComplexKernel{MR, NR, T, W}
    ) where {MR, NR, T, W, R, NV}
    m, n = _store_prologue!(destination, alpha, beta)
    (m == 0 || n == 0) && return destination

    if _complex_vector_eligible(destination, T)
        return _store_tile_planar_vector!(destination, acc, alpha, beta, kernel, m, n)
    end

    return _store_tile_planar!(destination, acc, alpha, beta, kernel, m, n)
end
