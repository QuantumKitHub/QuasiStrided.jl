# 1m (Van Zee's induced method): the real `SIMDKernel` of `2MR x NR`, run over
# `2*k_block_length` real K steps against 1e-packed A and planar B. At logical K
# step `p`, `OneEFormat` A holds `(re_0, im_0, re_1, im_1, ...)` then `(-im_0,
# re_0, -im_1, re_1, ...)`, and planar B `re_0..` then `im_0..`, so accumulator
# real row `2i-1` is the real and `2i` the imaginary part of complex row `i`.
# There is deliberately no FMA loop here: 1m reuses the real kernel body
# verbatim.

using SIMD: Vec

"""
    OneMKernel(::Val{MR}, ::Val{NR}, ::Type{T}[, ::Val{W}])

Van Zee's induced 1m complex microkernel: a real `SIMDKernel` of `2MR x NR`
over `SIMD.Vec{W,real(T)}` lanes. `2MR` must be a multiple of `W`, and `W`
must be even. Used only when named in `plan_contract(...; kernel = ...)`.
"""
struct OneMKernel{MR, NR, T, W, KI <: SIMDKernel} <: DescriptorKernel{MR, NR, T}
    descriptor::Descriptor{MR, NR, T, OneEFormat, PlanarFormat}
    # `KI` stands in for `SIMDKernel{2MR,NR,real(T),W}`, which is not a legal
    # field type; the constructor pins it to exactly that.
    inner::KI

    function OneMKernel{MR, NR, T, W, KI}(
            descriptor::Descriptor{MR, NR, T, OneEFormat, PlanarFormat},
            inner::KI
        ) where {MR, NR, T, W, KI <: SIMDKernel}
        _check_vector_shape("OneMKernel", 2 * MR, W, true)
        KI === SIMDKernel{2 * MR, NR, real(T), W} || throw(
            ArgumentError("OneMKernel's inner kernel must be SIMDKernel{$(2 * MR),$NR,$(real(T)),$W}, got $KI")
        )
        return new{MR, NR, T, W, KI}(descriptor, inner)
    end
end

function OneMKernel(::Val{MR}, ::Val{NR}, ::Type{T}, ::Val{W}) where {MR, NR, T, W}
    T <: Complex ||
        throw(ArgumentError("OneMKernel requires a complex element type, got $T"))
    descriptor = Descriptor(Val(MR), Val(NR), T, OneEFormat(), PlanarFormat())
    inner = SIMDKernel(Val(2 * MR), Val(NR), real(T), Val(W))
    return OneMKernel{MR, NR, T, W, typeof(inner)}(descriptor, inner)
end
function OneMKernel(::Val{MR}, ::Val{NR}, ::Type{T}) where {MR, NR, T}
    T <: Complex ||
        throw(ArgumentError("OneMKernel requires a complex element type, got $T"))
    return OneMKernel(Val(MR), Val(NR), T, Val(_default_lanewidth(real(T))))
end

complex_method(::OneMKernel) = OneMMethod()
lanewidth(::OneMKernel{MR, NR, T, W}) where {MR, NR, T, W} = W

zero_accumulator(kernel::OneMKernel) = zero_accumulator(kernel.inner)

function Base.accumulate(
        kernel::OneMKernel{MR, NR, T, W}, acc::NTuple{NV, Vec{W, R}},
        packed_a::PA, packed_b::PB, k_block_length::Int
    ) where {MR, NR, T, W, R, NV, PA <: PackedPanel, PB}
    # Checked here so the message reports the logical k_block_length, not the
    # doubled one.
    k_block_length >= 0 || throw(ArgumentError("accumulate requires k_block_length >= 0, got k_block_length = $k_block_length"))
    return accumulate(kernel.inner, acc, packed_a, packed_b, 2 * k_block_length)
end

# Scalar store for an interleaved accumulator (1m and fmaddsub): with `W` even
# and `2MR` a multiple of `W`, complex row `i = v*(W÷2) + u` is lanes `2u-1`
# (re) and `2u` (im) of vector `v` (zero-based) -- never split across two
# vectors.
@generated function _store_tile_lanepair!(
        destination::Tile, acc::NTuple{NV, Vec{W, R}},
        alpha::T, beta::T, kernel::DescriptorKernel{MR, NR, T},
        m::Int, n::Int
    ) where {MR, NR, T, W, R, NV}
    iseven(W) || throw(ArgumentError("_store_tile_lanepair!: requires an even W, got $W"))
    MV = (2 * MR) ÷ W
    _check_acc(:_store_tile_lanepair!, R, T, NV, MV * NR)
    HW = W ÷ 2

    blocks = Any[]
    for j in 1:NR, v in 0:(MV - 1)
        idx = v + MV * (j - 1) + 1
        push!(
            blocks, quote
                if $j <= n
                    vec = acc[$idx]
                    for u in 1:$HW
                        i = $(v * HW) + u
                        i <= m || break
                        _axpby_tile!(
                            destination, i, $j, alpha,
                            Complex(vec[2 * u - 1], vec[2 * u]), beta
                        )
                    end
                end
            end
        )
    end
    return quote
        @inbounds begin
            $(blocks...)
        end
        return destination
    end
end

# Scalar only: 1m has no vector store. Not `@inline`, as for fmaddsub.
function store_tile!(
        destination::Tile, acc::NTuple{NV, Vec{W, R}},
        alpha::T, beta::T, kernel::OneMKernel{MR, NR, T, W}
    ) where {MR, NR, T, W, R, NV}
    m, n = _store_prologue!(destination, alpha, beta)
    (m == 0 || n == 0) && return destination
    return _store_tile_lanepair!(destination, acc, alpha, beta, kernel, m, n)
end
