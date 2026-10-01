# The vector kernels' types, their accumulator layouts and the generator-time
# pieces of the stores, ahead of every implementation: a `@generated`
# function's generator only sees methods defined before it.

using SIMD: Vec

"""
    VectorKernel{MR, NR, T, W}

An explicit-SIMD microkernel over `SIMD.Vec{W, real(T)}` lanes. Its
accumulator is an immutable tuple of vectors in one of three layouts
([`accumulator_layout`](@ref)), which fixes the stores; each kernel supplies
its K step, `accumulate_step`, and the generic `add_tile` loops it
`k_steps(kernel, k_block_length)` times.
"""
abstract type VectorKernel{MR, NR, T, W} <: Microkernel{MR, NR, T} end

"""
    SIMDKernel(::Val{MR}, ::Val{NR}, ::Type{T}[, ::Val{W}])

Explicit-SIMD real microkernel over `SIMD.Vec{W,T}` lanes. `MR` must be a
multiple of `W` (default: one 256-bit register, 4 for `Float64`, 8 for `Float32`).
"""
struct SIMDKernel{MR, NR, T, W} <: VectorKernel{MR, NR, T, W}
    descriptor::RealDescriptor{MR, NR, T}
end

"""
    PlanarKernel(::Val{MR}, ::Val{NR}, ::Type{T}[, ::Val{W}])

Split-complex microkernel for `T = ComplexF32/ComplexF64` over
`SIMD.Vec{W,real(T)}` lanes. `MR`/`NR` are complex extents; `MR` must be a
multiple of `W`, which counts reals.
"""
struct PlanarKernel{MR, NR, T, W} <: VectorKernel{MR, NR, T, W}
    descriptor::Descriptor{MR, NR, T, PlanarFormat, PlanarFormat}
end

"""
    OneMKernel(::Val{MR}, ::Val{NR}, ::Type{T}[, ::Val{W}])

Van Zee's induced 1m complex microkernel: a real `SIMDKernel` of `2MR x NR`
over `SIMD.Vec{W,real(T)}` lanes. `2MR` must be a multiple of `W`, and `W`
must be even. Used only when named in `plan_contract(...; kernel = ...)`.
"""
struct OneMKernel{MR, NR, T, W} <: VectorKernel{MR, NR, T, W}
    descriptor::Descriptor{MR, NR, T, OneEFormat, PlanarFormat}
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
end

"""
    ComplexRealKernel(::Val{MR}, ::Val{NR}, ::Type{T}[, ::Val{W}])

Mixed-domain microkernel for complex `A` times real `B`, computing in complex
`T`: a real `SIMDKernel` of `2MR x NR` over `SIMD.Vec{W,real(T)}` lanes. `2MR`
must be a multiple of `W`, and `W` must be even.
"""
struct ComplexRealKernel{MR, NR, T, W} <: VectorKernel{MR, NR, T, W}
    descriptor::Descriptor{MR, NR, T, InterleavedFormat, RealFormat}
end

"""
    RealComplexKernel(::Val{MR}, ::Val{NR}, ::Type{T}[, ::Val{W}])

Mixed-domain microkernel for real `A` times complex `B`, computing in complex
`T`: a real `SIMDKernel` of `MR x 2NR` over `SIMD.Vec{W,real(T)}` lanes. `MR`
must be a multiple of `W`.
"""
struct RealComplexKernel{MR, NR, T, W} <: VectorKernel{MR, NR, T, W}
    descriptor::Descriptor{MR, NR, T, RealFormat, InterleavedFormat}
end

complex_method(::Type{<:SIMDKernel}) = RealMethod()
complex_method(::Type{<:PlanarKernel}) = PlanarMethod()
complex_method(::Type{<:OneMKernel}) = OneMMethod()
complex_method(::Type{<:FMAddSubKernel}) = FMAddSubMethod()
complex_method(::Type{<:ComplexRealKernel}) = ComplexRealMethod()
complex_method(::Type{<:RealComplexKernel}) = RealComplexMethod()
complex_method(kernel::VectorKernel) = complex_method(typeof(kernel))

"""
    accumulator_layout(kernel::VectorKernel)

How the accumulator holds the tile, as `(MR ÷ rows) * NR` vectors of `rows`
consecutive tile rows each, column-major (`rows_per_vector`):

  - `RealLayout()`: one real per lane (`SIMDKernel`).
  - `SplitLayout()`: separate re and im vectors (`PlanarKernel`: a real and an
    imaginary plane in one flat tuple; `RealComplexKernel`: the real kernel's
    columns `2j-1` and `2j`).
  - `LanePairLayout()`: lanes `2u-1` and `2u` are re and im of one row
    (1m, fmaddsub, `ComplexRealKernel`).
"""
function accumulator_layout end

abstract type AccumulatorLayout end
struct RealLayout <: AccumulatorLayout end
struct SplitLayout <: AccumulatorLayout end
struct LanePairLayout <: AccumulatorLayout end

accumulator_layout(::Type{<:SIMDKernel}) = RealLayout()
accumulator_layout(::Type{<:Union{PlanarKernel, RealComplexKernel}}) = SplitLayout()
accumulator_layout(::Type{<:Union{OneMKernel, FMAddSubKernel, ComplexRealKernel}}) = LanePairLayout()
accumulator_layout(kernel::VectorKernel) = accumulator_layout(typeof(kernel))

rows_per_vector(::AccumulatorLayout, W::Int) = W
rows_per_vector(::LanePairLayout, W::Int) = W ÷ 2
accumulator_length(::AccumulatorLayout, MV::Int, NR::Int) = MV * NR
accumulator_length(::SplitLayout, MV::Int, NR::Int) = 2 * MV * NR

function (::Type{K})(::Val{MR}, ::Val{NR}, ::Type{T}, ::Val{W}) where {K <: VectorKernel, MR, NR, T, W}
    method = complex_method(K)
    method === RealMethod() || T <: Complex ||
        throw(ArgumentError("$(nameof(K)) requires a complex element type, got $T"))
    pairs = accumulator_layout(K) isa LanePairLayout
    check_vector_shape(nameof(K), pairs ? 2 * MR : MR, W, pairs)
    return K{MR, NR, T, W}(Descriptor(Val(MR), Val(NR), T, pack_formats(method)...))
end
(::Type{K})(::Val{MR}, ::Val{NR}, ::Type{T}) where {K <: VectorKernel, MR, NR, T} =
    K(Val(MR), Val(NR), T, Val(default_lanewidth(real(T))))

lanewidth(::VectorKernel{MR, NR, T, W}) where {MR, NR, T, W} = W

# Real and split stores are `@inline`: out of line, the whole accumulator is
# spilled to the stack and reloaded on every micro-tile. The lane-pair store is
# not: inlining it cost time (code growth). The scalar stores never are: they
# are scalar anyway, and inlining them bloats the tile function.
inline_store(::AccumulatorLayout) = true
inline_store(::LanePairLayout) = false

# Generator-time pieces of the stores. `acc_bindings` names the accumulator
# vector(s) of row block `v` (zero-based) of column `j`, `lane_value` is the
# element at row `lane` of the block, `block_store` stores a full block whose
# first row is at zero-based storage index `first`, for `beta` case `B`
# (`:zero`, `:one` or `:general`).
acc_index(MV::Int, v::Int, j::Int) = v + MV * (j - 1) + 1
split_index(::Type{<:PlanarKernel}, MV::Int, NR::Int, v::Int, j::Int) =
    (acc_index(MV, v, j), MV * NR + acc_index(MV, v, j))
split_index(::Type{<:RealComplexKernel}, MV::Int, NR::Int, v::Int, j::Int) =
    (v + MV * (2j - 2) + 1, v + MV * (2j - 1) + 1)

acc_bindings(::AccumulatorLayout, kernel::Type, MV::Int, NR::Int, v::Int, j::Int) =
    :(vec = acc[$(acc_index(MV, v, j))])
function acc_bindings(::SplitLayout, kernel::Type, MV::Int, NR::Int, v::Int, j::Int)
    ire, iim = split_index(kernel, MV, NR, v, j)
    return quote
        revec = acc[$ire]
        imvec = acc[$iim]
    end
end

lane_value(::RealLayout) = :(vec[lane])
lane_value(::SplitLayout) = :(Complex(revec[lane], imvec[lane]))
lane_value(::LanePairLayout) = :(Complex(vec[2 * lane - 1], vec[2 * lane]))

function block_store(::RealLayout, first, W::Int, R::Type, RC::Type, B::Symbol)
    old = :(convert(Vec{$W, $R}, vload(Vec{$W, $RC}, storage, at)))
    new = B === :zero ? :(alpha * vec) : B === :one ? :(muladd(alpha, vec, $old)) : :(muladd(alpha, vec, beta * $old))
    return quote
        at = $first + 1
        vstore(convert(Vec{$W, $RC}, $new), storage, at)
    end
end
block_store(::SplitLayout, first, W::Int, R::Type, RC::Type, B::Symbol) =
    :(split_store_block!(sp, 2 * $first, revec, imvec, ar, ai, br, bi, Val($W), Val($(QuoteNode(B)))))
block_store(::LanePairLayout, first, W::Int, R::Type, RC::Type, B::Symbol) =
    :(lanepair_store_block!(sp, 2 * $first, vec, ar, ai, br, bi, Val($(QuoteNode(B)))))

# The real lane type of the vector store's storage `S`. The complex layouts
# reinterpret the storage as reals, only sound on dense rank-1 complex storage.
store_lanetype(::RealLayout, S::Type, T::Type) = eltype(S)
function store_lanetype(::AccumulatorLayout, S::Type, T::Type)
    S <: DenseVector && lane_convertible(eltype(S), T) ||
        throw(ArgumentError("vector_store!: storage $S is not a dense vector convertible to $T"))
    return real(eltype(S))
end

# The vector store's body around its blocks: the complex layouts broadcast
# `alpha`/`beta` once and store through a raw pointer, only dereferenced
# inside `GC.@preserve`.
store_body(::RealLayout, W::Int, R::Type, RC::Type, body) = :(@inbounds $body)
store_body(::AccumulatorLayout, W::Int, R::Type, RC::Type, body) = quote
    ar = Vec{$W, $R}(real(alpha))
    ai = Vec{$W, $R}(imag(alpha))
    br = Vec{$W, $R}(real(beta))
    bi = Vec{$W, $R}(imag(beta))
    GC.@preserve storage begin
        sp = reinterpret(Ptr{$RC}, pointer(storage))
        @inbounds $body
    end
end
