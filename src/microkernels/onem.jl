# 1m (Van Zee's induced method): the *real* microkernel, run over `2*kc` real
# K steps against "1e"-packed A and "1r"-packed B.
#
# There is deliberately NO FMA loop in this file. `accumulate` is
#
#     accumulate(kernel.inner, acc, packed_a, packed_b, 2 * kc)
#
# i.e. one kernel body written once and parameterised, so that a planar-vs-1m
# measurement compares two *methods* rather than two hand-tunings. Writing an
# inner loop here would invalidate that comparison.
#
# Why the real kernel computes the right thing: at logical K step `p`, 1e-packed
# A holds two real K steps `(re_0, im_0, re_1, im_1, ...)` then
# `(-im_0, re_0, -im_1, re_1, ...)`, and planar B holds `re_0..re_{NR-1}` then
# `im_0..im_{NR-1}`. A real `2MR x NR` kernel walks both linearly, so real row
# `2t` accumulates `re*re - im*im` and row `2t+1` accumulates `im*re + re*im`:
# accumulator real row `2i` is the real part and `2i+1` the imaginary part of
# complex row `i`. The `2*kc` doubling is confined to this file's own
# delegation and never appears in a length, an offset or a driver loop bound.
#
# Cliff B (Julia heap-allocating a dynamically indexed `NTuple` above NV = 16;
# see src/microkernels/simd.jl) applies here exactly as it does to planar: the
# store below is `@generated` with literal tuple indices including the lane
# tail. `accumulate` inherits the real kernel's already-`@generated` body.

using SIMD: Vec

"""
    OneMKernel{MR,NR,T,W,KI}(descriptor, inner)
    OneMKernel(::Val{MR}, ::Val{NR}, ::Type{T}, ::Val{W})
    OneMKernel(::Val{MR}, ::Val{NR}, ::Type{T})

Van Zee's induced 1m microkernel, implementing [`OneMMethod`](@ref): a real
`SIMDKernel` of shape `2MR x NR` over `SIMD.Vec{W,real(T)}` lanes, driven over
`2*kc` real K steps by [`OneEFormat`](@ref) A and [`PlanarFormat`](@ref) B.

`MR`/`NR` are the **logical** (complex) register-tile extents and `T` is the
**storage** element type (`ComplexF32`/`ComplexF64`); `W` is the lane count of
the *real* type. Note the shape bookkeeping: `mr(kernel) == MR` while
`mr(kernel.inner) == 2MR`, and it is `2MR` -- not `MR` -- that must be a
multiple of `W`. That is why the shipped menu contains `MR = 12` at `W = 8`:
`2*12 = 24` is the multiple. `W` must additionally be **even** (see the
constructor).

The field is named `descriptor`, so `mr`/`nr`/`scalartype`/`packed_a_length`/
`packed_b_length`/`realtype`/`packed_a_per_k`/`packed_b_per_k`/`a_format`/
`b_format` all forward through the `DescriptorKernel` methods in
src/microkernels/interface.jl.

`KI` exists only because Julia cannot compute a field type from type
parameters: the intended field `inner::SIMDKernel{2MR,NR,real(T),W}` is
not a legal struct field type (`2MR` and `real(T)` are computations on
`TypeVar`s). `KI` carries that computed type instead, and the inner constructor
pins it to exactly `SIMDKernel{2MR,NR,real(T),W}`, so nothing else can be
stored there. `OneMKernel{MR,NR,T,W}` remains a usable (partially applied)
spelling for `isa` and for dispatch.

The 3-argument form defaults `W` via `_default_lanewidth(real(T))`.
"""
struct OneMKernel{MR, NR, T, W, KI <: SIMDKernel} <: DescriptorKernel{MR, NR, T}
    descriptor::ComplexKernelDescriptor{MR, NR, T, OneEFormat, PlanarFormat}
    inner::KI

    function OneMKernel{MR, NR, T, W, KI}(
            descriptor::ComplexKernelDescriptor{MR, NR, T, OneEFormat, PlanarFormat},
            inner::KI
        ) where {MR, NR, T, W, KI <: SIMDKernel}
        _check_lanewidth("OneMKernel", W)
        # GUARDRAIL: even `W` is load-bearing, not cosmetic. An ODD `W` can
        # satisfy `mod(2MR, W) == 0` (MR = 3, W = 3 does) while a complex row
        # straddles a vector boundary, silently breaking the tile reader below,
        # which reads complex row `i` from lanes `2u+1`/`2u+2` of ONE vector.
        # Wrong answer, not an error -- hence checked here.
        iseven(W) ||
            throw(
            ArgumentError(
                "OneMKernel requires an even vector width W (the two halves of a " *
                    "complex row must be adjacent lanes of one Vec), got W = $W"
            )
        )
        mod(2 * MR, W) == 0 ||
            throw(
            ArgumentError(
                "OneMKernel requires 2*mr(kernel) = $(2 * MR) to be a multiple of " *
                    "the vector width W = $W (1m runs a REAL microkernel of 2*MR " *
                    "rows, so it is 2*MR and not MR that must divide by W)"
            )
        )
        KI === SIMDKernel{2 * MR, NR, real(T), W} ||
            throw(
            ArgumentError(
                "OneMKernel's inner kernel must be SIMDKernel{$(2 * MR),$NR," *
                    "$(real(T)),$W}, got $KI"
            )
        )
        return new{MR, NR, T, W, KI}(descriptor, inner)
    end
end

function OneMKernel(::Val{MR}, ::Val{NR}, ::Type{T}, ::Val{W}) where {MR, NR, T, W}
    T <: Complex ||
        throw(ArgumentError("OneMKernel requires a complex element type, got $T"))
    descriptor = ComplexKernelDescriptor(Val(MR), Val(NR), T, OneEFormat(), PlanarFormat())
    inner = SIMDKernel(Val(2 * MR), Val(NR), real(T), Val(W))
    return OneMKernel{MR, NR, T, W, typeof(inner)}(descriptor, inner)
end
function OneMKernel(::Val{MR}, ::Val{NR}, ::Type{T}) where {MR, NR, T}
    T <: Complex ||
        throw(ArgumentError("OneMKernel requires a complex element type, got $T"))
    return OneMKernel(Val(MR), Val(NR), T, Val(_default_lanewidth(real(T))))
end

complex_method(::OneMKernel) = OneMMethod()

"""
    lanewidth(kernel::OneMKernel) -> Int

The kernel's `SIMD.Vec` lane width `W`, counted in **reals**. Always even.
"""
lanewidth(::OneMKernel{MR, NR, T, W}) where {MR, NR, T, W} = W

"""
    avecs_per_column(kernel::OneMKernel) -> Int

`2*mr(kernel) ÷ lanewidth(kernel)`: A vectors (and accumulator vectors) per
output column. Counted in the inner **real** kernel's rows, so it is `2MR÷W`
and not `MR÷W`; each vector covers `W÷2` complex rows.
"""
avecs_per_column(::OneMKernel{MR, NR, T, W}) where {MR, NR, T, W} = (2 * MR) ÷ W

"""
    onem_register_pressure(kernel::OneMKernel) -> Int

Vector registers live at the bottom of the inner real kernel's K loop:

    MV*NR accumulators + MV A vectors + 1 B broadcast

with `MV = 2*mr(kernel) ÷ lanewidth(kernel)`. This is the **real** kernel's
budget, unchanged -- 1m holds one accumulator plane over a doubled real row
count where planar holds two planes over `MR` rows, so at equal `(MV, NR)` 1m
needs a little over half of planar's registers. **Cliff A**: it must be `<=`
the architectural register count (`target_profile().nregisters`; 32 zmm under
AVX-512).

Every shipped 1m shape is spill-free -- 0 stack stores and 0 reloads at
`(12,8,8)`, `(16,6,8)`, `(8,8,8)` and `(8,4,4)`, at exactly `MV*NR` FMAs per
*real* K step -- as checked on `kernel.inner`, because the code running 1m's K
loop **is** the real path's `accumulate`, byte for byte. Spill counts are not
timings; shapes are ranked on measured throughput.
"""
onem_register_pressure(::OneMKernel{MR, NR, T, W}) where {MR, NR, T, W} =
    ((2 * MR) ÷ W) * NR + ((2 * MR) ÷ W) + 1

# ----------------------------------------------------------------------------
# zero_accumulator, accumulate -- both are the real kernel's, verbatim
# ----------------------------------------------------------------------------

"""
    zero_accumulator(kernel::OneMKernel{MR,NR,T,W}) -> NTuple{NV,SIMD.Vec{W,real(T)}}

The **inner real kernel's** zero accumulator: `NV = (2MR÷W)*NR` zero
`Vec{W,real(T)}` values, entry `(v, j)` at 1-based tuple position
`v + (2MR÷W)*j + 1`, exactly as `zero_accumulator(::SIMDKernel)` defines it.

Interpreted as a real `2MR x NR` tile, real row `2i` holds the real part and
real row `2i+1` the imaginary part of complex row `i`; see
[`store_tile!`](@ref) for how that is read back.
"""
zero_accumulator(kernel::OneMKernel) = zero_accumulator(kernel.inner)

"""
    accumulate(kernel::OneMKernel, acc, packed_a, packed_b, kc::Int) -> acc

The induced method itself: the real microkernel over `2*kc` real K steps.
`kc` is the **logical** (complex) K depth; `kc == 0` returns `acc` unchanged
without reading the packed buffers. `packed_a` must be in
[`OneEFormat`](@ref) and `packed_b` in [`PlanarFormat`](@ref).

There is no arithmetic here and there must never be: the body is
`accumulate(kernel.inner, acc, packed_a, packed_b, 2 * kc)`, the same
`_accumulate_step` the real path runs. Not bitwise identical to a scalar
complex dot product (FMA grouping differs) — compare with a tolerance, never
`==`. No tolerance widening relative to the real path at the same precision is
needed or expected.
"""
function Base.accumulate(
        kernel::OneMKernel{MR, NR, T, W}, acc::NTuple{NV, Vec{W, R}},
        packed_a::PA, packed_b::PB, kc::Int
    ) where {MR, NR, T, W, R, NV, PA, PB}
    # Checked here rather than left to the inner kernel so the message reports
    # the LOGICAL kc the caller passed, not the doubled real one.
    kc >= 0 || throw(ArgumentError("accumulate requires kc >= 0, got kc = $kc"))
    return accumulate(kernel.inner, acc, packed_a, packed_b, 2 * kc)
end

# ----------------------------------------------------------------------------
# store_tile! -- the one 1m-specific piece
# ----------------------------------------------------------------------------

# The `OneM` tile reader. The accumulator is the inner real kernel's, holding a
# real `2MR x NR` tile; complex row `i` is real rows `2i` (re) and `2i+1` (im).
#
# ADJACENCY, which is what the even-`W` constructor check buys: accumulator
# vector `v` covers real rows `v*W .. v*W+W-1`, and with `W` even and `2MR` a
# multiple of `W`, `v*W` is even and `MR == MV*(W÷2)` exactly -- complex rows
# partition cleanly, `W÷2` per vector, none split across a vector boundary. So
# the two halves of complex row `i = v*(W÷2) + u` are lanes `2u+1` and `2u+2`
# (1-based) of vector `v`: always adjacent, in one vector. No cross-vector
# shuffle, no second load.
#
# `@generated` with literal tuple indices (Cliff B), unrolled over `(v, j)`
# exactly as `_store_tile_scattered!` (src/microkernels/simd.jl) and
# `_store_tile_planar!` (src/microkernels/planar.jl). Scattered/scalar path
# only: 1m has no unit-stride vector store counterpart to planar's
# `_store_tile_planar_vector!`.
@generated function _store_tile_onem!(
        destination::QSTile, acc::NTuple{NV, Vec{W, R}},
        alpha::T, beta::T, kernel::OneMKernel{MR, NR, T, W},
        m::Int, n::Int
    ) where {MR, NR, T, W, R, NV}
    R === real(T) ||
        throw(
        ArgumentError(
            "_store_tile_onem!: accumulator lane type $R does not match " *
                "real($T) = $(real(T))"
        )
    )
    iseven(W) ||
        throw(ArgumentError("_store_tile_onem!: the lane-pair reader requires an even W, got $W"))
    MV = (2 * MR) ÷ W
    MV * NR == NV ||
        throw(
        ArgumentError(
            "_store_tile_onem!: accumulator length $NV does not match " *
                "(2*mr÷W)*nr = $(MV * NR) for MR=$MR, NR=$NR, W=$W"
        )
    )
    HW = W ÷ 2  # complex rows per accumulator vector

    blocks = Any[]
    for j in 0:(NR - 1), v in 0:(MV - 1)
        idx = v + MV * j + 1
        push!(
            blocks, quote
                if $j < n
                    vec = acc[$idx]
                    for u in 0:$(HW - 1)
                        i = $(v * HW) + u
                        i < m || break
                        _axpby_tile!(
                            destination, i, $j, alpha,
                            Complex(vec[2 * u + 1], vec[2 * u + 2]), beta
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

"""
    store_tile!(destination::QSTile, acc::NTuple{NV,SIMD.Vec{W,real(T)}}, alpha::T, beta::T, kernel::OneMKernel) -> destination

1m counterpart of `SIMDKernel`'s and `PlanarKernel`'s `store_tile!`, with the
**same contract**: `C[i,j] = alpha*R[i,j] + beta*C[i,j]` over the valid
rectangle only; `alpha == 0` never reads `acc`; `beta == 0` never reads old
`C`; `beta == 1` skips the multiplication; padding lanes (`i >= m`, `j >= n`)
are never read, so nonfinite padding in `acc` cannot propagate; an empty
destination is a no-op in every branch.

The accumulator is the *inner real* kernel's, read as a real `2MR x NR` tile in
which real row `2i` is the real part and real row `2i+1` the imaginary part of
complex row `i`. Because `W` is even and accumulator vectors therefore start at
even real rows, those two halves are always **adjacent lanes of the same
`Vec`**: complex row `i = v*(W÷2) + u` is `Complex(acc[v + MV*j + 1][2u+1],
acc[v + MV*j + 1][2u+2])` with `MV = 2MR÷W`. No shuffle, no second load.

Scattered/scalar path only, delegating to the generic `_axpby_tile!`
(src/microkernels/interface.jl). Not `@inline`, for the reason given at the
FMAddSub `store_tile!` (src/microkernels/fmaddsub.jl): scalar-only store.
"""
function store_tile!(
        destination::QSTile, acc::NTuple{NV, Vec{W, R}},
        alpha::T, beta::T, kernel::OneMKernel{MR, NR, T, W}
    ) where {MR, NR, T, W, R, NV}
    m, n = _store_prologue!(destination, alpha, beta)
    (m == 0 || n == 0) && return destination
    return _store_tile_onem!(destination, acc, alpha, beta, kernel, m, n)
end
