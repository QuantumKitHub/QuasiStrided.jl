# Explicit-SIMD microkernel (SIMD.jl's Vec{N,T}), matching ScalarKernel's
# packed-format contract and API. Accumulator is an immutable
# tuple of Vec{W,T} (not a heap array) for register residency; the K-step
# body is a @generated, closure-free function so `acc = _accumulate_step(...)`
# is a plain reassignment that never boxes.

using SIMD: Vec, vload, vstore

"""
    SIMDKernel{MR,NR,T,W}(descriptor::KernelDescriptor{MR,NR,T})
    SIMDKernel(::Val{MR}, ::Val{NR}, ::Type{T}, ::Val{W})
    SIMDKernel(::Val{MR}, ::Val{NR}, ::Type{T})

Explicit-SIMD microkernel using `SIMD.Vec{W,T}` lanes. Wraps a
`KernelDescriptor` and adds vector width `W`; `MR` must be a multiple of `W`
(checked at construction). `NR` need not be — B contributes one scalar per
column per K step. The 3-argument form defaults `W` via
[`_default_lanewidth`](@ref).
"""
struct SIMDKernel{MR, NR, T, W} <: DescriptorKernel{MR, NR, T}
    descriptor::KernelDescriptor{MR, NR, T}

    function SIMDKernel{MR, NR, T, W}(descriptor::KernelDescriptor{MR, NR, T}) where {MR, NR, T, W}
        _check_lanewidth("SIMDKernel", W)
        _check_mr_multiple("SIMDKernel", MR, W)
        return new{MR, NR, T, W}(descriptor)
    end
end

SIMDKernel(::Val{MR}, ::Val{NR}, ::Type{T}, ::Val{W}) where {MR, NR, T, W} =
    SIMDKernel{MR, NR, T, W}(KernelDescriptor(Val(MR), Val(NR), T))
SIMDKernel(::Val{MR}, ::Val{NR}, ::Type{T}) where {MR, NR, T} =
    SIMDKernel(Val(MR), Val(NR), T, Val(_default_lanewidth(T)))

"""
    lanewidth(kernel::SIMDKernel) -> Int

The kernel's `SIMD.Vec` lane width `W`.
"""
lanewidth(::SIMDKernel{MR, NR, T, W}) where {MR, NR, T, W} = W

"""
    avecs_per_column(kernel::SIMDKernel) -> Int

`mr(kernel) ÷ lanewidth(kernel)`: the number of full-width A vectors (and
thus accumulator vectors) per output column.
"""
avecs_per_column(::SIMDKernel{MR, NR, T, W}) where {MR, NR, T, W} = MR ÷ W

# ----------------------------------------------------------------------------
# zero_accumulator, accumulate
# ----------------------------------------------------------------------------

"""
    zero_accumulator(kernel::SIMDKernel{MR,NR,T,W}) -> NTuple{NV,SIMD.Vec{W,T}}

Return a logical `MR`-by-`NR` zero accumulator, represented as an immutable
tuple of `NV = (MR÷W)*NR` zero `Vec{W,T}` values (an ordinary heap array of
accumulators would not stay in registers). Entry
`(v, j)` (`v` the row-vector index in `0:MR÷W-1`, `j` the output column in
`0:NR-1`) lives at 1-based tuple position `v + (MR÷W)*j + 1`; physical row
`i` of that vector is lane `i - v*W + 1` (1-based `SIMD.Vec` indexing).
"""
function zero_accumulator(kernel::SIMDKernel{MR, NR, T, W}) where {MR, NR, T, W}
    z = zero(Vec{W, T})
    return ntuple(_ -> z, Val((MR ÷ W) * NR))
end

# Address of A's logical row `i` at K step `p`, as `_accumulate_step` hands it
# to `panel_vload`. The default -- every `packed_a` that is an actual packed
# panel (`PackedPanel`, or a plain `AbstractVector` in packed layout) -- is
# exactly the kernel's packed formula `packed_a_offset(kernel, i, p)`, so the
# packed paths (`execute!`, `execute_tilewise!`, 1m's inner kernel) compile to
# the same code as before this indirection existed. The one other method is
# `UnpackedAView`'s (src/execution/halfpack.jl), for an A read in place from
# its own storage. Dispatch is on `packed_a`'s TYPE, which `_accumulate_step`
# is specialized on anyway, so the choice is made at compile time: there is
# no runtime branch on any path.
@inline _a_step_offset(::PA, kernel, i::Int, p::Int) where {PA} = packed_a_offset(kernel, i, p)

# Fully unrolled, closure-free K-step body: one vector load per A row-vector,
# one scalar load per B column, NVECA*NR FMAs, generated as straight-line code.
@generated function _accumulate_step(
        kernel::SIMDKernel{MR, NR, T, W}, acc::NTuple{NV, Vec{W, T}},
        packed_a::PA, packed_b::PB, p::Int
    ) where {MR, NR, T, W, NV, PA, PB}
    NVECA = MR ÷ W
    NVECA * NR == NV ||
        throw(
        ArgumentError(
            "_accumulate_step: accumulator length $NV does not match " *
                "(mr÷W)*nr = $(NVECA * NR) for MR=$MR, NR=$NR, W=$W"
        )
    )

    avars = [Symbol(:a, v) for v in 0:(NVECA - 1)]
    bvars = [Symbol(:b, j) for j in 0:(NR - 1)]

    load_a = [
        :($(avars[v + 1]) = panel_vload(Vec{$W, $T}, packed_a, _a_step_offset(packed_a, kernel, $(v * W), p)))
            for v in 0:(NVECA - 1)
    ]
    load_b = [
        :($(bvars[j + 1]) = panel_load(packed_b, packed_b_offset(kernel, $j, p)))
            for j in 0:(NR - 1)
    ]

    acc_exprs = Vector{Any}(undef, NV)
    for j in 0:(NR - 1), v in 0:(NVECA - 1)
        idx = v + NVECA * j + 1
        acc_exprs[idx] = :(muladd($(avars[v + 1]), $(bvars[j + 1]), acc[$idx]))
    end

    return quote
        Base.@_inline_meta
        @inbounds begin
            $(load_a...)
            $(load_b...)
            return $(Expr(:tuple, acc_exprs...))
        end
    end
end

"""
    accumulate(kernel::SIMDKernel, acc, packed_a, packed_b, kc::Int) -> acc

SIMD counterpart of `ScalarKernel`'s `accumulate`; same contract (`kc == 0`
is a no-op), but **not bitwise identical** (FMA grouping/order differ —
compare with a tolerance, never `==`). `packed_a`/`packed_b` may be a
[`PackedPanel`](@ref) — what the driver passes, and the only form that keeps
a large accumulator register-resident — or any contiguous `AbstractVector{T}`
such as a `Vector` or unit-range `view`. `packed_a` may also be an
[`UnpackedAView`](@ref) (A read in place by `execute_half_packed!`), which
`_a_step_offset` addresses by its own formula instead of the packed one.
"""
@inline function Base.accumulate(
        kernel::SIMDKernel{MR, NR, T, W}, acc::NTuple{NV, Vec{W, T}},
        packed_a::PA, packed_b::PB, kc::Int
    ) where {MR, NR, T, W, NV, PA, PB}
    kc == 0 && return acc
    kc > 0 || _throw_negative_kc(:accumulate, kc)
    @inbounds for p in 0:(kc - 1)
        acc = _accumulate_step(kernel, acc, packed_a, packed_b, p)
    end
    return acc
end

"""
    _vector_store_eligible(tile::QSTile, ::Type{T}) -> Bool

Whether `tile` can take [`store_tile!`](@ref)'s vectorized path: unit-stride
`AffineAxis` rows into *any concrete rank-1 dense* storage of `T`
(`Vector{T}`, `Memory{T}`, ...), which is exactly the set `SIMD.jl`'s
`vload`/`vstore` array methods are defined on
(`FastContiguousArray{T,1} ⊇ DenseVector{T}`). The storage half is a
compile-time constant (it only inspects `typeof(tile.storage)`), so the whole
predicate folds to `_unit_stride_rows` or to `false` at each specialization.

Rank-2 storage (`Matrix`) and non-`DenseArray` storage (`SubArray`, even a
contiguous one) are excluded and keep taking the scalar fallback, as do
non-unit-stride affine rows and scattered rows. The check must admit
`Memory{T}`, not just `Vector{T}`: on Julia >= 1.11 the `parent` of an
`Array`-backed `StridedView`, and hence the driver's storage, is `Memory{T}`.
"""
@inline _vector_store_eligible(tile::QSTile, ::Type{T}) where {T} =
    _unit_stride_rows(tile.rows) && tile.storage isa DenseVector{T}

# Scalar/scattered store path.
#
# GUARDRAIL (Cliff B): every `acc[...]` here must be a *literal* tuple index,
# which is why this is `@generated` and unrolled over `(v, j)` rather than a
# plain `for j, i` loop. Indexing an `NTuple` dynamically forces the whole
# tuple to memory, and above NV = 16 the compiler heap-allocates it on every
# call (e.g. at (MR,NR,W) = (32,6,8), but not at (16,6,8)). Scattered
# destinations are this engine's reason to exist, so a dynamic index here
# silently caps the usable register tile on exactly the workload that matters.
# Only the lane index inside a single `Vec` may be a runtime value.
@generated function _store_tile_scattered!(
        destination::QSTile, acc::NTuple{NV, Vec{W, T}},
        alpha::T, beta::T, kernel::SIMDKernel{MR, NR, T, W},
        m::Int, n::Int
    ) where {MR, NR, T, W, NV}
    NVECA = MR ÷ W
    blocks = Any[]
    for j in 0:(NR - 1), v in 0:(NVECA - 1)
        idx = v + NVECA * j + 1
        push!(
            blocks, quote
                if $j < n
                    vec = acc[$idx]
                    for lane in 1:$W
                        i = $(v * W) + lane - 1
                        i < m || break
                        _axpby_tile!(destination, i, $j, alpha, vec[lane], beta)
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

# Vectorized store path: unit-stride `AffineAxis` rows into rank-1 dense
# storage, i.e. exactly the tiles `_vector_store_eligible` admits (the caller
# checks; `rows::AffineAxis` is pinned in the signature so a violation is a
# MethodError rather than a wrong answer).
#
# GUARDRAIL (Cliff B): same rule as `_store_tile_scattered!` above, and the
# reason this is `@generated` too. Both the whole-block stores and the lane
# tail must index `acc` with a *literal* tuple position, so the unrolling over
# `(v, j)` happens here at compile time rather than in a runtime loop; only the
# lane index inside a single `Vec` may be a runtime value. A runtime loop over
# `v`/`j` indexing `acc[v + NVECA * j + 1]` is precisely the dynamic-index
# pattern that heap-allocates the accumulator above NV = 16, and this is the
# path the driver takes for every unit-stride destination.
#
# `m`/`n` stay runtime values, compared against literal row/column positions:
# nothing outside the valid rectangle is loaded or stored, so a partial tile
# never over-reads a padding lane or a neighbouring tile's element.
@generated function _store_tile_vector!(
        destination::QSTile{S, <:AffineAxis}, acc::NTuple{NV, Vec{W, T}},
        alpha::T, beta::T, kernel::SIMDKernel{MR, NR, T, W},
        m::Int, n::Int
    ) where {S, MR, NR, T, W, NV}
    NVECA = MR ÷ W
    blocks = Any[]
    for j in 0:(NR - 1)
        vblocks = Any[]
        for v in 0:(NVECA - 1)
            idx = v + NVECA * j + 1
            push!(
                vblocks, quote
                    vec = acc[$idx]
                    if $((v + 1) * W) <= m
                        # Rows v*W .. v*W+W-1 are all inside [0, m): one
                        # W-wide load/store at that one-based storage index.
                        at = colbase + $(v * W) + 1
                        vstore(
                            iszero(beta) ? alpha * vec :
                                isone(beta) ? muladd(alpha, vec, vload(Vec{$W, $T}, storage, at)) :
                                muladd(alpha, vec, beta * vload(Vec{$W, $T}, storage, at)),
                            storage, at
                        )
                    elseif $(v * W) < m
                        # Row tail: this vector straddles m, so store the
                        # valid lanes one at a time (runtime lane index into a
                        # literally indexed `Vec` is allowed).
                        for lane in 1:$W
                            i = $(v * W) + lane - 1
                            i < m || break
                            _axpby_at!(storage, colbase + i + 1, alpha, vec[lane], beta)
                        end
                    end
                end
            )
        end
        push!(
            blocks, quote
                if $j < n
                    colbase = rowbase0 + axis_offset(cols, $j)  # zero-based address of (i=0, j)
                    $(vblocks...)
                end
            end
        )
    end
    return quote
        Base.@_inline_meta
        storage = destination.storage
        cols = destination.cols
        # zero-based address at (i=0, j=0)'s row contribution; rows are
        # unit-stride, so row `i` is `rowbase0 + i`.
        rowbase0 = destination.base + destination.rows.base
        @inbounds begin
            $(blocks...)
        end
        return destination
    end
end

"""
    store_tile!(destination::QSTile, acc::NTuple{NV,SIMD.Vec{W,T}}, alpha::T, beta::T, kernel::SIMDKernel) -> destination

SIMD counterpart of `ScalarKernel`'s `store_tile!`; same contract and
alpha/beta shortcuts. Fast path: unit-stride `AffineAxis` rows into any 1-D
dense storage (`Vector`, `Memory`, ...) get vector load/store for whole
`W`-row blocks (scalar tail for the remainder, never over-reading past the
valid rectangle); see [`_vector_store_eligible`](@ref) for exactly which
storage qualifies. Otherwise falls back to the scalar path, one lane at a
time. Empty destination is a no-op.

`@inline`, and so is the vectorized store body: left to the inliner's cost
model this call was made out of line, which forces the whole accumulator (12
zmm at `(16,6,8)`) through the stack -- a spill after the K loop and a reload
inside the store -- on every micro-tile. Inlined, the accumulator stays in
registers from the last FMA to the `vstore`. Measured 2026-09-26 (ccqlin038,
Cascade Lake, together with moving the prologue's throws out of line): the
fixed per-tile cost of a 16x6 Float64 tile in `execute!` fell from ~37 to
~26 ns, and a 128^3 matmul's tile phase from 56 to 47 us. The scattered store
stays out of line: it is scalar per element (~160 ns per full tile), so the
spill it forces is noise, while inlining it grows the tile function by ~4000
instructions.
"""
@inline function store_tile!(
        destination::QSTile, acc::NTuple{NV, Vec{W, T}},
        alpha::T, beta::T, kernel::SIMDKernel{MR, NR, T, W}
    ) where {MR, NR, T, W, NV}
    m, n = _store_prologue!(destination, alpha, beta)
    (m == 0 || n == 0) && return destination

    if _vector_store_eligible(destination, T)
        return _store_tile_vector!(destination, acc, alpha, beta, kernel, m, n)
    end

    return _store_tile_scattered!(destination, acc, alpha, beta, kernel, m, n)
end
