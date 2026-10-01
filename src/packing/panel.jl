# GUARDRAIL: packers and kernels read and write packed micro-panels only as
# `PackedPanel`s, borrowed pointers, never `view`s: a `SubArray`'s address
# arithmetic evicts the accumulator at large register tiles and heap-allocates
# per `execute!`. Only the checked entry points also take a `DenseVector`.

using SIMD: Vec, vload, vstore

# `len` borrowed elements at `ptr`; the caller must `GC.@preserve` the owner.
# `len` exists only for capacity checks.
struct PackedPanel{T}
    ptr::Ptr{T}
    len::Int
end

Base.length(panel::PackedPanel) = panel.len
Base.eltype(::Type{PackedPanel{T}}) where {T} = T

@inline packed_panel(buffer::AbstractVector{T}, first1::Int, len::Int) where {T} =
    PackedPanel{T}(pointer(buffer, first1), len)

# Zero-based panel element access.
@inline panel_vload(::Type{Vec{W, T}}, p::PackedPanel{T}, o::Int) where {W, T} =
    vload(Vec{W, T}, p.ptr + sizeof(T) * o)
@inline panel_load(p::PackedPanel{T}, o::Int) where {T} = unsafe_load(p.ptr + sizeof(T) * o)
@inline panel_store!(p::PackedPanel{T}, o::Int, x::T) where {T} =
    unsafe_store!(p.ptr + sizeof(T) * o, x)

# --- Line-by-line packing of a split macro block ---

# Pack a macro block of `fcount` free coordinates (offsets `fbuf`, enumerated by
# a split group, whole `E x L` groups) by `k_block_length` K steps into the
# panels the per-sliver packer would write, reading each group line by line:
# coordinate `i0 + E * y2` is element `y2` of the line at `i0`. `@noinline`: one
# call per block, and inlined into the nest it slows the nest's micro-kernel
# loop.
@noinline function _pack_block_transposed!(
        plane_offset::PO, format::FMT, buffer::PK, descriptor::DS, ::Val{R}, Rp::Int,
        storage::ST, base::Int, fbuf::Vector{Int}, kaxis::KA, transform::F,
        fcount::Int, k_block_length::Int, split
    ) where {PO, FMT <: PackFormat, PK, DS, R, ST, KA <: AbstractVector{Int}, F}
    E, L = Int(split.E), Int(split.L)
    G = E * L
    # The sliver and lane come from a per-element divrem by the constant `R`;
    # hoisting them per line lets LLVM rewrite the loop so the line misses no
    # longer overlap.
    emit(kbase, i, p) = @inbounds _pack_line_element!(
        plane_offset, format, buffer, descriptor, Val(R), Rp * k_block_length, transform(storage[kbase + fbuf[i + 1]]), i, p
    )
    if split.kinner
        for g0 in 0:G:(fcount - 1), y1 in 0:(E - 1), p in 1:k_block_length
            kbase = @inbounds base + kaxis[p] + 1
            for y2 in 0:(L - 1)
                emit(kbase, g0 + y1 + E * y2, p)
            end
        end
    else
        for g0 in 0:G:(fcount - 1), p in 1:k_block_length
            kbase = @inbounds base + kaxis[p] + 1
            for y1 in 0:(E - 1), y2 in 0:(L - 1)
                emit(kbase, g0 + y1 + E * y2, p)
            end
        end
    end
    valid = fcount % R
    if valid != 0
        rb = (fcount ÷ R) * Rp * k_block_length
        pad = (plane, idx, pp) -> rb + plane_offset(descriptor, idx, pp, plane)
        for p in 1:k_block_length, t in (valid + 1):R
            _pack_emit_zero!(buffer, format, pad, t, p, realtype(descriptor))
        end
    end
    return nothing
end

# Element `x` of zero-based block coordinate `i` (sliver `i ÷ R`, lane
# `i % R + 1`) at K step `p`.
@inline function _pack_line_element!(
        plane_offset::PO, format::FMT, buffer::PK, descriptor::DS, ::Val{R}, panel::Int, x, i::Int, p::Int
    ) where {PO, FMT, PK, DS, R}
    r, t = divrem(i, R)
    po = (plane, idx, pp) -> r * panel + plane_offset(descriptor, idx, pp, plane)
    _pack_emit!(buffer, format, po, t + 1, p, convert(_element_type(format, scalartype(descriptor)), x))
    return nothing
end
