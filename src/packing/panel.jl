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
