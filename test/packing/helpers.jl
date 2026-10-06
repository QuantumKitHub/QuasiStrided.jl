# Reference packed layouts and fixtures shared by the packing test files.

include("../helpers.jl")

using QuasiStrided: RealFormat, PlanarFormat, OneEFormat, InterleavedFormat, PackedPanel,
    packed_panel, packed_length, copies_unchanged, real_contiguous_eligible

# Packed layouts written from the format definitions, not from the offset
# helpers. `g(t, p)` is the source element at lane `t`, K step `p`; lanes
# `>= valid` are padding and stay literal `+0.0`.
_ref_emit!(o, ::RealFormat, vr, p, t, z) = (o[vr * p + t + 1] = z)
function _ref_emit!(o, ::PlanarFormat, vr, p, t, z)
    o[2vr * p + t + 1] = real(z)
    return o[2vr * p + vr + t + 1] = imag(z)
end
function _ref_emit!(o, ::InterleavedFormat, vr, p, t, z)
    o[2vr * p + 2t + 1] = real(z)
    return o[2vr * p + 2t + 2] = imag(z)
end
function _ref_emit!(o, ::OneEFormat, vr, p, t, z)   # [[re, -im], [im, re]]
    b = 4vr * p
    o[b + 2t + 1], o[b + 2t + 2] = real(z), imag(z)
    return o[b + 2vr + 2t + 1], o[b + 2vr + 2t + 2] = -imag(z), real(z)
end
_ref_rpe(::RealFormat) = 1
_ref_rpe(::Union{PlanarFormat, InterleavedFormat}) = 2
_ref_rpe(::OneEFormat) = 4
function ref_pack(fmt, ::Type{T}, vr, k_block_length, valid, g, f) where {T}
    out = zeros(real(T), _ref_rpe(fmt) * vr * k_block_length)
    for p in 0:(k_block_length - 1), t in 0:(valid - 1)
        _ref_emit!(out, fmt, vr, p, t, convert(T, f(g(t, p))))
    end
    return out
end

resized(ax::AffineAxis, n) = AffineAxis(ax.base, ax.stride, n)
resized(ax::SubArray, n) = view(parent(ax), 1:n)

# A sliver source with lanes along `lane` and K steps along `step` (B's tile
# transposed), and its direct-indexing reader.
function pack_fixture(storage, base, lane, step)
    return Tile(storage, base, lane, step), (t, p) -> storage[base + lane[t + 1] + step[p + 1] + 1]
end

# Packs `src` into a destination of kind `dst` (canaried unless a bare Vector)
# and returns the packed prefix and whether the canaries survived.
function pack_into(dst, R, len, src, spec, f)
    buf = fill(R(-777), len + 12)
    if dst === :vector
        v = buf[1:len]
        @test pack!(v, src, spec, f) === v
        return v, true
    end
    GC.@preserve buf begin
        d = packed_panel(buf, 5, len)
        @test pack!(d, src, spec, f) === d
    end
    return buf[5:(4 + len)], all(==(R(-777)), buf[1:4]) && all(==(R(-777)), buf[(5 + len):end])
end

const SCATTER_LANES = [5, 30, 1, 17, 9, 44, 2, 23, 11, 38, 7, 60, 14, 51, 3, 29]
const SCATTER_STEPS = [7, 900, 300, 1500, 60, 1210, 420]

# Exactly representable in Float32, with signed and unsigned zeros sprinkled in:
# `conj` and 1e's `-im` are sign flips.
function complex_value(::Type{T}, i) where {T}
    R = real(T)
    i % 17 == 0 && return T(R(0), R(0))
    i % 19 == 0 && return T(R(3), R(0))
    i % 23 == 0 && return T(R(0), R(-5))
    return T(R(10i + 1), R(-(10i + 2)))
end
complex_storage(::Type{T}, n) where {T} = [complex_value(T, i) for i in 1:n]
