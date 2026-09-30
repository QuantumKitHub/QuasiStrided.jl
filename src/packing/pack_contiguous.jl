# Contiguous packing fast paths, each for the one sliver shape it can serve: a
# full sliver whose lane axis is unit-stride, dense storage, a `PackedPanel`
# destination and the driver's `identity`/`conj` transform. They read exactly
# the addresses the scalar loop would, so they skip no validation.

using SIMD: shufflevector

# `conj` is the identity on a real element type.
@inline _copies_unchanged(::typeof(identity), ::Type) = true
@inline _copies_unchanged(::typeof(conj), ::Type{T}) where {T <: Real} = true
@inline _copies_unchanged(::Any, ::Type) = false

# Dense rank-1 storage whose elements load as SIMD lanes and convert lane-wise
# to `T`: `T` itself, or another supported type of the same domain.
const _LaneFloat = Union{Float32, Float64}
@inline _dense_lanes(storage::S, ::Type{T}) where {S, T} =
    storage isa DenseVector && _lane_convertible(eltype(S), T)
_lane_convertible(::Type, ::Type) = false
_lane_convertible(::Type{<:_LaneFloat}, ::Type{<:_LaneFloat}) = true
_lane_convertible(::Type{Complex{S}}, ::Type{Complex{T}}) where {S <: _LaneFloat, T <: _LaneFloat} = true

# Everything but `m == MR` and the stride test folds at compile time.
# `_unit_stride_rows` deliberately has no fallback method: an unknown axis type
# must be a MethodError.
@inline function _pack_a_contiguous_eligible(
        packed::V, source::QSTile, transform::F, m::Int, ::Val{MR}, ::Type{T}
    ) where {V, F, MR, T}
    return packed isa PackedPanel{T} && _dense_lanes(source.storage, T) &&
        _copies_unchanged(transform, T) && m == MR && _unit_stride_rows(source.rows)
end

# Real A: one `Vec{MR}` load/convert/store per K step.
@inline function _pack_a_contiguous!(
        packed::PackedPanel{T}, storage::DenseVector{S}, rowbase::Int, cols::C,
        ::Val{MR}, k_block_length::Int
    ) where {T, S, C, MR}
    GC.@preserve storage begin
        sp = pointer(storage)
        dp = packed.ptr
        for p in 0:(k_block_length - 1)
            v = vload(Vec{MR, S}, sp + sizeof(S) * (rowbase + axis_offset(cols, p)))
            vstore(convert(Vec{MR, T}, v), dp + sizeof(T) * (MR * p))
        end
    end
    return packed
end

# The transforms `_pack_alt` covers; anything else must take the scalar path.
@inline _complex_pack_transform_eligible(::typeof(identity)) = true
@inline _complex_pack_transform_eligible(::typeof(conj)) = true
@inline _complex_pack_transform_eligible(::Any) = false

# Complex: a deinterleave of the source's native `[re, im, ...]` layout plus,
# for `conj`, a sign flip. `lane_axis` is the axis the packed index runs along
# (`source.rows` for A, `source.cols` for B); unit stride over dense rank-1
# storage is what makes reading `PD` elements as `2PD` reals a sound bitcast.
# `valid == PD` keeps padding (and its no-`-0.0` rule) on the scalar path.
@inline function _pack_complex_contiguous_eligible(
        packed::V, storage::S, lane_axis::AX, transform::F, format::FMT,
        valid::Int, ::Val{PD}, ::Type{T}
    ) where {V, S, AX, F, FMT, PD, T}
    return !(format isa RealFormat) &&
        packed isa PackedPanel{real(T)} && _dense_lanes(storage, T) &&
        _complex_pack_transform_eligible(transform) &&
        valid == PD && _unit_stride_rows(lane_axis) &&
        complex_fastpath_isa_eligible()
end

# The shuffles read `re` lanes from `src` and `im` lanes from `alt`, so the
# transform is entirely the choice of `alt`. `-src` is a sign-bit flip,
# bit-identical to the scalar `imag(conj(z))` (a multiply by -1 would not be,
# on NaN payloads).
@inline _pack_alt(src::Vec, ::typeof(identity)) = src
@inline _pack_alt(src::Vec, ::typeof(conj)) = -src
# 1e's second region stores `-im`, so it wants the opposite choice.
@inline _pack_alt_flipped(src::Vec, ::typeof(identity)) = -src
@inline _pack_alt_flipped(src::Vec, ::typeof(conj)) = src

# `@generated` because `shufflevector` needs a literal `Val` index tuple, which
# `Val(ntuple(...))` is not reliably. `src` holds `PD` elements of one K step.

# Planar, one whole K step: `[re_0 .. re_{PD-1} | im_0 .. im_{PD-1}]`.
@generated function _planar_pack_shuffle(
        src::Vec{N, R}, alt::Vec{N, R}, ::Val{PD}
    ) where {N, R, PD}
    N == 2 * PD || return :(throw(ArgumentError("_planar_pack_shuffle: expected N == 2PD")))
    idx = ntuple(k -> (k - 1) < PD ? 2 * (k - 1) : N + 2 * ((k - 1) - PD) + 1, 2 * PD)
    return :(shufflevector(src, alt, Val($idx)))
end

# 1e's first region (and interleaved): `[re_0, ±im_0, re_1, ±im_1, ...]`.
@generated function _onee_pack_shuffle_a(
        src::Vec{N, R}, alt::Vec{N, R}, ::Val{PD}
    ) where {N, R, PD}
    N == 2 * PD || return :(throw(ArgumentError("_onee_pack_shuffle_a: expected N == 2PD")))
    idx = ntuple(k -> iseven(k) ? N + (k - 1) : (k - 1), 2 * PD)
    return :(shufflevector(src, alt, Val($idx)))
end

# 1e's second region: `[∓im_0, re_0, ∓im_1, re_1, ...]`.
@generated function _onee_pack_shuffle_b(
        src::Vec{N, R}, alt::Vec{N, R}, ::Val{PD}
    ) where {N, R, PD}
    N == 2 * PD || return :(throw(ArgumentError("_onee_pack_shuffle_b: expected N == 2PD")))
    idx = ntuple(k -> isodd(k) ? N + k : k - 2, 2 * PD)
    return :(shufflevector(src, alt, Val($idx)))
end

_single_region_shuffle(::PlanarFormat) = _planar_pack_shuffle
_single_region_shuffle(::InterleavedFormat) = _onee_pack_shuffle_a

# Lane `t` of K step `p` is element `elembase + axis_offset(steps, p) + t`;
# only the lane axis must be unit-stride, `steps` may be scattered. Pinning
# `Complex{RS}` in the signature keeps the bitcast to `RS` lanes sound; the
# lanes convert to `R` before the shuffle.
@inline function _pack_complex_contiguous!(
        format::Union{PlanarFormat, InterleavedFormat}, packed::PackedPanel{R},
        storage::DenseVector{Complex{RS}}, elembase::Int, steps::C, ::Val{PD}, k_block_length::Int,
        transform::F
    ) where {R, RS, C, PD, F}
    shuffle = _single_region_shuffle(format)
    GC.@preserve storage begin
        sp = reinterpret(Ptr{RS}, pointer(storage))
        dp = packed.ptr
        for p in 0:(k_block_length - 1)
            src = convert(
                Vec{2 * PD, R},
                vload(Vec{2 * PD, RS}, sp + sizeof(RS) * (2 * (elembase + axis_offset(steps, p))))
            )
            vstore(
                shuffle(src, _pack_alt(src, transform), Val(PD)),
                dp + sizeof(R) * (2 * PD * p)
            )
        end
    end
    return packed
end

@inline function _pack_complex_contiguous!(
        ::OneEFormat, packed::PackedPanel{R}, storage::DenseVector{Complex{RS}},
        elembase::Int, steps::C, ::Val{PD}, k_block_length::Int, transform::F
    ) where {R, RS, C, PD, F}
    GC.@preserve storage begin
        sp = reinterpret(Ptr{RS}, pointer(storage))
        dp = packed.ptr
        for p in 0:(k_block_length - 1)
            src = convert(
                Vec{2 * PD, R},
                vload(Vec{2 * PD, RS}, sp + sizeof(RS) * (2 * (elembase + axis_offset(steps, p))))
            )
            at = 4 * PD * p
            vstore(
                _onee_pack_shuffle_a(src, _pack_alt(src, transform), Val(PD)),
                dp + sizeof(R) * at
            )
            vstore(
                _onee_pack_shuffle_b(src, _pack_alt_flipped(src, transform), Val(PD)),
                dp + sizeof(R) * (at + 2 * PD)
            )
        end
    end
    return packed
end
