# Contiguous packing fast paths, each for the one sliver shape it can serve: a
# full sliver whose lane axis is unit-stride, dense storage and the driver's
# `identity`/`conj` transform. They read exactly the addresses the scalar loop
# would, so they skip no validation.

using SIMD: shufflevector

# --- Vector-path predicates (the microkernels' vector stores use them too) ---

# Whether an axis steps through storage one element at a time.
is_unit_stride(ax::AffineAxis) = ax.stride == 1
is_unit_stride(::AbstractVector{Int}) = false

# Dense rank-1 storage whose elements load as SIMD lanes and convert lane-wise
# to `T`: `T` itself, or another supported type of the same domain.
const LaneFloat = Union{Float32, Float64}
@inline dense_lanes(storage::S, ::Type{T}) where {S, T} =
    storage isa DenseVector && lane_convertible(eltype(S), T)
lane_convertible(::Type, ::Type) = false
lane_convertible(::Type{<:LaneFloat}, ::Type{<:LaneFloat}) = true
lane_convertible(::Type{Complex{S}}, ::Type{Complex{T}}) where {S <: LaneFloat, T <: LaneFloat} = true

# The deinterleaving complex packer and the complex vector stores (unmeasured
# on NEON).
@inline complex_fastpath_isa_eligible(profile::TargetProfile) = profile.isa in (:avx512, :avx2)
@inline complex_fastpath_isa_eligible() = complex_fastpath_isa_eligible(target_profile())

# --- Real ---

# `conj` is the identity on a real element type.
@inline copies_unchanged(::typeof(identity), ::Type) = true
@inline copies_unchanged(::typeof(conj), ::Type{T}) where {T <: Real} = true
@inline copies_unchanged(::Any, ::Type) = false

# Everything but `lanes == L` and the stride test folds at compile time.
@inline function real_contiguous_eligible(
        tile::Tile, spec::SliverSpec{I, L, F}, transform, lanes::Int
    ) where {I, L, F}
    T = realtype(spec)
    return F === RealFormat && dense_lanes(tile.storage, T) &&
        copies_unchanged(transform, T) && lanes == L && is_unit_stride(tile.rows)
end

# One `Vec{L}` load/convert/store per K step.
@inline function pack_real_contiguous!(
        panel::PackedPanel{T}, tile::Tile{<:DenseVector{S}}, ::SliverSpec{I, L}, k_block_length::Int
    ) where {T, S, I, L}
    storage = tile.storage
    lanebase = @inbounds tile.base + tile.rows[1]
    GC.@preserve storage begin
        sp = pointer(storage)
        dp = panel.ptr
        for p in 1:k_block_length
            v = vload(Vec{L, S}, sp + sizeof(S) * (lanebase + (@inbounds tile.cols[p])))
            vstore(convert(Vec{L, T}, v), dp + sizeof(T) * (L * (p - 1)))
        end
    end
    return panel
end

# --- Complex ---

# The transforms `pack_alt` covers; anything else must take the scalar path.
@inline complex_pack_transform_eligible(::typeof(identity)) = true
@inline complex_pack_transform_eligible(::typeof(conj)) = true
@inline complex_pack_transform_eligible(::Any) = false

# A deinterleave of the source's native `[re, im, ...]` layout plus, for
# `conj`, a sign flip. Unit stride over dense rank-1 storage is what makes
# reading `L` elements as `2L` reals a sound bitcast. `lanes == L` keeps
# padding (and its no-`-0.0` rule) on the scalar path.
@inline function complex_contiguous_eligible(
        tile::Tile, spec::SliverSpec{I, L, F, T}, transform, lanes::Int
    ) where {I, L, F, T}
    return F !== RealFormat && dense_lanes(tile.storage, T) &&
        complex_pack_transform_eligible(transform) &&
        lanes == L && is_unit_stride(tile.rows) &&
        complex_fastpath_isa_eligible()
end

# The shuffles read `re` lanes from `src` and `im` lanes from `alt`, so the
# transform is entirely the choice of `alt`. `-src` is a sign-bit flip,
# bit-identical to the scalar `imag(conj(z))` (a multiply by -1 would not be,
# on NaN payloads).
@inline pack_alt(src::Vec, ::typeof(identity)) = src
@inline pack_alt(src::Vec, ::typeof(conj)) = -src
# 1e's second region stores `-im`, so it wants the opposite choice.
@inline pack_alt_flipped(src::Vec, ::typeof(identity)) = -src
@inline pack_alt_flipped(src::Vec, ::typeof(conj)) = src

# `@generated` because `shufflevector` needs a literal `Val` index tuple, which
# `Val(ntuple(...))` is not reliably. `src` holds `L` elements of one K step.

# Planar, one whole K step: `[re_0 .. re_{L-1} | im_0 .. im_{L-1}]`.
@generated function planar_pack_shuffle(
        src::Vec{N, R}, alt::Vec{N, R}, ::Val{L}
    ) where {N, R, L}
    N == 2 * L || return :(throw(ArgumentError("planar_pack_shuffle: expected N == 2L")))
    idx = ntuple(k -> (k - 1) < L ? 2 * (k - 1) : N + 2 * ((k - 1) - L) + 1, 2 * L)
    return :(shufflevector(src, alt, Val($idx)))
end

# 1e's first region (and interleaved): `[re_0, ±im_0, re_1, ±im_1, ...]`.
@generated function onee_pack_shuffle_a(
        src::Vec{N, R}, alt::Vec{N, R}, ::Val{L}
    ) where {N, R, L}
    N == 2 * L || return :(throw(ArgumentError("onee_pack_shuffle_a: expected N == 2L")))
    idx = ntuple(k -> iseven(k) ? N + (k - 1) : (k - 1), 2 * L)
    return :(shufflevector(src, alt, Val($idx)))
end

# 1e's second region: `[∓im_0, re_0, ∓im_1, re_1, ...]`.
@generated function onee_pack_shuffle_b(
        src::Vec{N, R}, alt::Vec{N, R}, ::Val{L}
    ) where {N, R, L}
    N == 2 * L || return :(throw(ArgumentError("onee_pack_shuffle_b: expected N == 2L")))
    idx = ntuple(k -> isodd(k) ? N + k : k - 2, 2 * L)
    return :(shufflevector(src, alt, Val($idx)))
end

# One K step's `2L` reals `src`, stored at `dp`.
@inline store_step!(dp::Ptr, src::Vec, ::SliverSpec{I, L, PlanarFormat}, transform) where {I, L} =
    vstore(planar_pack_shuffle(src, pack_alt(src, transform), Val(L)), dp)
@inline store_step!(dp::Ptr, src::Vec, ::SliverSpec{I, L, InterleavedFormat}, transform) where {I, L} =
    vstore(onee_pack_shuffle_a(src, pack_alt(src, transform), Val(L)), dp)
@inline function store_step!(dp::Ptr{R}, src::Vec, ::SliverSpec{I, L, OneEFormat}, transform) where {R, I, L}
    vstore(onee_pack_shuffle_a(src, pack_alt(src, transform), Val(L)), dp)
    vstore(onee_pack_shuffle_b(src, pack_alt_flipped(src, transform), Val(L)), dp + sizeof(R) * 2L)
    return nothing
end

# Lane `t` of K step `p` is element `tile.base + tile.rows[1] + tile.cols[p] +
# (t - 1)`; only the lane axis must be unit-stride, the K steps may be
# scattered. Pinning `Complex{RS}` in the signature keeps the bitcast to `RS`
# lanes sound; the lanes convert to `R` before the shuffle.
@inline function pack_complex_contiguous!(
        panel::PackedPanel{R}, tile::Tile{<:DenseVector{Complex{RS}}}, spec::SliverSpec{I, L},
        k_block_length::Int, transform::F
    ) where {R, RS, I, L, F}
    storage = tile.storage
    elembase = @inbounds tile.base + tile.rows[1]
    GC.@preserve storage begin
        sp = reinterpret(Ptr{RS}, pointer(storage))
        for p in 1:k_block_length
            src = convert(
                Vec{2 * L, R},
                vload(Vec{2 * L, RS}, sp + sizeof(RS) * (2 * (elembase + (@inbounds tile.cols[p]))))
            )
            store_step!(panel.ptr + sizeof(R) * panel_offset(spec, 1, p), src, spec, transform)
        end
    end
    return panel
end
