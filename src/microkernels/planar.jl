# Planar (split-complex, BLIS "1r") microkernel, the default complex method.
# Both panels are `PlanarFormat` (`[re_0..re_{n-1} | im_0..im_{n-1}]` per K
# step), so the data is already in the right lanes: four real FMAs per
# (A-vector, B-scalar) pair, no shuffles. Everything works in `real(T)`.

using SIMD: Vec, vload, vstore, shufflevector

"""
    PlanarKernel(::Val{MR}, ::Val{NR}, ::Type{T}[, ::Val{W}])

Split-complex microkernel for `T = ComplexF32/ComplexF64` over
`SIMD.Vec{W,real(T)}` lanes. `MR`/`NR` are complex extents; `MR` must be a
multiple of `W`, which counts reals.
"""
struct PlanarKernel{MR, NR, T, W} <: Microkernel{MR, NR, T}
    descriptor::Descriptor{MR, NR, T, PlanarFormat, PlanarFormat}

    function PlanarKernel{MR, NR, T, W}(
            descriptor::Descriptor{MR, NR, T, PlanarFormat, PlanarFormat}
        ) where {MR, NR, T, W}
        check_vector_shape("PlanarKernel", MR, W)
        return new{MR, NR, T, W}(descriptor)
    end
end

function PlanarKernel(::Val{MR}, ::Val{NR}, ::Type{T}, ::Val{W}) where {MR, NR, T, W}
    return PlanarKernel{MR, NR, T, W}(
        Descriptor(Val(MR), Val(NR), T, PlanarFormat(), PlanarFormat())
    )
end
function PlanarKernel(::Val{MR}, ::Val{NR}, ::Type{T}) where {MR, NR, T}
    T <: Complex ||
        throw(ArgumentError("PlanarKernel requires a complex element type, got $T"))
    return PlanarKernel(Val(MR), Val(NR), T, Val(default_lanewidth(real(T))))
end

complex_method(::PlanarKernel) = PlanarMethod()
lanewidth(::PlanarKernel{MR, NR, T, W}) where {MR, NR, T, W} = W

# One flat tuple: the real plane at `1:NV`, the imaginary plane at `NV+1:2NV`,
# each laid out as the real kernel's. Flat keeps the accumulator in registers.
function zero_accumulator(kernel::PlanarKernel{MR, NR, T, W}) where {MR, NR, T, W}
    z = zero(Vec{W, real(T)})
    return ntuple(_ -> z, Val(2 * (MR ÷ W) * NR))
end

# B column `j` at K step `p` as `(re, im)`; `UnpackedBView` overrides it.
# Shared with the fmaddsub kernel.
@inline _b_step_load2(packed_b::PB, kernel, j::Int, p::Int) where {PB} = (
    panel_load(packed_b, packed_b_offset(kernel, j, p)),
    panel_load(packed_b, packed_b_offset(kernel, j, p, 1)),
)

# GUARDRAIL: the real part is `muladd(-ai, bi, muladd(ar, br, c))`. `c - ai*bi`
# does NOT fuse (Julia sets no LLVM `contract` flag): two instructions, and
# different rounding. `-ai` is hoisted out of the `j` loop and folds into
# `vfnmadd`, so the negation is free.
@generated function _accumulate_step_planar(
        kernel::PlanarKernel{MR, NR, T, W}, acc::NTuple{NA, Vec{W, R}},
        packed_a::PA, packed_b::PB, p::Int
    ) where {MR, NR, T, W, R, NA, PA, PB}
    MV = MR ÷ W
    NV = MV * NR
    check_acc(:_accumulate_step_planar, R, T, NA, 2NV)

    arv = [Symbol(:ar, v) for v in 0:(MV - 1)]
    aiv = [Symbol(:ai, v) for v in 0:(MV - 1)]
    naiv = [Symbol(:nai, v) for v in 0:(MV - 1)]
    brv = [Symbol(:br, j) for j in 1:NR]
    biv = [Symbol(:bi, j) for j in 1:NR]

    load_a = Any[]
    for v in 0:(MV - 1)
        push!(
            load_a,
            :(
                $(arv[v + 1]) = panel_vload(
                    Vec{$W, $R}, packed_a, packed_a_offset(kernel, $(v * W + 1), p)
                )
            )
        )
        push!(
            load_a,
            :(
                $(aiv[v + 1]) = panel_vload(
                    Vec{$W, $R}, packed_a, packed_a_offset(kernel, $(v * W + 1), p, 1)
                )
            )
        )
        push!(load_a, :($(naiv[v + 1]) = -$(aiv[v + 1])))
    end

    load_b = Any[]
    for j in 1:NR
        push!(load_b, :(($(brv[j]), $(biv[j])) = _b_step_load2(packed_b, kernel, $j, p)))
    end

    acc_exprs = Vector{Any}(undef, NA)
    for j in 1:NR, v in 0:(MV - 1)
        idx = v + MV * (j - 1) + 1
        acc_exprs[idx] = :(  # re: ar*br - ai*bi
            muladd(
                $(naiv[v + 1]), $(biv[j]),
                muladd($(arv[v + 1]), $(brv[j]), acc[$idx])
            )
        )
        acc_exprs[NV + idx] = :(  # im: ar*bi + ai*br
            muladd(
                $(aiv[v + 1]), $(brv[j]),
                muladd($(arv[v + 1]), $(biv[j]), acc[$(NV + idx)])
            )
        )
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

function add_tile(
        kernel::PlanarKernel{MR, NR, T, W}, acc::NTuple{NA, Vec{W, R}},
        packed_a::PA, packed_b::PB, k_block_length::Int
    ) where {MR, NR, T, W, R, NA, PA <: PackedPanel, PB}
    k_block_length == 0 && return acc
    k_block_length > 0 || throw_negative_k_block_length(:add_tile, k_block_length)
    @inbounds for p in 1:k_block_length
        acc = _accumulate_step_planar(kernel, acc, packed_a, packed_b, p)
    end
    return acc
end

# Generator-time accumulator indices of the re/im vectors of block `v`
# (zero-based), column `j`: planar's two planes, or `RealComplexKernel`'s
# column pairs. A `<:` test rather than dispatch, as `RealComplexKernel` is
# defined after this file.
function _planar_acc_index(kernel::Type, MV::Int, NR::Int, v::Int, j::Int)
    kernel <: PlanarKernel && return (v + MV * (j - 1) + 1, MV * NR + v + MV * (j - 1) + 1)
    kernel <: RealComplexKernel && return (v + MV * (2j - 2) + 1, v + MV * (2j - 1) + 1)
    throw(ArgumentError("_planar_acc_index: no planar accumulator layout for $kernel"))
end

# Scalar fallback store, for every destination the vector store cannot take.
@generated function _store_tile_planar!(
        destination::Tile, acc::NTuple{NA, Vec{W, R}},
        alpha::T, beta::T, kernel::Microkernel{MR, NR, T},
        m::Int, n::Int
    ) where {MR, NR, T, W, R, NA}
    MV = MR ÷ W
    NV = MV * NR
    check_acc(:_store_tile_planar!, R, T, NA, 2NV)

    blocks = Any[]
    for j in 1:NR, v in 0:(MV - 1)
        ire, iim = _planar_acc_index(kernel, MV, NR, v, j)
        push!(
            blocks, quote
                if $j <= n
                    revec = acc[$ire]
                    imvec = acc[$iim]
                    for lane in 1:$W
                        i = $(v * W) + lane
                        i <= m || break
                        axpby_tile!(
                            destination, i, $j, alpha,
                            Complex(revec[lane], imvec[lane]), beta
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

# Vector store: unit-stride rows into rank-1 dense `Complex` storage (so the
# storage can be reinterpreted as `2W` consecutive reals per `W` rows), on an
# ISA the complex fast paths ship for (shared with the complex pack fast path).
@inline _complex_vector_eligible(tile::Tile, ::Type{T}) where {T} =
    is_unit_stride(tile.rows) && dense_lanes(tile.storage, T) &&
    complex_fastpath_isa_eligible()

# Shuffle patterns built from `W` at specialization time, never hardcoded to
# one ISA. `v` holds `W` complex values `[re_0, im_0, ..., re_{W-1}, im_{W-1}]`.
@generated function _deinterleave_re(v::Vec{N, R}, ::Val{W}) where {N, R, W}
    N == 2 * W || return :(throw(ArgumentError("_deinterleave_re: expected N == 2W")))
    idx = ntuple(k -> 2 * (k - 1), W)
    return :(shufflevector(v, Val($idx)))
end

@generated function _deinterleave_im(v::Vec{N, R}, ::Val{W}) where {N, R, W}
    N == 2 * W || return :(throw(ArgumentError("_deinterleave_im: expected N == 2W")))
    idx = ntuple(k -> 2 * (k - 1) + 1, W)
    return :(shufflevector(v, Val($idx)))
end

@generated function _interleave_planes(re::Vec{W, R}, im::Vec{W, R}, ::Val{W}) where {W, R}
    idx = ntuple(k -> isodd(k) ? (k - 1) ÷ 2 : W + (k - 1) ÷ 2, 2 * W)
    return :(shufflevector(re, im, Val($idx)))
end

# One full `W`-row block; `at` is the zero-based index of its first real.
# The arithmetic transcribes Base's `Complex` expression trees (the ones
# `axpby_tile!` reaches), so full blocks match them bitwise: `*` is unfused,
# `muladd(z, w, x) = (muladd(zr, wr, -muladd(zi, wi, -xr)), muladd(zr, wi,
# muladd(zi, wr, xi)))`. Against the scalar fallback it is exact at
# `beta == 0/1` and ~1 ULP otherwise, because LLVM contracts Base's scalar
# complex `muladd` depending on inlining context.
@inline function _planar_store_block!(
        sp::Ptr{RC}, at::Int, rev::Vec{W, R}, imv::Vec{W, R},
        ar::Vec{W, R}, ai::Vec{W, R}, br::Vec{W, R}, bi::Vec{W, R},
        beta::Complex{R}, ::Val{W}
    ) where {RC, R, W}
    if iszero(beta)
        newre = ar * rev - ai * imv
        newim = ar * imv + ai * rev
    else
        old = convert(Vec{2 * W, R}, vload(Vec{2 * W, RC}, sp + sizeof(RC) * at))
        orv = _deinterleave_re(old, Val(W))
        oiv = _deinterleave_im(old, Val(W))
        if isone(beta)
            xr, xi = orv, oiv
        else
            xr = br * orv - bi * oiv
            xi = br * oiv + bi * orv
        end
        newre = muladd(ar, rev, -muladd(ai, imv, -xr))
        newim = muladd(ar, imv, muladd(ai, rev, xi))
    end
    vstore(convert(Vec{2 * W, RC}, _interleave_planes(newre, newim, Val(W))), sp + sizeof(RC) * at)
    return nothing
end

# Same unroll and full-block / row-tail split as `_store_tile_vector!`, with a
# block of `2W` reals. The raw pointer is only dereferenced inside `GC.@preserve`.
@generated function _store_tile_planar_vector!(
        destination::Tile{S, <:AffineAxis}, acc::NTuple{NA, Vec{W, R}},
        alpha::T, beta::T, kernel::Microkernel{MR, NR, T},
        m::Int, n::Int
    ) where {S, MR, NR, T, W, R, NA}
    # The pointer reinterpretation is only sound on dense rank-1 complex storage.
    S <: DenseVector && lane_convertible(eltype(S), T) ||
        throw(ArgumentError("_store_tile_planar_vector!: storage $S is not a dense vector convertible to $T"))
    RC = real(eltype(S))
    MV = MR ÷ W
    NV = MV * NR
    check_acc(:_store_tile_planar_vector!, R, T, NA, 2NV)

    blocks = Any[]
    for j in 1:NR
        vblocks = Any[]
        for v in 0:(MV - 1)
            ire, iim = _planar_acc_index(kernel, MV, NR, v, j)
            push!(
                vblocks, quote
                    revec = acc[$ire]
                    imvec = acc[$iim]
                    if $((v + 1) * W) <= m
                        _planar_store_block!(
                            sp, 2 * (colbase + $(v * W)), revec, imvec,
                            ar, ai, br, bi, beta, Val($W)
                        )
                    elseif $(v * W) < m
                        for lane in 1:$W
                            i = $(v * W) + lane
                            i <= m || break
                            axpby_at!(
                                storage, colbase + i, alpha,
                                Complex(revec[lane], imvec[lane]), beta
                            )
                        end
                    end
                end
            )
        end
        push!(
            blocks, quote
                if $j <= n
                    colbase = rowbase0 + cols[$j]  # the address of (1, j)
                    $(vblocks...)
                end
            end
        )
    end

    return quote
        Base.@_inline_meta
        storage = destination.storage
        cols = destination.cols
        rowbase0 = @inbounds destination.base + destination.rows[1]
        ar = Vec{$W, $R}(real(alpha))
        ai = Vec{$W, $R}(imag(alpha))
        br = Vec{$W, $R}(real(beta))
        bi = Vec{$W, $R}(imag(beta))
        GC.@preserve storage begin
            sp = reinterpret(Ptr{$RC}, pointer(storage))
            @inbounds begin
                $(blocks...)
            end
        end
        return destination
    end
end

# `@inline` with the vector store, for the reason at the real `store_tile!`.
@inline function store_tile!(
        destination::Tile, acc::NTuple{NA, Vec{W, R}},
        alpha::T, beta::T, kernel::PlanarKernel{MR, NR, T, W}
    ) where {MR, NR, T, W, R, NA}
    m, n = store_prologue!(destination, alpha, beta)
    (m == 0 || n == 0) && return destination

    if _complex_vector_eligible(destination, T)
        return _store_tile_planar_vector!(destination, acc, alpha, beta, kernel, m, n)
    end

    return _store_tile_planar!(destination, acc, alpha, beta, kernel, m, n)
end
