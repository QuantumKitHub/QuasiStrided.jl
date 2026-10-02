# The stores: one generic `store_tile!` per kernel, picking the vector or the
# scalar store; each accumulator layout supplies its full-block store and lane
# value.

using SIMD: Vec, vload, vstore

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

# Vector store eligibility: unit-stride rows into rank-1 dense storage, exactly
# what SIMD.jl's array `vload`/`vstore` accept. Must admit `Memory{T}`: that is
# the `parent` of an Array-backed `StridedView` on Julia >= 1.11. The complex
# layouts reinterpret `W` rows as `2W` consecutive reals, on an ISA the complex
# fast paths ship for (shared with the complex pack fast path).
@inline vector_store_eligible(::RealLayout, tile::Tile, ::Type{T}) where {T} =
    is_unit_stride(tile.rows) && dense_lanes(tile.storage, T)
@inline vector_store_eligible(::AccumulatorLayout, tile::Tile, ::Type{T}) where {T} =
    is_unit_stride(tile.rows) && dense_lanes(tile.storage, T) &&
    complex_fastpath_isa_eligible()

# One full `W`-row block for `beta` case `B`; `at` is the zero-based index of
# its first real.
# The arithmetic transcribes Base's `Complex` expression trees (the ones
# `axpby_tile!` reaches), so full blocks match them bitwise: `*` is unfused,
# `muladd(z, w, x) = (muladd(zr, wr, -muladd(zi, wi, -xr)), muladd(zr, wi,
# muladd(zi, wr, xi)))`. Against the scalar fallback it is exact at
# `beta == 0/1` and ~1 ULP otherwise, because LLVM contracts Base's scalar
# complex `muladd` depending on inlining context.
@inline function split_store_block!(
        sp::Ptr{RC}, at::Int, rev::Vec{W, R}, imv::Vec{W, R},
        ar::Vec{W, R}, ai::Vec{W, R}, br::Vec{W, R}, bi::Vec{W, R},
        ::Val{W}, ::Val{B}
    ) where {RC, R, W, B}
    if B === :zero
        newre = ar * rev - ai * imv
        newim = ar * imv + ai * rev
    else
        old = convert(Vec{2 * W, R}, vload(Vec{2 * W, RC}, sp + sizeof(RC) * at))
        orv = deinterleave_re(old, Val(W))
        oiv = deinterleave_im(old, Val(W))
        if B === :one
            xr, xi = orv, oiv
        else
            xr = br * orv - bi * oiv
            xi = br * oiv + bi * orv
        end
        newre = muladd(ar, rev, -muladd(ai, imv, -xr))
        newim = muladd(ar, imv, muladd(ai, rev, xi))
    end
    vstore(convert(Vec{2 * W, RC}, interleave_planes(newre, newim, Val(W))), sp + sizeof(RC) * at)
    return nothing
end

# One full `W÷2`-row block, already in `Complex`'s memory order, so no
# interleave shuffle. Base's `Complex` expression trees in lanes, as planar's
# `split_store_block!`:
#     beta == 0:  addsub(ar*r, ai*swap(r))
#     beta == 1:  fmaddsub(ar, r, fmaddsub(ai, swap(r), C))
#     otherwise:  as beta == 1 with C := addsub(br*C, bi*swap(C))
@inline function lanepair_store_block!(
        sp::Ptr{RC}, at::Int, r::Vec{W, R},
        ar::Vec{W, R}, ai::Vec{W, R}, br::Vec{W, R}, bi::Vec{W, R},
        ::Val{B}
    ) where {RC, R, W, B}
    s = swap_pairs(r)
    if B === :zero
        new = addsub(ar * r, ai * s)
    else
        old = convert(Vec{W, R}, vload(Vec{W, RC}, sp + sizeof(RC) * at))
        x = B === :one ? old : addsub(br * old, bi * swap_pairs(old))
        new = fmaddsub(ar, r, fmaddsub(ai, s, x))
    end
    vstore(convert(Vec{W, RC}, new), sp + sizeof(RC) * at)
    return nothing
end

@generated function store_tile!(
        destination::Tile, acc::NTuple{NA, Vec{W, R}},
        alpha::T, beta::T, kernel::VectorKernel{MR, NR, T, W}
    ) where {MR, NR, T, W, R, NA}
    layout = AccumulatorLayout(kernel)
    return quote
        $(inline_store(layout) ? :(Base.@_inline_meta) : nothing)
        m, n = store_prologue!(destination, alpha, beta)
        (m == 0 || n == 0) && return destination

        if vector_store_eligible($layout, destination, T)
            return vector_store!(destination, acc, alpha, beta, kernel, m, n)
        end

        return scalar_store!(destination, acc, alpha, beta, kernel, m, n)
    end
end

# Element by element, for every destination the vector store cannot take.
@generated function scalar_store!(
        destination::Tile, acc::NTuple{NA, Vec{W, R}},
        alpha::T, beta::T, kernel::VectorKernel{MR, NR, T, W},
        m::Int, n::Int
    ) where {MR, NR, T, W, R, NA}
    layout = AccumulatorLayout(kernel)
    rows = rows_per_vector(layout, W)
    MV = MR ÷ rows
    check_acc(:scalar_store!, R, T, NA, accumulator_length(layout, MV, NR))

    blocks = Any[]
    for j in 1:NR, v in 0:(MV - 1)
        push!(
            blocks, quote
                if $j <= n
                    $(acc_bindings(layout, kernel, MV, NR, v, j))
                    for lane in 1:$rows
                        i = $(v * rows) + lane
                        i <= m || break
                        axpby_tile!(destination, i, $j, alpha, $(lane_value(layout)), beta)
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

# Whole row blocks are one vector load/store; a block straddling `m` is stored
# lane by lane, so nothing outside the valid rectangle is touched. The blocks
# are generated once per `beta` case, so `beta` is tested once per tile.
# `rows::AffineAxis` in the signature: an ineligible tile is a MethodError.
@generated function vector_store!(
        destination::Tile{S, <:AffineAxis}, acc::NTuple{NA, Vec{W, R}},
        alpha::T, beta::T, kernel::VectorKernel{MR, NR, T, W},
        m::Int, n::Int
    ) where {S, MR, NR, T, W, R, NA}
    layout = AccumulatorLayout(kernel)
    rows = rows_per_vector(layout, W)
    MV = MR ÷ rows
    check_acc(:vector_store!, R, T, NA, accumulator_length(layout, MV, NR))
    RC = store_lanetype(layout, S, T)

    function blocks(B)
        out = Any[]
        for j in 1:NR
            vblocks = Any[]
            for v in 0:(MV - 1)
                push!(
                    vblocks, quote
                        $(acc_bindings(layout, kernel, MV, NR, v, j))
                        if $((v + 1) * rows) <= m
                            $(block_store(layout, :(colbase + $(v * rows)), W, R, RC, B))
                        elseif $(v * rows) < m
                            for lane in 1:$rows
                                i = $(v * rows) + lane
                                i <= m || break
                                axpby_at!(storage, colbase + i, alpha, $(lane_value(layout)), beta)
                            end
                        end
                    end
                )
            end
            push!(
                out, quote
                    if $j <= n
                        colbase = rowbase0 + cols[$j]  # the address of (1, j)
                        $(vblocks...)
                    end
                end
            )
        end
        return out
    end
    by_beta = quote
        if iszero(beta)
            $(blocks(:zero)...)
        elseif isone(beta)
            $(blocks(:one)...)
        else
            $(blocks(:general)...)
        end
    end
    return quote
        $(inline_store(layout) ? :(Base.@_inline_meta) : nothing)
        storage = destination.storage
        cols = destination.cols
        rowbase0 = @inbounds destination.base + destination.rows[1]
        $(store_body(layout, W, R, RC, by_beta))
        return destination
    end
end

function store_tile!(
        destination::Tile, acc::NTuple{N, T},
        alpha::T, beta::T, kernel::ScalarKernel{MR}
    ) where {MR, N, T}
    m, n = store_prologue!(destination, alpha, beta)
    (m == 0 || n == 0) && return destination
    if iszero(beta)
        @inbounds for j in 1:n, i in 1:m
            destination[i, j] = alpha * acc[i + MR * (j - 1)]
        end
    elseif isone(beta)
        @inbounds for j in 1:n, i in 1:m
            destination[i, j] = muladd(alpha, acc[i + MR * (j - 1)], convert(T, destination[i, j]))
        end
    else
        @inbounds for j in 1:n, i in 1:m
            destination[i, j] = muladd(alpha, acc[i + MR * (j - 1)], beta * convert(T, destination[i, j]))
        end
    end
    return destination
end
