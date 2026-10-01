# The vector kernels' shared `add_tile` and stores, and the explicit-SIMD real
# K step. The accumulator is an immutable tuple of `Vec`s so it stays in
# registers, and every K step and store is `@generated` straight-line code.
#
# GUARDRAIL (all kernels): every `acc[...]` must be a *literal* tuple index.
# Indexing an `NTuple` dynamically forces it to memory, and above 16 vectors
# the compiler heap-allocates it on every call. Only the lane index inside one
# `Vec` may be a runtime value.

using SIMD: Vec, vload, vstore

function zero_accumulator(kernel::VectorKernel{MR, NR, T, W}) where {MR, NR, T, W}
    layout = accumulator_layout(kernel)
    z = zero(Vec{W, real(T)})
    return ntuple(_ -> z, Val(accumulator_length(layout, MR ÷ rows_per_vector(layout, W), NR)))
end

k_steps(::VectorKernel, k_block_length::Int) = k_block_length

@inline function add_tile(
        kernel::VectorKernel, acc::NTuple{NA, Vec{W, R}},
        packed_a::PA, packed_b::PB, k_block_length::Int
    ) where {NA, W, R, PA <: PackedPanel, PB}
    k_block_length == 0 && return acc
    k_block_length > 0 || throw_negative_k_block_length(:add_tile, k_block_length)
    @inbounds for p in 1:k_steps(kernel, k_block_length)
        acc = accumulate_step(kernel, acc, packed_a, packed_b, p)
    end
    return acc
end

# B column `j` at K step `p`, as a real or as `(re, im)`; `UnpackedBView`
# (src/execution/unpackedb.jl) overrides both to read B in place, resolved at
# compile time.
@inline b_scalar(packed_b::PB, kernel, j::Int, p::Int) where {PB} =
    panel_load(packed_b, packed_b_offset(kernel, j, p))
@inline b_complex(packed_b::PB, kernel, j::Int, p::Int) where {PB} = (
    panel_load(packed_b, packed_b_offset(kernel, j, p)),
    panel_load(packed_b, packed_b_offset(kernel, j, p, 1)),
)

@generated function accumulate_step(
        kernel::SIMDKernel{MR, NR, T, W}, acc::NTuple{NV, Vec{W, T}},
        packed_a::PA, packed_b::PB, p::Int
    ) where {MR, NR, T, W, NV, PA, PB}
    NVECA = MR ÷ W
    check_acc(:accumulate_step, T, T, NV, NVECA * NR)

    avars = [Symbol(:a, v) for v in 0:(NVECA - 1)]
    bvars = [Symbol(:b, j) for j in 1:NR]

    load_a = [
        :($(avars[v + 1]) = panel_vload(Vec{$W, $T}, packed_a, packed_a_offset(kernel, $(v * W + 1), p)))
            for v in 0:(NVECA - 1)
    ]
    load_b = [
        :($(bvars[j]) = b_scalar(packed_b, kernel, $j, p))
            for j in 1:NR
    ]

    acc_exprs = Vector{Any}(undef, NV)
    for j in 1:NR, v in 0:(NVECA - 1)
        idx = acc_index(NVECA, v, j)
        acc_exprs[idx] = :(muladd($(avars[v + 1]), $(bvars[j]), acc[$idx]))
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

# A kernel without a K step of its own runs its `inner(kernel)`'s.
@inline accumulate_step(kernel::VectorKernel, acc::NTuple, packed_a::PA, packed_b::PB, p::Int) where {PA, PB} =
    accumulate_step(inner(kernel), acc, packed_a, packed_b, p)

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

@generated function store_tile!(
        destination::Tile, acc::NTuple{NA, Vec{W, R}},
        alpha::T, beta::T, kernel::VectorKernel{MR, NR, T, W}
    ) where {MR, NR, T, W, R, NA}
    layout = accumulator_layout(kernel)
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
    layout = accumulator_layout(kernel)
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
# lane by lane, so nothing outside the valid rectangle is touched.
# `rows::AffineAxis` in the signature: an ineligible tile is a MethodError.
@generated function vector_store!(
        destination::Tile{S, <:AffineAxis}, acc::NTuple{NA, Vec{W, R}},
        alpha::T, beta::T, kernel::VectorKernel{MR, NR, T, W},
        m::Int, n::Int
    ) where {S, MR, NR, T, W, R, NA}
    layout = accumulator_layout(kernel)
    rows = rows_per_vector(layout, W)
    MV = MR ÷ rows
    check_acc(:vector_store!, R, T, NA, accumulator_length(layout, MV, NR))
    RC = store_lanetype(layout, S, T)

    blocks = Any[]
    for j in 1:NR
        vblocks = Any[]
        for v in 0:(MV - 1)
            push!(
                vblocks, quote
                    $(acc_bindings(layout, kernel, MV, NR, v, j))
                    if $((v + 1) * rows) <= m
                        $(block_store(layout, :(colbase + $(v * rows)), W, R, RC))
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
            blocks, quote
                if $j <= n
                    colbase = rowbase0 + cols[$j]  # the address of (1, j)
                    $(vblocks...)
                end
            end
        )
    end
    return quote
        $(inline_store(layout) ? :(Base.@_inline_meta) : nothing)
        storage = destination.storage
        cols = destination.cols
        rowbase0 = @inbounds destination.base + destination.rows[1]
        $(store_body(layout, W, R, RC, blocks))
        return destination
    end
end
