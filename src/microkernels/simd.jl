# Explicit-SIMD real microkernel. The accumulator is an immutable tuple of
# `Vec{W,T}` so it stays in registers, and every K step and store is
# `@generated` straight-line code.
#
# GUARDRAIL (all kernels): every `acc[...]` must be a *literal* tuple index.
# Indexing an `NTuple` dynamically forces it to memory, and above 16 vectors
# the compiler heap-allocates it on every call. Only the lane index inside one
# `Vec` may be a runtime value.

using SIMD: Vec, vload, vstore

"""
    SIMDKernel(::Val{MR}, ::Val{NR}, ::Type{T}[, ::Val{W}])

Explicit-SIMD real microkernel over `SIMD.Vec{W,T}` lanes. `MR` must be a
multiple of `W` (default: one 256-bit register, 4 for `Float64`, 8 for `Float32`).
"""
struct SIMDKernel{MR, NR, T, W} <: DescriptorKernel{MR, NR, T}
    descriptor::KernelDescriptor{MR, NR, T}

    function SIMDKernel{MR, NR, T, W}(descriptor::KernelDescriptor{MR, NR, T}) where {MR, NR, T, W}
        _check_vector_shape("SIMDKernel", MR, W)
        return new{MR, NR, T, W}(descriptor)
    end
end

SIMDKernel(::Val{MR}, ::Val{NR}, ::Type{T}, ::Val{W}) where {MR, NR, T, W} =
    SIMDKernel{MR, NR, T, W}(KernelDescriptor(Val(MR), Val(NR), T))
SIMDKernel(::Val{MR}, ::Val{NR}, ::Type{T}) where {MR, NR, T} =
    SIMDKernel(Val(MR), Val(NR), T, Val(_default_lanewidth(T)))

lanewidth(::SIMDKernel{MR, NR, T, W}) where {MR, NR, T, W} = W

# `(MR÷W)*NR` vectors; vector `v` of column `j` is at `v + (MR÷W)*j + 1`.
function zero_accumulator(kernel::SIMDKernel{MR, NR, T, W}) where {MR, NR, T, W}
    z = zero(Vec{W, T})
    return ntuple(_ -> z, Val((MR ÷ W) * NR))
end

# B column `j` at K step `p`; `UnpackedBView` (src/execution/unpackedb.jl)
# overrides it to read B in place, resolved at compile time.
@inline _b_step_load(packed_b::PB, kernel, j::Int, p::Int) where {PB} =
    panel_load(packed_b, packed_b_offset(kernel, j, p))

@generated function _accumulate_step(
        kernel::SIMDKernel{MR, NR, T, W}, acc::NTuple{NV, Vec{W, T}},
        packed_a::PA, packed_b::PB, p::Int
    ) where {MR, NR, T, W, NV, PA, PB}
    NVECA = MR ÷ W
    _check_acc(:_accumulate_step, T, T, NV, NVECA * NR)

    avars = [Symbol(:a, v) for v in 0:(NVECA - 1)]
    bvars = [Symbol(:b, j) for j in 0:(NR - 1)]

    load_a = [
        :($(avars[v + 1]) = panel_vload(Vec{$W, $T}, packed_a, packed_a_offset(kernel, $(v * W), p)))
            for v in 0:(NVECA - 1)
    ]
    load_b = [
        :($(bvars[j + 1]) = _b_step_load(packed_b, kernel, $j, p))
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

# Vector store eligibility: unit-stride rows into rank-1 dense storage of `T`
# or of another real type the lanes convert to and from, exactly what SIMD.jl's
# array `vload`/`vstore` accept. Must admit `Memory`: that is the `parent` of an
# Array-backed `StridedView` on Julia >= 1.11.
@inline _vector_store_eligible(tile::QSTile, ::Type{T}) where {T} =
    _unit_stride_rows(tile.rows) && _dense_lanes(tile.storage, T)

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

# Whole `W`-row blocks are one vector load/store; a block straddling `m` is
# stored lane by lane, so nothing outside the valid rectangle is touched.
# `rows::AffineAxis` in the signature: an ineligible tile is a MethodError.
# `C` is loaded into and rounded back from `T` lanes.
@generated function _store_tile_vector!(
        destination::QSTile{S, <:AffineAxis}, acc::NTuple{NV, Vec{W, T}},
        alpha::T, beta::T, kernel::SIMDKernel{MR, NR, T, W},
        m::Int, n::Int
    ) where {S, MR, NR, T, W, NV}
    NVECA = MR ÷ W
    RC = eltype(S)
    old = :(convert(Vec{$W, $T}, vload(Vec{$W, $RC}, storage, at)))
    blocks = Any[]
    for j in 0:(NR - 1)
        vblocks = Any[]
        for v in 0:(NVECA - 1)
            idx = v + NVECA * j + 1
            push!(
                vblocks, quote
                    vec = acc[$idx]
                    if $((v + 1) * W) <= m
                        at = colbase + $(v * W) + 1
                        vstore(
                            convert(
                                Vec{$W, $RC},
                                iszero(beta) ? alpha * vec :
                                    isone(beta) ? muladd(alpha, vec, $old) :
                                    muladd(alpha, vec, beta * $old)
                            ),
                            storage, at
                        )
                    elseif $(v * W) < m
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
                    colbase = rowbase0 + axis_offset(cols, $j)  # zero-based (0, j)
                    $(vblocks...)
                end
            end
        )
    end
    return quote
        Base.@_inline_meta
        storage = destination.storage
        cols = destination.cols
        rowbase0 = destination.base + destination.rows.base
        @inbounds begin
            $(blocks...)
        end
        return destination
    end
end

# `@inline` with the vector store: out of line, the whole accumulator is spilled
# to the stack and reloaded on every micro-tile. The scattered store stays out
# of line: it is scalar anyway, and inlining it bloats the tile function.
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
