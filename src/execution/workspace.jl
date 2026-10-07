# One M or N macro block's offsets in the group's two operands, and the
# per-sliver descriptors of each.
struct GroupBuffers
    offsets::NTuple{2, Vector{Int}}
    descriptors::NTuple{2, Vector{BlockDescriptor}}
end

GroupBuffers(block::Int, tiles::Int) = GroupBuffers(
    (Vector{Int}(undef, block), Vector{Int}(undef, block)),
    (Vector{BlockDescriptor}(undef, tiles), Vector{BlockDescriptor}(undef, tiles))
)

"""
    ContractWorkspace{T,VT<:AbstractVector,PT<:AbstractVector}

The buffers [`execute!`](@ref) needs, built by [`plan_contract`](@ref) and
reused by every `execute!` on that plan. One from a non-default allocator must
be [`release!`](@ref)d. `T` is the plan's compute type; the packed panels
(`VT`) hold `real(T)`, the panel of C (`PT`) holds `T`. Field layout is an
implementation detail.
"""
# A `mutable struct` with `const` fields so a plan holds one pointer to it
# instead of copying every GC-tracked field.
mutable struct ContractWorkspace{T, VT <: AbstractVector, PT <: AbstractVector}
    const m::GroupBuffers
    const n::GroupBuffers
    # K blocks are described whole, so K has offsets only.
    const k::NTuple{2, Vector{Int}}

    # The only allocator-routed buffers.
    const packed_a::VT
    const packed_b::VT
    # The compute-type panel of C for an eltype of C narrower than `T` (see
    # `PanelPath`); empty otherwise.
    const c_panel::PT
    # The dot path's gathered vector (complex: and its pair-swapped copy).
    const dot_vector::Vector{T}

    # GUARDRAIL: the packed panels hold `real(T)`, never `T`.
    function ContractWorkspace{T}(
            m::GroupBuffers, n::GroupBuffers, k::NTuple{2, Vector{Int}},
            packed_a::VT, packed_b::VT, c_panel::PT, dot_vector::Vector{T}
        ) where {T, VT <: AbstractVector, PT <: AbstractVector}
        eltype(VT) === real(T) || throw(
            ArgumentError(
                "ContractWorkspace{$T,$VT}: the packed panels must hold $(real(T)) " *
                    "(the real type of the compute type $T), got $(eltype(VT))"
            )
        )
        return new{T, VT, PT}(m, n, k, packed_a, packed_b, c_panel, dot_vector)
    end
end

# Buffer lengths for `kernel` on `path`: M and N offsets (`m`, `n`) and
# sliver counts (`m_tiles`, `n_tiles`), packed reals and the dot vector. Every
# path keeps at least one tile of M and N offsets for `scale_all_of_C!`.
# GUARDRAIL: the packed lengths count reals, the sliver counts logical
# `m_tile`/`n_tile`; do not rescale either.
function workspace_sizes(::Type{T}, kernel, blocking::Blocking, path) where {T}
    m_tile, n_tile = tile_size(kernel)
    a_reals, b_reals = map(reals_per_element, pack_formats(typeof(kernel)))
    m_tiles = cld(blocking.m_block, m_tile)
    n_tiles = cld(blocking.n_block, n_tile)
    return (
        m = blocking.m_block, n = blocking.n_block, m_tiles, n_tiles,
        packed_a = m_tiles * m_tile * a_reals * blocking.k_block,
        packed_b = n_tiles * n_tile * b_reals * blocking.k_block, dot_vector = 0,
    )
end
# The dot path reads the matrix in place; its vector side's block is one tile.
workspace_sizes(::Type{T}, kernel, blocking::Blocking, ::DotPath) where {T} = (
    m = blocking.m_block, n = blocking.n_block, m_tiles = 0, n_tiles = 0, packed_a = 0, packed_b = 0,
    dot_vector = (T <: Complex ? 2 : 1) * blocking.k_block,
)
# M offsets only for `scale_all_of_C!`.
workspace_sizes(::Type{T}, kernel, blocking::Blocking, ::OuterPath) where {T} = (
    m = tile_size(kernel, 1), n = blocking.n_block, m_tiles = 0, n_tiles = 0, packed_a = 0, packed_b = 0,
    dot_vector = 0,
)

"""
    ContractWorkspace(T, kernel, blocking::Blocking;
                      allocator = TensorOperations.DefaultAllocator(), panel = 0,
                      path = nothing)

A workspace for compute type `T`, `kernel` and an effective `blocking`.
`panel` is the length of the compute-type panel of C (see
[`plan_contract`](@ref)); `path` is the plan's execution path, and the
dot and outer-product paths get no packed panels. The packed panels and the
panel of C come from `allocator` (`tensoralloc`) and must be handed back with
[`release!`](@ref) unless it is the default one.
"""
function ContractWorkspace(
        ::Type{T}, kernel, blocking::Blocking;
        allocator = TO.DefaultAllocator(), panel::Int = 0, path = nothing
    ) where {T}
    s = workspace_sizes(T, kernel, blocking, path)
    R = realtype(kernel)
    # `release!` frees in exactly the reverse order, for arena allocators.
    packed_a = TO.tensoralloc(Vector{R}, (s.packed_a,), Val(true), allocator)
    packed_b = TO.tensoralloc(Vector{R}, (s.packed_b,), Val(true), allocator)
    c_panel = TO.tensoralloc(Vector{T}, (panel,), Val(true), allocator)
    k = (Vector{Int}(undef, blocking.k_block), Vector{Int}(undef, blocking.k_block))
    return ContractWorkspace{T}(
        GroupBuffers(s.m, s.m_tiles), GroupBuffers(s.n, s.n_tiles), k,
        packed_a, packed_b, c_panel, Vector{T}(undef, s.dot_vector)
    )
end

"""
    release!(ws::ContractWorkspace, allocator)
    release!(plan::ContractPlan, allocator)

Hand the packed panels and panel of C back to `allocator`
(`TensorOperations.tensorfree!`) in reverse acquisition order; the workspace
must not be used afterwards. A no-op for the default allocator.
"""
function release!(ws::ContractWorkspace, allocator)
    TO.tensorfree!(ws.c_panel, allocator)
    TO.tensorfree!(ws.packed_b, allocator)
    TO.tensorfree!(ws.packed_a, allocator)
    return nothing
end
