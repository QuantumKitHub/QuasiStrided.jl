# Typed payload slots for the dispatch barriers (src/execution/barrier.jl):
# one `Base.RefValue{P}` per payload type, plus the last one used. They live
# as long as the workspace, which is what lets a barrier call allocate nothing.
mutable struct _SlotCache
    last::Any
    const slots::IdDict{Any, Any}
end
_SlotCache() = _SlotCache(nothing, IdDict{Any, Any}())

"""
    ContractWorkspace{T,VT<:AbstractVector}

The buffers [`execute!`](@ref) needs. Pass one back as
`plan_contract(...; workspace = ws)` to reuse (and grow) it across
contractions; one from a non-default allocator is single-use and must be
[`release!`](@ref)d. `T` is the storage element type; the packed panels (`VT`)
hold `real(T)`. Field layout is an implementation detail.
"""
# A `mutable struct` with `const` fields so a plan and a barrier slot hold one
# pointer to it instead of copying twenty-one GC-tracked fields.
mutable struct ContractWorkspace{T, VT <: AbstractVector}
    # Macro-block offset buffers and their per-sliver descriptors.
    const m_buf_A::Vector{Int}
    const m_buf_C::Vector{Int}
    const n_buf_B::Vector{Int}
    const n_buf_C::Vector{Int}
    const k_buf_A::Vector{Int}
    const k_buf_B::Vector{Int}
    const m_desc_A::Vector{BlockDescriptor}
    const m_desc_C::Vector{BlockDescriptor}
    const n_desc_B::Vector{BlockDescriptor}
    const n_desc_C::Vector{BlockDescriptor}

    # Packed macro panels: the only allocator-routed buffers.
    const packed_a::VT
    const packed_b::VT

    # One register tile's offsets: the beta-only pass and the oracle.
    const tile_m_buf_A::Vector{Int}
    const tile_m_buf_C::Vector{Int}
    const tile_n_buf_B::Vector{Int}
    const tile_n_buf_C::Vector{Int}

    # `execute_tilewise!`'s own buffers, empty under `oracle = false`; not
    # shared, so the oracle has no state in common with what it checks.
    const tw_k_buf_A::Vector{Int}
    const tw_k_buf_B::Vector{Int}
    const tw_packed_a::VT
    const tw_packed_b::VT

    const slots::_SlotCache

    # GUARDRAIL: the packed panels hold `real(T)`, never `T`.
    function ContractWorkspace{T, VT}(
            m_buf_A::Vector{Int}, m_buf_C::Vector{Int},
            n_buf_B::Vector{Int}, n_buf_C::Vector{Int},
            k_buf_A::Vector{Int}, k_buf_B::Vector{Int},
            m_desc_A::Vector{BlockDescriptor}, m_desc_C::Vector{BlockDescriptor},
            n_desc_B::Vector{BlockDescriptor}, n_desc_C::Vector{BlockDescriptor},
            packed_a::VT, packed_b::VT,
            tile_m_buf_A::Vector{Int}, tile_m_buf_C::Vector{Int},
            tile_n_buf_B::Vector{Int}, tile_n_buf_C::Vector{Int},
            tw_k_buf_A::Vector{Int}, tw_k_buf_B::Vector{Int},
            tw_packed_a::VT, tw_packed_b::VT
        ) where {T, VT <: AbstractVector}
        eltype(VT) === real(T) || throw(
            ArgumentError(
                "ContractWorkspace{$T,$VT}: the packed panels must hold $(real(T)) " *
                    "(the real type of the storage element type $T), got $(eltype(VT))"
            )
        )
        return new{T, VT}(
            m_buf_A, m_buf_C, n_buf_B, n_buf_C, k_buf_A, k_buf_B,
            m_desc_A, m_desc_C, n_desc_B, n_desc_C,
            packed_a, packed_b,
            tile_m_buf_A, tile_m_buf_C, tile_n_buf_B, tile_n_buf_C,
            tw_k_buf_A, tw_k_buf_B, tw_packed_a, tw_packed_b, _SlotCache(),
        )
    end
end

# Buffer lengths for `kernel` at the effective `blocking`, shared by the
# constructors and `reserve!`. GUARDRAIL: `packed_a_length` already counts
# reals, the sliver counts are in logical `mr`/`nr`; do not rescale either.
@inline function _workspace_sizes(kernel, blocking::Blocking)
    MRk = mr(kernel)
    NRk = nr(kernel)
    kc = blocking.kc
    pa = packed_a_length(kernel, kc)
    pb = packed_b_length(kernel, kc)
    m_slivers = cld(blocking.mc, MRk)
    n_slivers = cld(blocking.nc, NRk)
    return (
        mc = blocking.mc, nc = blocking.nc, kc = kc,
        mr = MRk, nr = NRk,
        m_slivers = m_slivers, n_slivers = n_slivers,
        packed_a = m_slivers * pa, packed_b = n_slivers * pb,
        tw_packed_a = pa, tw_packed_b = pb,
    )
end

# A temporary, never `resize!`d.
@inline function _alloc_temp(::Type{T}, n::Int, allocator) where {T}
    return TO.tensoralloc(Vector{T}, (n,), Val(true), allocator)
end

# A non-temporary, which every allocator serves as a plain `Vector{Int}`.
@inline function _alloc_offsets(n::Int, allocator)
    return TO.tensoralloc(Vector{Int}, (n,), Val(false), allocator)::Vector{Int}
end

# Not tensor buffers, so never through `tensoralloc`.
@inline _alloc_descriptors(n::Int) = Vector{BlockDescriptor}(undef, n)

@inline function _build_workspace(
        ::Type{T}, s, ntw::Int, ints::F,
        packed_a::VT, packed_b::VT, tw_packed_a::VT, tw_packed_b::VT
    ) where {T, F, VT <: AbstractVector}
    return ContractWorkspace{T, VT}(
        ints(s.mc), ints(s.mc),
        ints(s.nc), ints(s.nc),
        ints(s.kc), ints(s.kc),
        _alloc_descriptors(s.m_slivers), _alloc_descriptors(s.m_slivers),
        _alloc_descriptors(s.n_slivers), _alloc_descriptors(s.n_slivers),
        packed_a, packed_b,
        ints(s.mr), ints(s.mr),
        ints(s.nr), ints(s.nr),
        ints(ntw), ints(ntw),
        tw_packed_a, tw_packed_b,
    )
end

"""
    ContractWorkspace(T, kernel, blocking::Blocking;
                      oracle = true, allocator = TensorOperations.DefaultAllocator())

A workspace for storage type `T`, `kernel` and an effective `blocking`.
`oracle = false` leaves the `execute_tilewise!` buffers empty. Under a
non-default `allocator` the packed panels come from `tensoralloc`, are never
resized, and must be handed back with [`release!`](@ref).
"""
function ContractWorkspace(
        ::Type{T}, kernel, blocking::Blocking, oracle::Bool, ::TO.DefaultAllocator
    ) where {T}
    s = _workspace_sizes(kernel, blocking)
    R = realtype(kernel)
    return _build_workspace(
        T, s, oracle ? s.kc : 0, n -> Vector{Int}(undef, n),
        Vector{R}(undef, s.packed_a), Vector{R}(undef, s.packed_b),
        Vector{R}(undef, oracle ? s.tw_packed_a : 0),
        Vector{R}(undef, oracle ? s.tw_packed_b : 0),
    )
end

function ContractWorkspace(
        ::Type{T}, kernel, blocking::Blocking, oracle::Bool, allocator
    ) where {T}
    s = _workspace_sizes(kernel, blocking)
    R = realtype(kernel)

    # `release!` frees in exactly the reverse order, for arena allocators.
    packed_a = _alloc_temp(R, s.packed_a, allocator)
    packed_b = _alloc_temp(R, s.packed_b, allocator)
    tw_packed_a = _alloc_temp(R, oracle ? s.tw_packed_a : 0, allocator)
    tw_packed_b = _alloc_temp(R, oracle ? s.tw_packed_b : 0, allocator)

    return _build_workspace(
        T, s, oracle ? s.kc : 0, n -> _alloc_offsets(n, allocator),
        packed_a, packed_b, tw_packed_a, tw_packed_b
    )
end

function ContractWorkspace(
        ::Type{T}, kernel, blocking::Blocking;
        oracle::Bool = true, allocator = TO.DefaultAllocator()
    ) where {T}
    return ContractWorkspace(T, kernel, blocking, oracle, allocator)
end

# Grow `ws` in place (never shrink, never reallocate what is large enough)
# for `kernel` at `blocking`. Only for the GC-owned path: an allocator's
# temporaries must never be resized.
function reserve!(
        ws::ContractWorkspace{T, Vector{R}}, kernel, blocking::Blocking, oracle::Bool
    ) where {T, R}
    s = _workspace_sizes(kernel, blocking)

    _grow!(ws.m_buf_A, s.mc)
    _grow!(ws.m_buf_C, s.mc)
    _grow!(ws.n_buf_B, s.nc)
    _grow!(ws.n_buf_C, s.nc)
    _grow!(ws.k_buf_A, s.kc)
    _grow!(ws.k_buf_B, s.kc)

    _grow!(ws.m_desc_A, s.m_slivers)
    _grow!(ws.m_desc_C, s.m_slivers)
    _grow!(ws.n_desc_B, s.n_slivers)
    _grow!(ws.n_desc_C, s.n_slivers)

    _grow!(ws.packed_a, s.packed_a)
    _grow!(ws.packed_b, s.packed_b)

    _grow!(ws.tile_m_buf_A, s.mr)
    _grow!(ws.tile_m_buf_C, s.mr)
    _grow!(ws.tile_n_buf_B, s.nr)
    _grow!(ws.tile_n_buf_C, s.nr)

    if oracle
        _grow!(ws.tw_k_buf_A, s.kc)
        _grow!(ws.tw_k_buf_B, s.kc)
        _grow!(ws.tw_packed_a, s.tw_packed_a)
        _grow!(ws.tw_packed_b, s.tw_packed_b)
    end

    return ws
end

@inline function _grow!(v::Vector, n::Int)
    length(v) < n && resize!(v, n)
    return v
end

"""
    release!(ws::ContractWorkspace, allocator)

Hand `ws`'s packed panels back to `allocator` (`TensorOperations.tensorfree!`)
in reverse acquisition order; `ws` must not be used afterwards. Correct for
every allocator.
"""
function release!(ws::ContractWorkspace, allocator)
    TO.tensorfree!(ws.tw_packed_b, allocator)
    TO.tensorfree!(ws.tw_packed_a, allocator)
    TO.tensorfree!(ws.packed_b, allocator)
    TO.tensorfree!(ws.packed_a, allocator)
    return nothing
end

# Build or reuse the plan's workspace, dispatching on the allocator type.
function _resolve_workspace(
        ::Type{T}, workspace, kernel, blocking::Blocking, oracle::Bool,
        allocator::TO.DefaultAllocator
    ) where {T}
    workspace === nothing &&
        return ContractWorkspace(T, kernel, blocking, oracle, allocator)
    return _reuse_workspace(T, workspace, kernel, blocking, oracle)
end

# Explicit allocator: sized once, never reused; the caller owns `release!`.
function _resolve_workspace(
        ::Type{T}, workspace, kernel, blocking::Blocking, oracle::Bool, allocator
    ) where {T}
    workspace === nothing || throw(
        ArgumentError(
            "plan_contract: `workspace` cannot be combined with a non-default " *
                "`allocator` ($(typeof(allocator))); an allocator-provided workspace is " *
                "sized once at construction and must not be resized or reused"
        )
    )
    return ContractWorkspace(T, kernel, blocking, oracle, allocator)
end

# The pool is keyed by `eltype(C)` alone, so also check the packed type
# (folds at compile time).
@inline function _reuse_workspace(
        ::Type{T}, ws::ContractWorkspace{T, Vector{R}}, kernel, blocking::Blocking,
        oracle::Bool
    ) where {T, R}
    R === realtype(kernel) || _throw_packed_eltype_mismatch(T, ws, kernel)
    return reserve!(ws, kernel, blocking, oracle)
end

@noinline function _throw_packed_eltype_mismatch(::Type{T}, ws, kernel) where {T}
    throw(
        ArgumentError(
            "plan_contract: cannot reuse a $(typeof(ws)) whose packed panels hold " *
                "$(eltype(ws.packed_a)) for a kernel packing $(realtype(kernel))"
        )
    )
end

@noinline function _reuse_workspace(
        ::Type{T}, ws::ContractWorkspace, kernel, blocking::Blocking, oracle::Bool
    ) where {T}
    throw(
        ArgumentError(
            "plan_contract: cannot reuse a $(typeof(ws)) for an eltype-$T contraction " *
                "on the default allocator; only a " *
                "ContractWorkspace{$T,Vector{$(realtype(kernel))}} -- storage element " *
                "type $T, packed panels of $(realtype(kernel)) -- is `reserve!`-able"
        )
    )
end
