# The buffers a plan executes into, kept separate from `ContractPlan` so they
# can be reused across contractions. Typing discipline: `VT` is a `where`-bound
# parameter resolved at construction (never a `Union`- or
# `AbstractVector`-typed field), and the offset buffers stay concretely
# `Vector{Int}` -- acquired as non-temporaries, so only the packed panels
# route through the allocator.

# The workspace's typed slots for `_barrier_slot!` (src/execution/barrier.jl):
# one `Base.RefValue{P}` per payload type `P` ever sent across a dispatch
# barrier with this workspace, plus the most recently used one, so that the
# common case -- the same contraction type as last time -- is one type-tag
# compare. Mutable (and heap-allocated once, with the workspace) because the
# slots must outlive every call: a payload is written into its slot and read
# back on the far side of a dynamic call, which is what lets that call pass
# only already-boxed arguments and allocate nothing.
mutable struct _SlotCache
    last::Any
    const slots::IdDict{Any, Any}
end
_SlotCache() = _SlotCache(nothing, IdDict{Any, Any}())

"""
    ContractWorkspace{T,VT<:AbstractVector}

Every buffer [`execute!`](@ref) and `execute_tilewise!` need, sized once at
construction (or grown by [`reserve!`](@ref)) and never (re)allocated during
execution. Built by [`plan_contract`](@ref) and held by [`ContractPlan`](@ref);
pass an existing one back as `plan_contract(...; workspace = ws)` to reuse its
buffers across contractions of different shapes.

`T` is the **storage** element type (`eltype(C)`), which is what the backend's
workspace pool is keyed by; `VT` is the vector type of the two packed macro
panels, whose element type is `real(T)` -- the same type on the real path,
`Float64`/`Float32` for a complex contraction, since every packed buffer below
the kernel boundary holds reals. The bound on `VT` is therefore only
`AbstractVector`, with the `eltype(VT) === real(T)` invariant enforced by the
inner constructor; `VT` is still a `where`-bound parameter resolved to a
concrete vector type at construction.

`VT` is `Vector{real(T)}` on the default, GC-owned, [`reserve!`](@ref)-able
path; on an explicit-allocator path it is whatever that allocator returns, and
the workspace is then scoped to that one call -- [`release!`](@ref) it and drop
it, never pass it back as `workspace = ws`.

The `tw_*` buffers belong to `execute_tilewise!`, the independent oracle, and
are allocated only under `oracle = true`. The four register-tile-sized
`tile_*` offset buffers are always allocated: the beta-only pass of both
drivers uses them, and so does the oracle's tile loop.

A `mutable struct` with every field `const`: nothing is ever reassigned
(buffers grow in place, `reserve!`), but a workspace must be ONE heap object,
so that a plan holds a single pointer to it and a dispatch barrier
(src/execution/barrier.jl) passes or stores it without copying its twenty-one
fields: every copy of an immutable workspace -- into a plan, through a
barrier slot, out of the task pool's `Dict` -- was twenty-one GC-tracked
pointer copies. Measured through `tensorcontract!` (ccqlin038, Julia 1.12.7,
pooled workspace, 8x8x8): Float64 0.47 -> 0.38 us, ComplexF64 0.54 -> 0.45
us per call.

Field layout is an implementation detail, not part of the public interface.
"""
mutable struct ContractWorkspace{T, VT <: AbstractVector}
    # Macro-block-sized offset buffers: one fill_offsets! per jc/pc/ic block,
    # reused by every sliver inside it.
    const m_buf_A::Vector{Int}
    const m_buf_C::Vector{Int}
    const n_buf_B::Vector{Int}
    const n_buf_C::Vector{Int}
    const k_buf_A::Vector{Int}
    const k_buf_B::Vector{Int}

    # Per-sliver descriptors classified from the buffers above (3-arg
    # describe_block), reused across the pc/ic loops of a given jc/ic.
    const m_desc_A::Vector{BlockDescriptor}
    const m_desc_C::Vector{BlockDescriptor}
    const n_desc_B::Vector{BlockDescriptor}
    const n_desc_C::Vector{BlockDescriptor}

    # Packed macro-panel buffers: cld(mc,MRk)/cld(nc,NRk) slivers at kc-eff
    # depth. The ONLY allocator-routed temporaries in this struct.
    const packed_a::VT
    const packed_b::VT

    # One register tile's offsets (MR/NR-sized): the beta-only pass and the
    # oracle's tile loop.
    const tile_m_buf_A::Vector{Int}
    const tile_m_buf_C::Vector{Int}
    const tile_n_buf_B::Vector{Int}
    const tile_n_buf_C::Vector{Int}

    # execute_tilewise!'s own K-panel buffers (blocking.kc-sized). Deliberately
    # NOT shared with the macro buffers above: the oracle must have no packed
    # or K-offset state in common with the code it checks.
    const tw_k_buf_A::Vector{Int}
    const tw_k_buf_B::Vector{Int}
    const tw_packed_a::VT
    const tw_packed_b::VT

    # Typed payload slots for the dispatch barriers (`_barrier_slot!`,
    # src/execution/barrier.jl). Not a buffer: never sized, grown or released.
    const slots::_SlotCache

    # GUARDRAIL, the one invariant the loose `VT` bound needs: the packed
    # panels hold `real(T)`, NEVER `T`. Enforced here so a mis-paired (T, VT)
    # cannot be constructed at all, rather than failing later inside a packer.
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

# Element counts every buffer needs for `kernel` at the *effective* `blocking`
# (the rounded/clamped one `plan_contract` stores on the plan). Shared by the
# constructors and by `reserve!` so the two can never disagree.
#
# GUARDRAIL: complex-correct as written; do not rescale either count. The
# packed lengths come from `packed_a_length`/`packed_b_length`, which already
# return a count of REALS at the logical `kc` once the descriptor is complex,
# while the sliver counts use the LOGICAL register extents `mr`/`nr`, which is
# what `mc`/`nc` are expressed in. Do not mix them: `m_slivers` counts register
# tiles, `packed_a` counts reals.
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

# `Val(true)`: a genuine temporary, routed through `allocator` and never
# `resize!`d afterwards.
@inline function _alloc_temp(::Type{T}, n::Int, allocator) where {T}
    return TO.tensoralloc(Vector{T}, (n,), Val(true), allocator)
end

# `Val(false)`: a non-temporary, which every allocator serves as a plain,
# GC-owned `Vector{Int}`, as the offset buffers require. The assertion makes a hypothetical violation fail loudly here rather than
# silently inside `fill_offsets!`.
@inline function _alloc_offsets(n::Int, allocator)
    return TO.tensoralloc(Vector{Int}, (n,), Val(false), allocator)::Vector{Int}
end

# Descriptor arrays are not tensor buffers (`BlockDescriptor` has no
# `scalartype`), so they never route through `tensoralloc`. `undef` is safe:
# `_classify_slivers!` writes entry `s+1` before the same iteration reads it.
@inline _alloc_descriptors(n::Int) = Vector{BlockDescriptor}(undef, n)

# The 20-field layout, written out exactly once for both constructors below:
# `ints` allocates a `Vector{Int}` of a given length, while the packed panels
# arrive already allocated because their type is what fixes `VT`.
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
    ContractWorkspace(T, kernel, blocking::Blocking, oracle::Bool, allocator)
    ContractWorkspace(T, kernel, blocking::Blocking;
                      oracle = true, allocator = TensorOperations.DefaultAllocator())

Build a workspace for scalar type `T`, `kernel` and an already-effective
`blocking`. `oracle = false` skips the `execute_tilewise!`-only buffers
entirely (they are left empty), which is what the TensorOperations backend
path passes.

Under `DefaultAllocator` every buffer is an ordinary `Vector`, so the result is
a `ContractWorkspace{T,Vector{R}}`, `R = realtype(kernel) === real(T)`, that
[`reserve!`](@ref) may later grow; on the real path `R === T`. Under any other
allocator the packed panels are acquired once via
`TensorOperations.tensoralloc(..., Val(true), allocator)`, are never resized,
and must be handed back with [`release!`](@ref).

Buffers are `undef`-initialized, not zeroed: the packers (`src/packing/pack.jl`)
write every slot of a panel it is given, padding included, and the offset
buffers are fully rewritten by `fill_offsets!` before each block is read.
"""
function ContractWorkspace(
        ::Type{T}, kernel, blocking::Blocking, oracle::Bool, ::TO.DefaultAllocator
    ) where {T}
    s = _workspace_sizes(kernel, blocking)
    # Reals, not storage elements; the inner constructor re-checks against `T`.
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

    # Acquisition order matters for arena allocators: `release!` frees in the
    # exact reverse order.
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

"""
    reserve!(ws::ContractWorkspace{T,Vector{R}}, kernel, blocking::Blocking,
             oracle::Bool) -> ws

Grow `ws` in place so every buffer is large enough for `kernel` at the
effective `blocking`, and return it. Grow-only: it never shrinks a buffer,
never reallocates one that is already large enough, and never hands back a
`view`; every consumer is length-tolerant and addresses only the live region
of the *current* block. `oracle = false` leaves the `execute_tilewise!`-only
buffers alone. Defined only for the default, GC-owned path -- an
allocator-provided temporary must never be `resize!`d, so that path sizes its
workspace exactly once at construction instead.
"""
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

# Grow-only resize. `resize!` upward leaves the new region undefined, which is
# the same `undef` discipline the constructors use.
@inline function _grow!(v::Vector, n::Int)
    length(v) < n && resize!(v, n)
    return v
end

"""
    release!(ws::ContractWorkspace, allocator) -> nothing

Hand `ws`'s allocator-routed temporaries (`packed_a`/`packed_b`, plus the
oracle's `tw_packed_*` when present) back to `allocator` via
`TensorOperations.tensorfree!`, in the exact reverse of acquisition order so
LIFO arenas unwind correctly. The offset and descriptor buffers are GC-owned
and untouched; `ws` must not be used afterwards. Calling this unconditionally
is correct for every allocator -- it returns memory for `ManualAllocator` and
is a no-op for the ones that unwind via `allocator_reset!`/`@no_escape`
instead.
"""
function release!(ws::ContractWorkspace, allocator)
    TO.tensorfree!(ws.tw_packed_b, allocator)
    TO.tensorfree!(ws.tw_packed_a, allocator)
    TO.tensorfree!(ws.packed_b, allocator)
    TO.tensorfree!(ws.packed_a, allocator)
    return nothing
end

# Whether `ws` carries `execute_tilewise!`'s buffers. `packed_a_length(kernel,
# kc) == mr(kernel) * kc >= 1` for every legal blocking (`kc >= 1`), so an
# empty `tw_packed_a` means -- and only means -- the workspace was built with
# `oracle = false`.
@inline _has_oracle(ws::ContractWorkspace) = !isempty(ws.tw_packed_a)

# Build or reuse the plan's workspace. Dispatching on the allocator type (not
# an `isa` branch on a value) keeps both paths concretely typed and makes the
# unreachable one disappear at compile time. Default path: a plain, GC-owned
# workspace, reused via `reserve!` when one is handed in, which is
# the zero-steady-state-allocation fast path.
function _resolve_workspace(
        ::Type{T}, workspace, kernel, blocking::Blocking, oracle::Bool,
        allocator::TO.DefaultAllocator
    ) where {T}
    workspace === nothing &&
        return ContractWorkspace(T, kernel, blocking, oracle, allocator)
    return _reuse_workspace(T, workspace, kernel, blocking, oracle)
end

# Explicit-allocator path: sized exactly once from the effective blocking, no
# `reserve!`, no resizing. The caller owns `release!`.
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

# `R` is the PACKED element type, `real(T)` by `ContractWorkspace`'s own
# invariant. The extra guard stops a pooled workspace of one precision serving
# a plan of another: the pool is keyed by `eltype(C)` alone, so `T` matching is
# not enough once the packed type is a separate notion. Both sides are
# compile-time constants, so this folds away entirely.
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
