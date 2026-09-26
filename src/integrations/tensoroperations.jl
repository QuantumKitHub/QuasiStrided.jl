# TensorOperations adapter: `QuasiStridedBackend`. This is the whole
# TensorOperations-facing surface; the engine (src/execution/execute.jl and
# everything it includes) knows nothing about TO.
#
# Import convention, not to be changed: every TensorOperations name is written
# `TO.<name>` (a bare `using TensorOperations` would collide on `scalartype`),
# except the TO-exported `Index2Tuple` and `linearize`.
import TensorOperations as TO
using TensorOperations: Index2Tuple, linearize
import TupleTools
using StridedViews: StridedView, isstrided

# ----------------------------------------------------------------------------
# Backend marker
# ----------------------------------------------------------------------------

"""
    QuasiStridedBackend()

TensorOperations backend routing a contraction to QuasiStrided's BLIS-style
macro-blocking engine ([`QuasiStrided.contract!`](@ref)):

    @tensor backend = QuasiStridedBackend() C[i, j] := A[i, k] * B[k, j]

Contraction only, and only for strided operands sharing a single element type
out of `Float32`/`Float64`/`ComplexF32`/`ComplexF64`; everything else (mixed
or unsupported eltypes -- mixed real/complex included, since promotion
belongs in TensorOperations' own `promote_contract` layer -- a non-strided
operand, an output aliased with an input, or a conjugated output view)
throws an `ArgumentError` from `TensorOperations.tensorcontract!`.

`TensorOperations.tensoradd!`/`TensorOperations.tensortrace!` fall back to
`TO.StridedNative()` (QuasiStrided has no analog of either): a timing taken
on those two operations under this backend measures `StridedNative`, not
this engine. This is the one exception to the "never falls back" rule below,
so that a `@tensor` network mixing a contraction with an add/trace step can
run wholesale under this backend.

TensorOperations' `conjA`/`conjB` flags are honored for complex eltypes: they
are forwarded to `plan_contract`, which folds each with the corresponding
operand's `StridedView.op` and applies the result in the packing pass. A
conjugated *output* `C` is rejected rather than supported.

It is not registered with `TensorOperations.select_backend`, and
`tensorcontract!` never falls back to another backend for any ineligible
input, so a timing taken on a *contraction* with this backend always
measures this engine.
"""
struct QuasiStridedBackend <: TO.AbstractBackend end

# ----------------------------------------------------------------------------
# Task-local workspace pooling
# ----------------------------------------------------------------------------

# One `task_local_storage` slot (not `threadid()`) holding a
# `Dict{DataType,ContractWorkspace}` keyed by scalar type.
const _QS_WORKSPACE_KEY = :quasistrided_contract_workspaces

@inline function _qs_workspace_pool()
    return get!(task_local_storage(), _QS_WORKSPACE_KEY) do
        return Dict{DataType, ContractWorkspace}()
    end::Dict{DataType, ContractWorkspace}
end

# Fetch (or lazily build) this task's persistent, `reserve!`-able
# `ContractWorkspace{T,Vector{real(T)}}` for scalar type `T`. Built once per
# `(task, T)` at `T`'s own default kernel/blocking; `plan_contract` grows it
# to whatever blocking the actual call needs via `reserve!`.
#
# The key is `eltype(C)` alone, with **no method component**: planar and 1m
# pack into the same `Vector{real(T)}` and `reserve!` is grow-only, so a
# workspace pooled under one complex method serves the other after at most a
# grow.
# Keying on the storage type rather than the packed type is what keeps a
# `Float64` and a `ComplexF64` contraction from colliding on one workspace.
#
# Both returns carry the same concrete type assertion. The build branch's
# type is derived from `_default_kernel(T)`, a Union over `T`'s kernel menu,
# so without the assertion the two branches join to the abstract
# `ContractWorkspace` -- which made the adapter's call into `_planned` a
# dynamic dispatch that boxed all three `StridedView`s, the label tuples and
# the executor on every call (measured: 3x64 + 3x32 + 48 B, ccqlin038, Julia
# 1.12.7, ComplexF64 8x8x8).
@inline function _qs_task_workspace(::Type{T}) where {T}
    pool = _qs_workspace_pool()
    ws = get(pool, T, nothing)
    ws === nothing || return ws::ContractWorkspace{T, Vector{real(T)}}
    return _qs_build_task_workspace!(pool, T)::ContractWorkspace{T, Vector{real(T)}}
end

@noinline function _qs_build_task_workspace!(pool::Dict{DataType, ContractWorkspace}, ::Type{T}) where {T}
    kernel = _default_kernel(T)
    new_ws = ContractWorkspace(T, kernel, default_blocking(kernel), false, TO.DefaultAllocator())
    pool[T] = new_ws
    return new_ws
end

# ----------------------------------------------------------------------------
# Label mapping
# ----------------------------------------------------------------------------

"""
    _qs_labels(pA::Index2Tuple, pB::Index2Tuple, pAB::Index2Tuple) -> (indA, indB, indC)

Translate TensorOperations' `pA`/`pB`/`pAB` index specification into
QuasiStrided's label convention (one `Int` label per axis, in axis order).

TensorOperations computes `C = β*C + α*permutedims(contract(opA(A), opB(B)), pAB)`,
contracting the axes `pA[2]` of `A` with `pB[1]` of `B` and permuting the
remaining `(pA[1]..., pB[2]...)` by `pAB`. The produced alphabet is `1:NoA` for
`A`'s open axes (in `pA[1]` order), `NoA+1 : NoA+NoB` for `B`'s open axes (in
`pB[2]` order), and `-1:-1:-Nk` for the contracted pairs (in the positionally
matched `pA[2]`/`pB[1]` order). `indC` is `linearize(pAB)` verbatim: the
intermediate tensor's slot `j` carries label `j` by construction, and
`permutedims`' convention is "output axis `c` takes input axis `perm[c]`".

Worked example (verified numerically):

```
pA  = ((3,1,4),(2,5))  ->  indA = ( 2, -1,  1,  3, -2)
pB  = ((3,1),(2,4))    ->  indB = (-2,  4, -1,  5)
pAB = ((4,2),(5,1,3))  ->  indC = ( 4,  2,  5,  1,  3)
```

Performs **no** validation of its own: `numout(pA) + numin(pB) == numind(pAB)`
and `numin(pA) == numout(pB)` are `TO.argcheck_tensorcontract`'s job, and the
adapter calls that first.
"""
function _qs_labels(pA::Index2Tuple, pB::Index2Tuple, pAB::Index2Tuple)
    NoA, Nk = TO.numout(pA), TO.numin(pA)
    qA = TupleTools.invperm(linearize(pA))
    qB = TupleTools.invperm(linearize(pB))
    indA = map(s -> s <= NoA ? s : -(s - NoA), qA)
    indB = map(s -> s <= Nk ? -s : NoA + (s - Nk), qB)
    return indA, indB, linearize(pAB)
end

# ----------------------------------------------------------------------------
# Eligibility and argument checking
# ----------------------------------------------------------------------------

# The two clauses of the eligibility predicate, split out only so the
# rejection message can name the one that failed. `_qs_eligible` below is the
# predicate itself; nothing else in this file re-states it.
const _QS_ELTYPES = (Float32, Float64, ComplexF32, ComplexF64)

_qs_eltype_ok(C, A, B) =
    eltype(A) === eltype(B) === eltype(C) && eltype(C) ∈ _QS_ELTYPES
_qs_strided_ok(C, A, B) = all(isstrided, (A, B, C))

"""
    _qs_eligible(C, A, B) -> Bool

Whether `QuasiStridedBackend` can serve `tensorcontract!(C, A, ..., B, ...)`:
a single shared element type out of `_QS_ELTYPES`
(`Float32`/`Float64`/`ComplexF32`/`ComplexF64`), and all three operands
strided. Mixed real/complex is deliberately *not* accepted. Ineligible inputs are rejected outright (see [`QuasiStridedBackend`](@ref)),
never routed to another backend.
"""
_qs_eligible(C, A, B) = _qs_eltype_ok(C, A, B) && _qs_strided_ok(C, A, B)

@noinline _qs_throw(msg::AbstractString) = throw(ArgumentError(msg))

# Step 1 of the required argument-checking order: hard-reject every ineligible
# input class before any TensorOperations check runs. `_qs_eligible` is the
# gate; the clause checks below it only exist to name the failure.
@noinline function _qs_check_eligible(f, C, A, B)
    _qs_eligible(C, A, B) && return nothing
    _qs_eltype_ok(C, A, B) || _qs_throw(
        "QuasiStridedBackend requires all tensors of $f to share a single " *
            "element type out of Float32, Float64, ComplexF32 and ComplexF64, got " *
            join(map(eltype, (C, A, B)), ", ")
    )
    _qs_strided_ok(C, A, B) || _qs_throw(
        "QuasiStridedBackend requires strided arrays for $f, got " *
            join(map(typeof, (C, A, B)), ", ")
    )
    return nothing
end

# ----------------------------------------------------------------------------
# Operations
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Conjugation
# ----------------------------------------------------------------------------
#
# `conjA`/`conjB` are NOT dropped: both are forwarded to `plan_contract`, which
# folds each with the corresponding view's `.op` through `_qs_isconj`
# (src/planning/conjugation.jl) and applies the result in the packing pass. Three facts about
# that split are load-bearing and are the reason these comments exist.
#
# (a) THE COMBINING RULE IS XOR, not `||`:
#
#         _qs_isconj(v::StridedView{T}, flag::Bool) where {T} =
#             (T <: Complex) && (flag ⊻ _op_conjugates(v.op))
#
#     TO's flag and `StridedView.op` are two *independent* requests to
#     conjugate the same data (`conjA` is applied on top of whatever the view
#     already carries), and `conj` is involutive -- so two conjugations cancel
#     and only the parity survives. `α`/`β` are never conjugated.
#
# (b) `_op_conjugates` IS TOTAL -- `identity`/`transpose` -> `false` (both are
#     elementwise identities on a `Number`), `conj`/`adjoint` -> `true`, plus an
#     `@noinline` throwing fallback. Deliberately NOT TensorOperations' TBLIS
#     extension's `(A.op === conj)` test: that is right for every `op`
#     `StridedViews` itself constructs but is not total, and
#     `StridedView(p, sz, st, off, adjoint)` is directly constructible and would
#     be classified as *unconjugated*, silently returning the wrong
#     contraction. Hard-rejecting an unknown `op` matches this backend's
#     "hard-reject, never fall back" rule, and the unreachable fallback costs
#     nothing.
#
# (c) A CONJUGATED OUTPUT `C` IS REJECTED, not supported. In-tree precedent:
#     TO's TBLIS extension does `isconj(SV(C), false) && throw_conj_output(f)`.
#     Supporting it would thread a conjugation flag as a type parameter through
#     `store_tile!` -> `execute_tile!` -> `_execute_micro_tile!` -> the nest,
#     doubling specialisations of the *innermost* code for a case TO's public
#     API cannot even express (there is no `conjC`), and would require
#     re-deriving the beta-applied-once argument under
#     `beta_eff = firstpanel ? betaT : one(T)`.
#
# The rejection lives in BOTH `plan_contract` and `_qs_prepare`, deliberately;
# see `_qs_prepare`. The real path is *structurally* immune: `StridedViews`
# defines `Base.conj(a::StridedView{<:Real}) = a`, and `_qs_isconj`
# short-circuits on `T <: Complex` regardless, so `conjA = true` on a real
# eltype cannot even create a new `execute!` specialisation.

# Shared prefix of both `tensorcontract!` methods below, in the required order:
#
#     eligibility -> argcheck -> dimcheck -> wrap -> aliasing
#         -> conjugated-C rejection
#
# The engine itself performs no aliasing check at all.
#
# The conjugated-`C` rejection is performed HERE as well as in
# `plan_contract`, and the duplication is deliberate. `plan_contract` owns it
# for direct `contract!`/`plan_contract` callers, who never run this prefix.
# But both methods below pass `workspace = _qs_task_workspace(...)` (or open
# an allocator checkpoint) as an *argument* to `plan_contract`, and Julia
# evaluates arguments first -- so deferring to the engine would acquire and
# possibly `reserve!`-grow a pooled workspace on behalf of a call that is about
# to be rejected, at a point the required order does not mention. Both sites call
# the identical `_qs_isconj(Cv, false)`, and each has its own pinning test.
#
# `conjA`/`conjB` deliberately do not flow through here: no step of this prefix
# consumes them, and their sole consumer `plan_contract` is called directly by
# each method below, so routing them through would widen this signature without
# moving any decision closer to the code that makes it.
@inline function _qs_prepare(C, A, pA, B, pB, pAB, α, β)
    _qs_check_eligible(TO.tensorcontract!, C, A, B)
    TO.argcheck_tensorcontract(C, A, pA, B, pB, pAB)
    TO.dimcheck_tensorcontract(C, A, pA, B, pB, pAB)

    Cv, Av, Bv = StridedView(C), StridedView(A), StridedView(B)

    # Aliasing is tested on the *wrapped* views: Base has no `dataids` for
    # `PermutedDimsArray`, but `StridedView` unwraps to the shared parent and
    # forwards `dataids` to it, so this catches cases raw arrays would miss.
    (Base.mightalias(Cv, Av) || Base.mightalias(Cv, Bv)) && _qs_throw(
        "output tensor must not be aliased with an input tensor in $(TO.tensorcontract!)"
    )

    # Same predicate and same reason as `plan_contract`'s; see the note above
    # this function for why the adapter does not simply defer to it.
    _qs_isconj(Cv, false) && _qs_throw(
        "output tensor of $(TO.tensorcontract!) must not be a conjugated view: " *
            "QuasiStrided writes through to the parent array and does not apply " *
            "`StridedView.op` on store, so a conjugated `C` would be silently wrong"
    )

    # Equivalent to `TO.standardize_scalartype` here. Discarding `Zero()`/
    # `One()`'s strong semantics is safe: every kernel branches on
    # `iszero(alpha)`/`iszero(beta)` rather than evaluating `β * C`.
    T = eltype(C)
    indA, indB, indC = _qs_labels(pA, pB, pAB)
    return Cv, Av, Bv, indA, indB, indC, convert(T, α), convert(T, β)
end

"""
    TensorOperations.tensorcontract!(C, A, pA, conjA, B, pB, conjB, pAB, α, β,
                                     ::QuasiStridedBackend, allocator)

Compute `C = β*C + α*permutedims(contract(opA(A), opB(B)), pAB)` with
QuasiStrided's macro-blocking engine, where `opA`/`opB` are `conj` or
`identity` per `conjA`/`conjB` folded with each operand's `StridedView.op`.

Throws an `ArgumentError` unless every operand is strided and they share a
single element type out of `Float32`/`Float64`/`ComplexF32`/`ComplexF64`,
unless `C` is unaliased with both `A` and `B`, and unless `C`'s view is
unconjugated; there is no fallback to another backend, so a timing taken with
this backend always measures this engine. See [`QuasiStridedBackend`](@ref) for
the full eligibility contract, and [`_qs_labels`](@ref) for the index
translation.

`conjA`/`conjB` are forwarded to `plan_contract`, which folds each with the
corresponding operand's `StridedView.op`; for a real element type the fold is
unconditionally `identity`, so the real path is unchanged by them.

Under `TensorOperations.DefaultAllocator` the buffers come from a persistent,
`reserve!`-grown task-local [`ContractWorkspace`](@ref); any other allocator
gets a workspace scoped to the single call. Dispatching on the allocator type,
rather than branching on its value, mirrors `_resolve_workspace` in
src/execution/workspace.jl and keeps both paths concretely typed.
"""
function TO.tensorcontract!(
        C::AbstractArray,
        A::AbstractArray, pA::Index2Tuple, conjA::Bool,
        B::AbstractArray, pB::Index2Tuple, conjB::Bool,
        pAB::Index2Tuple,
        α::Number, β::Number,
        backend::QuasiStridedBackend,
        allocator::TO.DefaultAllocator = TO.DefaultAllocator()
    )
    Cv, Av, Bv, indA, indB, indC, α′, β′ = _qs_prepare(C, A, pA, B, pB, pAB, α, β)
    # `plan_contract`'s body with `execute!` as the continuation (`_planned`,
    # src/planning/plan.jl): the plan is built and run inside the
    # `_plan_contract` barrier, where its type is concrete, so
    # `execute!(plan, ...)` is a static call and the plan never escapes this
    # frame -- `plan = plan_contract(...); execute!(plan, ...)` would instead
    # dispatch `execute!` at runtime on a Union-typed plan and box it (the
    # kernel choice is a runtime value; for a complex `T` its Union spans the
    # whole menu). Measured on ccqlin038 (Julia 1.12.7, 8x8x8, pooled
    # workspace): 1232 B and 1.1 us per Float64 call, 2032 B and 5.5 us per
    # ComplexF64 call before this and the planning-side changes it is paired
    # with (statically sized label tuples, the per-(profile, eltype) defaults
    # cache and the static kernel ladder); 0 B and ~0.5 / ~0.7 us after.
    # `_planned` `reserve!`s the pooled workspace itself. oracle=false: the
    # backend path never needs `execute_tilewise!`'s buffers.
    _planned(
        _QSExecute(α′, β′), Cv, Av, indA, Bv, indB, indC,
        nothing, conjA, conjB, nothing, nothing, nothing,
        _qs_task_workspace(eltype(C)), allocator, false
    )
    return C
end

# The continuation the method above hands `_planned`: `execute!` on the plan,
# with `alpha`/`beta` already converted to `eltype(C)`. A callable struct
# rather than a closure so the captured scalars are concretely typed fields.
struct _QSExecute{T}
    alpha::T
    beta::T
end
(e::_QSExecute)(plan::ContractPlan) = (execute!(plan, e.alpha, e.beta); nothing)

# NOT merged with the method above, although the two differ only in allocator
# handling: this one needs the plan back after `execute!` (to `release!` its
# allocator-owned workspace in `finally`), so it goes through `plan_contract`
# and cannot share the executor continuation. Two methods, deliberately.
function TO.tensorcontract!(
        C::AbstractArray,
        A::AbstractArray, pA::Index2Tuple, conjA::Bool,
        B::AbstractArray, pB::Index2Tuple, conjB::Bool,
        pAB::Index2Tuple,
        α::Number, β::Number,
        backend::QuasiStridedBackend, allocator
    )
    Cv, Av, Bv, indA, indB, indC, α′, β′ = _qs_prepare(C, A, pA, B, pB, pAB, α, β)
    # Explicit allocator: a workspace scoped to this call, never the pool,
    # bracketed with checkpoint!/reset! exactly as TO's own `blas_contract!`
    # brackets its temporaries so it composes with the surrounding network.
    checkpoint = TO.allocator_checkpoint!(allocator)
    plan = plan_contract(
        Cv, Av, indA, Bv, indB, indC;
        conjA = conjA, conjB = conjB,
        workspace = nothing, allocator = allocator, oracle = false
    )
    try
        execute!(plan, α′, β′)
    finally
        release!(plan.workspace, allocator)
        TO.allocator_reset!(allocator, checkpoint)
    end
    return C
end

# `tensoradd!`/`tensortrace!` fall back to `TO.StridedNative()`: QuasiStrided
# has no analog of either (no standalone add/permute step, and no trace/
# diagonal support at all -- `_classify_labels` in `src/planning/labels.jl` rejects
# repeated labels), so there is nothing of this engine's own to run. This
# fallback is confined to these two operations: `tensorcontract!` still
# hard-rejects every ineligible input and never falls back.

"""
    TensorOperations.tensoradd!(C, A, pA, conjA, α, β, ::QuasiStridedBackend, allocator)

Falls back to `TO.StridedNative()`: `QuasiStridedBackend` implements
contraction only ([`TensorOperations.tensorcontract!`](@ref)), has no analog
of a standalone add/permute step, and so has nothing of its own to run here.
This is the one exception to "never falls back" (see [`QuasiStridedBackend`](@ref));
a timing taken on a `tensoradd!` call under this backend measures
`StridedNative`, not this engine -- label results accordingly.
"""
function TO.tensoradd!(
        C,
        A, pA::Index2Tuple, conjA::Bool,
        α::Number, β::Number,
        backend::QuasiStridedBackend, allocator = TO.DefaultAllocator()
    )
    return TO.tensoradd!(C, A, pA, conjA, α, β, TO.StridedNative(), allocator)
end

"""
    TensorOperations.tensortrace!(C, A, p, q, conjA, α, β, ::QuasiStridedBackend, allocator)

Falls back to `TO.StridedNative()`: `QuasiStridedBackend` implements
contraction only ([`TensorOperations.tensorcontract!`](@ref)) and has no
trace/diagonal support at all (`_classify_labels` in `src/planning/labels.jl` rejects
repeated labels), so has nothing of its own to run here. This is the one
exception to "never falls back" (see [`QuasiStridedBackend`](@ref)); a timing
taken on a `tensortrace!` call under this backend measures `StridedNative`,
not this engine -- label results accordingly.
"""
function TO.tensortrace!(
        C,
        A, p::Index2Tuple, q::Index2Tuple, conjA::Bool,
        α::Number, β::Number,
        backend::QuasiStridedBackend, allocator = TO.DefaultAllocator()
    )
    return TO.tensortrace!(C, A, p, q, conjA, α, β, TO.StridedNative(), allocator)
end
