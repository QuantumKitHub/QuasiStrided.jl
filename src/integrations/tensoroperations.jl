# TensorOperations backend. The engine itself knows nothing about TO.
import TensorOperations as TO
using TensorOperations: Index2Tuple, linearize
import TupleTools
using StridedViews: StridedView, isstrided

"""
    QuasiStridedBackend(; accumulator = nothing)

TensorOperations backend running contractions on QuasiStrided's engine:

    @tensor backend = QuasiStridedBackend() C[i, j] := A[i, k] * B[k, j]

`tensorcontract!` accepts strided operands with element types out of
`Float32`/`Float64`/`ComplexF32`/`ComplexF64`, possibly mixed, and honours
`conjA`/`conjB`. The compute type and `accumulator` (`nothing`, `Float32` or
`Float64`) are as in [`plan_contract`](@ref). Any other input (other eltypes, a
complex input with a real output, a non-strided operand, an output aliased
with an input, a conjugated output view) throws an `ArgumentError`; it never
falls back to another backend. `tensoradd!` and `tensortrace!` are forwarded to
`TensorOperations.StridedNative()`.

The backend is not registered with `TensorOperations.select_backend`.
"""
struct QuasiStridedBackend{A} <: TO.AbstractBackend end

function QuasiStridedBackend(; accumulator = nothing)
    accumulator in (nothing, Float32, Float64) || throw(
        ArgumentError("QuasiStridedBackend: accumulator must be nothing, Float32 or Float64, got $accumulator")
    )
    return QuasiStridedBackend{accumulator}()
end

# Task-local pool of workspaces, keyed by the compute type only: all complex
# methods pack into `Vector{real(T)}` and `reserve!` is grow-only, so they can
# share one.
const _QS_WORKSPACE_KEY = :quasistrided_contract_workspaces

@inline function _qs_workspace_pool()
    return get!(task_local_storage(), _QS_WORKSPACE_KEY) do
        return Dict{DataType, ContractWorkspace}()
    end::Dict{DataType, ContractWorkspace}
end

@inline function _qs_task_workspace(::Type{T}) where {T}
    pool = _qs_workspace_pool()
    ws = get(pool, T, nothing)
    # Both assertions are needed: without them the branches join to the abstract
    # `ContractWorkspace`, and the call into `_planned` boxes its arguments.
    ws === nothing || return ws::ContractWorkspace{T, Vector{real(T)}, Vector{T}}
    return _qs_build_task_workspace!(pool, T)::ContractWorkspace{T, Vector{real(T)}, Vector{T}}
end

@noinline function _qs_build_task_workspace!(pool::Dict{DataType, ContractWorkspace}, ::Type{T}) where {T}
    kernel = _default_kernel(T)
    new_ws = ContractWorkspace(T, kernel, default_blocking(kernel), false, TO.DefaultAllocator())
    pool[T] = new_ws
    return new_ws
end

# TO's `pA`/`pB`/`pAB` -> one `Int` label per axis: `1:NoA` for A's open axes
# (in `pA[1]` order), `NoA+1:NoA+NoB` for B's open axes, `-1:-1:-Nk` for the
# contracted pairs; `indC` is then `linearize(pAB)`. For example
#     pA = ((3,1,4),(2,5)), pB = ((3,1),(2,4)), pAB = ((4,2),(5,1,3))
#     -> indA = (2,-1,1,3,-2), indB = (-2,4,-1,5), indC = (4,2,5,1,3).
# No validation: `TO.argcheck_tensorcontract` runs first.
function _qs_labels(pA::Index2Tuple, pB::Index2Tuple, pAB::Index2Tuple)
    NoA, Nk = TO.numout(pA), TO.numin(pA)
    qA = TupleTools.invperm(linearize(pA))
    qB = TupleTools.invperm(linearize(pB))
    indA = map(s -> s <= NoA ? s : -(s - NoA), qA)
    indB = map(s -> s <= Nk ? -s : NoA + (s - Nk), qB)
    return indA, indB, linearize(pAB)
end

@noinline _qs_throw(msg::AbstractString) = throw(ArgumentError(msg))

@noinline function _qs_check_strided(C, A, B)
    all(isstrided, (A, B, C)) || _qs_throw(
        "QuasiStridedBackend requires strided arrays for $(TO.tensorcontract!), got " *
            join(map(typeof, (C, A, B)), ", ")
    )
    return nothing
end

# Checks shared by both `tensorcontract!` methods. The eltype and conjugated-C
# checks precede `plan_contract`'s so that a rejected call never acquires or
# grows a pooled workspace (the workspace is an argument to `plan_contract`).
@inline function _qs_prepare(C, A, pA, B, pB, pAB, α, β, accumulator)
    T = _compute_type(eltype(A), eltype(B), eltype(C), accumulator)
    _qs_check_strided(C, A, B)
    TO.argcheck_tensorcontract(C, A, pA, B, pB, pAB)
    TO.dimcheck_tensorcontract(C, A, pA, B, pB, pAB)

    Cv, Av, Bv = StridedView(C), StridedView(A), StridedView(B)
    # On the wrapped views: Base has no `dataids` for `PermutedDimsArray`, but a
    # `StridedView` forwards to its parent.
    (Base.mightalias(Cv, Av) || Base.mightalias(Cv, Bv)) && _qs_throw(
        "output tensor must not be aliased with an input tensor in $(TO.tensorcontract!)"
    )
    _qs_isconj(Cv, false) && _qs_throw(
        "output tensor of $(TO.tensorcontract!) must not be a conjugated view: " *
            "QuasiStrided writes through to the parent array and does not apply " *
            "`StridedView.op` on store, so a conjugated `C` would be silently wrong"
    )

    indA, indB, indC = _qs_labels(pA, pB, pAB)
    # Dropping `Zero()`/`One()` is safe: the kernels branch on `iszero(alpha/beta)`.
    return Cv, Av, Bv, indA, indB, indC, convert(T, α), convert(T, β)
end

# Default allocator: pooled task-local workspace. `_planned` builds and runs the
# plan behind the kernel dispatch barrier, so the plan is never boxed (as
# `execute!(plan_contract(...), ...)` would be).
function TO.tensorcontract!(
        C::AbstractArray,
        A::AbstractArray, pA::Index2Tuple, conjA::Bool,
        B::AbstractArray, pB::Index2Tuple, conjB::Bool,
        pAB::Index2Tuple,
        α::Number, β::Number,
        backend::QuasiStridedBackend{AC},
        allocator::TO.DefaultAllocator = TO.DefaultAllocator()
    ) where {AC}
    Cv, Av, Bv, indA, indB, indC, α′, β′ = _qs_prepare(C, A, pA, B, pB, pAB, α, β, AC)
    _planned(
        _Execute(α′, β′), Cv, Av, indA, Bv, indB, indC,
        nothing, conjA, conjB, nothing, nothing, nothing,
        _qs_task_workspace(typeof(α′)), allocator, false, AC  # α′ has the compute type
    )
    return C
end

# Explicit allocator: a workspace scoped to this call, bracketed with
# checkpoint/reset like TO's own `blas_contract!`. This method needs the plan
# back to release it, hence `plan_contract` rather than `_planned`.
function TO.tensorcontract!(
        C::AbstractArray,
        A::AbstractArray, pA::Index2Tuple, conjA::Bool,
        B::AbstractArray, pB::Index2Tuple, conjB::Bool,
        pAB::Index2Tuple,
        α::Number, β::Number,
        backend::QuasiStridedBackend{AC}, allocator
    ) where {AC}
    Cv, Av, Bv, indA, indB, indC, α′, β′ = _qs_prepare(C, A, pA, B, pB, pAB, α, β, AC)
    checkpoint = TO.allocator_checkpoint!(allocator)
    plan = plan_contract(
        Cv, Av, indA, Bv, indB, indC;
        conjA = conjA, conjB = conjB,
        workspace = nothing, allocator = allocator, oracle = false, accumulator = AC
    )
    try
        execute!(plan, α′, β′)
    finally
        release!(plan.workspace, allocator)
        TO.allocator_reset!(allocator, checkpoint)
    end
    return C
end

# The engine has no add/permute or trace step; forward to StridedNative.
function TO.tensoradd!(
        C,
        A, pA::Index2Tuple, conjA::Bool,
        α::Number, β::Number,
        backend::QuasiStridedBackend, allocator = TO.DefaultAllocator()
    )
    return TO.tensoradd!(C, A, pA, conjA, α, β, TO.StridedNative(), allocator)
end

function TO.tensortrace!(
        C,
        A, p::Index2Tuple, q::Index2Tuple, conjA::Bool,
        α::Number, β::Number,
        backend::QuasiStridedBackend, allocator = TO.DefaultAllocator()
    )
    return TO.tensortrace!(C, A, p, q, conjA, α, β, TO.StridedNative(), allocator)
end
