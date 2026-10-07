# TensorOperations backend. The engine itself knows nothing about TO.
import TensorOperations as TO
using TensorOperations: Index2Tuple, linearize
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
struct QuasiStridedBackend{A} <: TO.AbstractBackend
    function QuasiStridedBackend{A}() where {A}
        A in (nothing, Float32, Float64) ||
            throw(ArgumentError("QuasiStridedBackend: accumulator must be nothing, Float32 or Float64, got $A"))
        return new{A}()
    end
end

QuasiStridedBackend(; accumulator = nothing) = QuasiStridedBackend{accumulator}()

# TO's `pA`/`pB`/`pAB` -> one `Int` label per axis: `1:NoA` for A's open axes
# (in `pA[1]` order), `NoA+1:NoA+NoB` for B's open axes, `-1:-1:-Nk` for the
# contracted pairs; `indC` is then `linearize(pAB)`. For example
#     pA = ((3,1,4),(2,5)), pB = ((3,1),(2,4)), pAB = ((4,2),(5,1,3))
#     -> indA = (2,-1,1,3,-2), indB = (-2,4,-1,5), indC = (4,2,5,1,3).
# No validation: `TO.argcheck_tensorcontract` runs first.
function contraction_labels(pA::Index2Tuple, pB::Index2Tuple, pAB::Index2Tuple)
    NoA, Nk = TO.numout(pA), TO.numin(pA)
    qA = TupleTools.invperm(linearize(pA))
    qB = TupleTools.invperm(linearize(pB))
    indA = map(s -> s <= NoA ? s : -(s - NoA), qA)
    indB = map(s -> s <= Nk ? -s : NoA + (s - Nk), qB)
    return indA, indB, linearize(pAB)
end

@noinline function check_strided(C, A, B)
    all(isstrided, (A, B, C)) || throw(
        ArgumentError(
            "QuasiStridedBackend requires strided arrays for $(TO.tensorcontract!), got " *
                join(map(typeof, (C, A, B)), ", ")
        )
    )
    return nothing
end

# The aliasing check runs on the views in `planned`: Base has no `dataids` for
# a `PermutedDimsArray`, but a `StridedView` forwards to its parent.
@inline function prepare_contraction(C, A, pA, B, pB, pAB, α, β, accumulator)
    T = compute_type(eltype(A), eltype(B), eltype(C), accumulator)
    check_strided(C, A, B)
    TO.argcheck_tensorcontract(C, A, pA, B, pB, pAB)
    TO.dimcheck_tensorcontract(C, A, pA, B, pB, pAB)
    # `Zero()`/`One()` become numbers: `static_beta` recovers the β cases
    # at the branch points.
    indA, indB, indC = contraction_labels(pA, pB, pAB)
    return StridedView(C), StridedView(A), StridedView(B), indA, indB, indC, convert(T, α), convert(T, β)
end

# `planned` builds, runs and releases the plan behind the kernel dispatch
# barrier, so the plan is never boxed (as `execute!(plan_contract(...), ...)`
# would be). Bracketed with checkpoint/reset like TO's own `blas_contract!`.
function TO.tensorcontract!(
        C::AbstractArray,
        A::AbstractArray, pA::Index2Tuple, conjA::Bool,
        B::AbstractArray, pB::Index2Tuple, conjB::Bool,
        pAB::Index2Tuple,
        α::Number, β::Number,
        backend::QuasiStridedBackend{AC}, allocator = TO.DefaultAllocator()
    ) where {AC}
    Cv, Av, Bv, indA, indB, indC, α′, β′ = prepare_contraction(C, A, pA, B, pB, pAB, α, β, AC)
    checkpoint = TO.allocator_checkpoint!(allocator)
    planned(Cv, Av, indA, Bv, indB, indC, α′, β′; conjA, conjB, allocator, accumulator = AC)
    TO.allocator_reset!(allocator, checkpoint)
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
