# Conjugation: folding `conjA`/`conjB` with each view's `.op`. The engine never
# indexes through a `StridedView`, so an unfolded `.op` would be silently dropped.

# GUARDRAIL: a TOTAL table with a throwing fallback, not an `op === conj` test,
# which classifies a directly constructed `adjoint` view as unconjugated.
_op_conjugates(::typeof(identity)) = false
_op_conjugates(::typeof(conj)) = true
# Elementwise on a `Number`, these are identity/conj; the axes are already resolved.
_op_conjugates(::typeof(transpose)) = false
_op_conjugates(::typeof(adjoint)) = true
@noinline _op_conjugates(f) = throw(
    ArgumentError(
        "unsupported StridedView.op $f: QuasiStrided folds a view's `op` into the " *
            "packing transform and recognizes only identity/conj/transpose/adjoint"
    )
)

# GUARDRAIL: `⊻`, not `||`: the flag and the view's `op` are independent
# conjugations and `conj` is involutive. Always `false` for real `T`, so the
# real path never gets a `conj` specialization.
_qs_isconj(v::StridedView{T}, flag::Bool) where {T} =
    (T <: Complex) && (flag ⊻ _op_conjugates(v.op))
