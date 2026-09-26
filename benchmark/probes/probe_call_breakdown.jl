# Where the per-call floor of `tensorcontract!(..., QuasiStridedBackend())`
# goes on a tiny problem: the adapter's prefix (`_qs_prepare`), the pooled
# workspace lookup, planning alone (`_planned` with `identity` as the
# continuation, i.e. `plan_contract`'s body), `execute!` alone on a prebuilt
# plan, and the whole call. Internal API: tracks the current checkout only.
#
#   JULIA_NUM_THREADS=1 taskset -c 4-7 \
#       julia --project=benchmark benchmark/probes/probe_call_breakdown.jl

using TensorOperations, Statistics, Printf, StridedViews
using QuasiStrided
const TO = TensorOperations
const QS = QuasiStrided

const SAMPLES = 15
const CALLS = 2000

function timeit(f, args...)
    for _ in 1:10
        f(args...)
    end
    bytes = @allocated f(args...)
    ts = map(1:SAMPLES) do _
        t = time_ns()
        for _ in 1:CALLS
            f(args...)
        end
        (time_ns() - t) / CALLS
    end
    return median(ts), minimum(ts), bytes
end

full(C, A, pA, B, pB, pAB) =
    TO.tensorcontract!(C, A, pA, false, B, pB, false, pAB, true, false, QuasiStridedBackend())
prep(C, A, pA, B, pB, pAB) = QS._qs_prepare(C, A, pA, B, pB, pAB, true, false)
wsget(::AbstractArray{T}) where {T} = QS._qs_task_workspace(T)
function plan_only(C, A, pA, B, pB, pAB)
    Cv, Av, Bv, iA, iB, iC, _, _ = QS._qs_prepare(C, A, pA, B, pB, pAB, true, false)
    return QS._planned(
        _ -> nothing, Cv, Av, iA, Bv, iB, iC, nothing, false, false,
        nothing, nothing, nothing, QS._qs_task_workspace(eltype(C)), TO.DefaultAllocator(), false
    )
end
exec_only(plan, a, b) = QS.execute!(plan, a, b)

const CASES = (
    ("8x8x8 gemm", (8, 8), (8, 8), (8, 8), ((1,), (2,)), ((1,), (2,)), ((1, 2), ())),
    ("d4 outer", (4, 4), (4, 4), (4, 4, 4, 4), ((1, 2), ()), ((), (1, 2)), ((1, 2, 3, 4), ())),
)

for T in (Float64, ComplexF64), (name, sa, sb, sc, pA, pB, pAB) in CASES
    A, B, C = rand(T, sa), rand(T, sb), zeros(T, sc)
    Cv, Av, Bv, iA, iB, iC, _, _ = prep(C, A, pA, B, pB, pAB)
    plan = QS.plan_contract(Cv, Av, iA, Bv, iB, iC; workspace = QS._qs_task_workspace(T))
    for (what, f, args) in (
            ("full", full, (C, A, pA, B, pB, pAB)),
            ("prepare", prep, (C, A, pA, B, pB, pAB)),
            ("workspace", wsget, (C,)),
            ("prep+plan", plan_only, (C, A, pA, B, pB, pAB)),
            ("execute!", exec_only, (plan, one(T), zero(T))),
        )
        med, lo, b = timeit(f, args...)
        @printf("%-10s\t%-10s\t%-10s\t%7.1f ns\t(min %7.1f)\t%4d B\n", T, name, what, med, lo, b)
    end
end
