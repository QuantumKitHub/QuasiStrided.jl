# Per-call floor on small problems:
#
#   JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 taskset -c 4-7 \
#       julia --project=benchmark benchmark/probes/probe_call_floor.jl [--label X]
#
# `tensorcontract!` through TensorOperations' public entry point with the
# default allocator, QuasiStridedBackend (QS) against StridedBLAS (BLAS); then
# QuasiStrided's own `contract!` (planning included) and `execute!` of a reused
# plan. Prints one line per (case, entry point): median ns/call over SAMPLES
# samples of CALLS calls each, the [min, max] of those samples, and bytes/call.
# Whatever QuasiStrided the `--project` environment resolves is what is
# measured, so an A/B is two runs of this script from two checkouts, interleaved.

using TensorOperations, StridedViews, Statistics, Printf, LinearAlgebra
using QuasiStrided
using QuasiStrided: plan_contract, execute!, contract!
const TO = TensorOperations

BLAS.set_num_threads(1)

const SAMPLES = 15
const CALLS = 2000

label = let i = findfirst(==("--label"), ARGS)
    i === nothing ? "" : ARGS[i + 1]
end

# (name, sizes of A, B, C, pA, pB, pAB)
const CASES = (
    ("8x8x8 gemm", (8, 8), (8, 8), (8, 8), ((1,), (2,)), ((1,), (2,)), ((1, 2), ())),
    ("16^3 gemm", (16, 16), (16, 16), (16, 16), ((1,), (2,)), ((1,), (2,)), ((1, 2), ())),
    # dim-4 outer product C[i,j,k,l] = A[i,j] * B[k,l]
    ("d4 outer", (4, 4), (4, 4), (4, 4, 4, 4), ((1, 2), ()), ((), (1, 2)), ((1, 2, 3, 4), ())),
)

function ncalls(f::F, n) where {F}
    for _ in 1:n
        f()
    end
    return
end

function measure(f)
    ncalls(f, 10)
    bytes = @allocated ncalls(f, 100)
    ts = map(1:SAMPLES) do _
        t = time_ns()
        ncalls(f, CALLS)
        (time_ns() - t) / CALLS
    end
    return median(ts), minimum(ts), maximum(ts), bytes / 100
end

function report(T, name, entry, f)
    med, lo, hi, b = measure(f)
    @printf(
        "%s\t%-10s\t%-10s\t%-9s\t%9.1f ns\t[%8.1f, %8.1f]\t%6.0f B\n",
        label, T, name, entry, med, lo, hi, b
    )
    return flush(stdout)
end

for T in (Float64, ComplexF64)
    for (name, sa, sb, sc, pA, pB, pAB) in CASES
        A, B, C = rand(T, sa), rand(T, sb), zeros(T, sc)
        for (bname, backend) in (("QS", QuasiStridedBackend()), ("BLAS", TO.StridedBLAS()))
            report(T, name, bname, () -> TO.tensorcontract!(C, A, pA, false, B, pB, false, pAB, true, false, backend))
        end
    end
    for n in (2, 64)
        A, B, C = StridedView(rand(T, n, n)), StridedView(rand(T, n, n)), StridedView(zeros(T, n, n))
        name = n == 2 ? "2x2 gemm" : "64^3 gemm"
        report(T, name, "contract!", () -> contract!(C, one(T), A, (1, 2), B, (2, 3), zero(T), (1, 3)))
        plan = plan_contract(C, A, (1, 2), B, (2, 3), (1, 3))
        report(T, name, "execute!", () -> execute!(plan, one(T), zero(T)))
    end
end
