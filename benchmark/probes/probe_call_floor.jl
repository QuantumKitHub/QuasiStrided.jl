# Per-call floor of `tensorcontract!` on tiny problems: QuasiStridedBackend vs
# StridedBLAS, through TensorOperations' public entry point with the default
# allocator.
#
#   JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 taskset -c 4-7 \
#       julia --project=benchmark benchmark/probes/probe_call_floor.jl [label]
#
# Prints one line per (case, backend): median ns/call over SAMPLES samples of
# CALLS calls each, the [min, max] of those samples, and bytes/call. Whatever
# QuasiStrided the `--project` environment resolves is what is measured, so an
# A/B is two runs of this script from two checkouts, interleaved.

using TensorOperations, Statistics, Printf, LinearAlgebra
using QuasiStrided
const TO = TensorOperations

BLAS.set_num_threads(1)

const SAMPLES = 15
const CALLS = 2000

# (name, sizes of A, B, C, pA, pB, pAB)
const CASES = (
    ("8x8x8 gemm", (8, 8), (8, 8), (8, 8), ((1,), (2,)), ((1,), (2,)), ((1, 2), ())),
    ("16^3 gemm", (16, 16), (16, 16), (16, 16), ((1,), (2,)), ((1,), (2,)), ((1, 2), ())),
    # dim-4 outer product C[i,j,k,l] = A[i,j] * B[k,l]
    ("d4 outer", (4, 4), (4, 4), (4, 4, 4, 4), ((1, 2), ()), ((), (1, 2)), ((1, 2, 3, 4), ())),
)

function ncalls(C, A, pA, B, pB, pAB, backend, n)
    for _ in 1:n
        TO.tensorcontract!(C, A, pA, false, B, pB, false, pAB, true, false, backend)
    end
    return C
end

function measure(C, A, pA, B, pB, pAB, backend)
    ncalls(C, A, pA, B, pB, pAB, backend, 10)
    bytes = @allocated ncalls(C, A, pA, B, pB, pAB, backend, 100)
    ts = map(1:SAMPLES) do _
        t = time_ns()
        ncalls(C, A, pA, B, pB, pAB, backend, CALLS)
        (time_ns() - t) / CALLS
    end
    return median(ts), minimum(ts), maximum(ts), bytes / 100
end

label = isempty(ARGS) ? "" : ARGS[1]
for T in (Float64, ComplexF64), (name, sa, sb, sc, pA, pB, pAB) in CASES
    A, B, C = rand(T, sa), rand(T, sb), zeros(T, sc)
    for (bname, backend) in (("QS", QuasiStridedBackend()), ("BLAS", TO.StridedBLAS()))
        med, lo, hi, b = measure(C, A, pA, B, pB, pAB, backend)
        @printf(
            "%s\t%-10s\t%-10s\t%-4s\t%8.1f ns\t[%7.1f, %7.1f]\t%6.0f B\n",
            label, T, name, bname, med, lo, hi, b
        )
    end
end
