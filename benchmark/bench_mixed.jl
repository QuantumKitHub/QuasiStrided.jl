# Mixed-domain GEMM, single core: complex x real (CR) and real x complex (RC)
# through the named mixed-domain kernel, the default plan (which promotes the
# real operand to complex), and the real GEMM of the same real FMA count
# (2M x N x K for CR, M x 2N x K for RC).
#
#   julia --project=benchmark benchmark/bench_mixed.jl [--dtypes ComplexF64,ComplexF32]
#       [--sizes 512,2048] [--reps 5] [--outdir DIR]
#
# The mixed kernels' inner real tile is the host's default real shape, so
# `real/mixed` is the fraction of real efficiency reached.

include(joinpath(@__DIR__, "harness.jl"))

using QuasiStrided: ComplexRealKernel, RealComplexKernel, target_profile, _derived_shape

const RUN_DTYPES = parse_dtypes(argopt("dtypes", "ComplexF64,ComplexF32"))
const SIZES = parse_ints(argopt("sizes", "512,2048"))
const REPS = argopt("reps", 5)
const OUTDIR = outdir()
const CSV_PATH = joinpath(OUTDIR, "bench_mixed.csv")

function time_plan(C, A, B; kw...)
    plan = plan_contract(StridedView(C), StridedView(A), (1, 2), StridedView(B), (2, 3), (1, 3); kw...)
    return median_time_s(() -> execute!(plan, 1, 0); reps = REPS)
end

function run(io)
    print_env_header(stdout, "bench_mixed.jl")
    println(io, "dtype,case,n,kernel,t_mixed,t_promoted,t_real")
    rng = MersenneTwister(1)
    for T in RUN_DTYPES, n in SIZES
        R = real(T)
        MRr, NRr, W = _derived_shape(target_profile(), R)
        cases = (
            (
                "CR", ComplexRealKernel(Val(MRr ÷ 2), Val(NRr), T, Val(W)),
                randn(rng, T, n, n), randn(rng, R, n, n), (2n, n),
            ),
            (
                "RC", RealComplexKernel(Val(MRr), Val(NRr ÷ 2), T, Val(W)),
                randn(rng, R, n, n), randn(rng, T, n, n), (n, 2n),
            ),
        )
        for (case, kernel, A, B, (Mr, Nr)) in cases
            C = zeros(T, n, n)
            tm = time_plan(C, A, B; kernel)
            tp = time_plan(C, A, B)
            tr = time_plan(zeros(R, Mr, Nr), randn(rng, R, Mr, n), randn(rng, R, n, Nr))
            tag = "$(mr(kernel))x$(nr(kernel))/W$(lanewidth(kernel))"
            @printf(
                "%-10s %s %5d  %-10s mixed %8.4f s  promoted %8.4f s  real %8.4f s  promoted/mixed %.2f  real/mixed %.2f\n",
                T, case, n, tag, tm, tp, tr, tp / tm, tr / tm
            )
            println(io, join((T, case, n, tag, tm, tp, tr), ','))
            flush(stdout)
        end
    end
    return
end

open(run, CSV_PATH, "w")
println("wrote ", CSV_PATH)
