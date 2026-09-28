# The MV = 4 real AVX-512 tile against its MV = 2 sibling as a function of the
# contracted depth `Qk` and of the C layout of the M composite: a leading
# unit-stride run of `r` rows in C, then a jump (C = [a, n, b], M = (a, b),
# `a` of extent `r`). Each kernel is named explicitly, so no swap and no
# run-length demotion applies: this is the raw store-layout x K-depth trade.
#
#   julia -t 1 --project=benchmark benchmark/probes/probe_mv4_store_sweep.jl \
#       --dtype Float64 --runs 16,24,32,4096 --ks 8,16,24,32,48,64,128,256 \
#       --qm 2048 --qn 512 --rounds 5
#
# `--runs` value >= `--qm` means a fully unit-stride M (C = [m, n]).
# Prints, per (r, Qk), the median GF/s of each shape over `--rounds`
# interleaved rounds (each a median of `--reps` calls) and the MV4/MV2 ratio.

using QuasiStrided
using QuasiStrided: plan_contract, execute!, _kernel_from_shape, mr
using StridedViews
using Statistics: median
using Printf
using Random

include(joinpath(@__DIR__, "..", "harness.jl"))

const T = parse_dtypes(argopt("dtype", "Float64"))[1]
const RUNS = parse.(Int, split(argopt("runs", "16,24,32,4096"), ','))
const KS = parse.(Int, split(argopt("ks", "8,16,24,32,48,64,128,256"), ','))
const QM = argopt("qm", 2048)
const QN = argopt("qn", 512)
const ROUNDS = argopt("rounds", 5)
const REPS = argopt("reps", 0)
const W = 64 ÷ sizeof(T)
const SHAPES = ((2W, 6, W), (4W, 6, W))

function setup(r, qk)
    rng = MersenneTwister(1)
    if r >= QM
        A = randn(rng, T, QM, qk); B = randn(rng, T, qk, QN); C = zeros(T, QM, QN)
        return (C, A, (1, -1), B, (-1, 2), (1, 2))
    end
    mb = QM ÷ r
    # labels: a = 1, b = 2, n = 3, k = -1
    A = randn(rng, T, r, mb, qk); B = randn(rng, T, qk, QN); C = zeros(T, r, QN, mb)
    return (C, A, (1, 2, -1), B, (-1, 3), (1, 3, 2))
end

println("T = $T  Qm = $QM  Qn = $QN  rounds = $ROUNDS  shapes = $SHAPES")
@printf("%6s %5s %10s %10s %8s %8s\n", "run", "Qk", "MV2 GF/s", "MV4 GF/s", "MV4/MV2", "spread")
for r in RUNS, qk in KS
    C, A, iA, B, iB, iC = setup(r, qk)
    plans = map(SHAPES) do s
        plan_contract(
            StridedView(C), StridedView(A), iA, StridedView(B), iB, iC;
            kernel = _kernel_from_shape(s, T), oracle = false
        )
    end
    flops = 2.0 * QM * QN * qk
    reps = REPS > 0 ? REPS : clamp(round(Int, 2.0e8 / flops), 5, 400)
    g = [Float64[] for _ in SHAPES]
    for _ in 1:ROUNDS, (i, p) in enumerate(plans)
        push!(g[i], flops / median_time_s(() -> execute!(p, one(T), zero(T)); reps) / 1.0e9)
    end
    ratios = g[2] ./ g[1]
    @printf(
        "%6d %5d %10.2f %10.2f %8.3f %8.3f\n", r, qk, median(g[1]), median(g[2]),
        median(ratios), maximum(ratios) - minimum(ratios)
    )
end
