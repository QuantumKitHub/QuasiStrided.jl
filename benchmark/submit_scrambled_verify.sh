#!/bin/bash
# Slurm job verifying branch perf/scrambled-strides on one node class, and
# collecting the stage breakdown of the Rome-only intensli_7_dim16 slowdown.
# Single core, plain sbatch (see submit_bench_to_suite.sh). From the repo
# root of the branch's checkout:
#
#   sbatch --reservation=rocky8 --constraint=icelake benchmark/submit_scrambled_verify.sh
#   sbatch --reservation=rocky8 --constraint=genoa   benchmark/submit_scrambled_verify.sh
#   sbatch --reservation=rocky8 --constraint=rome    benchmark/submit_scrambled_verify.sh
#
# Part 1: bench_to_suite.jl on the subset the K-order rule touches -- every
# synthetic layout at leg dims 32 and 96 (dim{32,96}_1_3_1_contract_scrambled
# are the targets; gemm_ready/a_permuted/b_permuted/both_permuted plan the same
# K group as before and are the regression guard) plus TCCG at 8/16/24
# (single-K-label cases: unchanged plans, regression guard; includes
# intensli_7_dim16). Compare against benchmark/results/to_suite-71120{88,89,90}-*.
#
# Part 2: benchmark/probes/probe_stage_breakdown.jl for intensli_7 and
# intensli_6 at dim 16 and 8 (Float64): pack B / pack A / kernel+store split,
# as planned (as-is) and with two hand-made M enumeration orders that move one
# cache line of A's unit-stride axis forward (`--msplit-pos 2/1 --msplit-li 8`).
# On Cascade Lake all of these run at 3-4 ns per A element; on Rome the suite
# has intensli_7_dim16 18x slower than intensli_6_dim16, and this says which
# stage and whether the enumeration order is the lever.
#SBATCH --job-name=qs-scrambled-verify
#SBATCH --partition=ccq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=benchmark/results/slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set -- run this script via sbatch, not directly}"

module load modules/2.5-beta1 2>/dev/null || true
module load julia/1.12.6 2>/dev/null || true
MODJULIA=$(command -v julia 2>/dev/null || true)
case "$MODJULIA" in /mnt/sw/*) ;; *) MODJULIA="" ;; esac  # only a module-provided one
JULIA=${QS_JULIA:-${MODJULIA:-$(ls -d "$HOME"/.julia/juliaup/julia-1.12*/bin/julia 2>/dev/null | tail -1)}}
[ -x "${JULIA:-}" ] || { echo "no julia found; set QS_JULIA" >&2; exit 127; }
echo "julia = $JULIA"
echo "git = $(git rev-parse --short HEAD 2>/dev/null || echo '?') on $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"

export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"$JULIA" --project=benchmark -e 'using Pkg; Pkg.instantiate()'

OUTDIR="benchmark/results/scrambled-verify-${SLURM_JOB_ID}-$(hostname -s)"
mkdir -p "$OUTDIR"

echo "=== part 1: suite subset ==="
"$JULIA" --project=benchmark benchmark/bench_to_suite.jl \
    --categories contract \
    --sources synthetic,tccg \
    --dtypes Float64,ComplexF64 \
    --synthetic-sizes 32,96 \
    --tccg-sizes 8,16,24 \
    --reps 21 --time-budget 5 \
    --outdir "$OUTDIR"

echo "=== part 2: intensli stage breakdown (Float64) ==="
for dim in 16 8; do
    for case in intensli_7 intensli_6; do
        for split in "0 8" "2 8" "1 8"; do
            set -- $split
            echo "--- $case dim=$dim msplit-pos=$1 li=$2"
            "$JULIA" --project=benchmark benchmark/probes/probe_stage_breakdown.jl \
                --case "$case" --dim "$dim" --dtype Float64 --reps 21 \
                --msplit-pos "$1" --msplit-li "$2" 2>&1 | grep -v "^WARNING"
        done
    done
done | tee "$OUTDIR/intensli_stage_breakdown.txt"

echo "=== node check ==="
echo "SLURM_JOB_NODELIST = ${SLURM_JOB_NODELIST:-?}"
lscpu | grep -i "model name\|cache" || true
cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || true
