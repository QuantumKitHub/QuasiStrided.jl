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
# Part 1: bench_to_suite.jl on everything the K-order cost model can touch --
# every synthetic layout at leg dims 16/32/63/96 (the contract_scrambled
# layouts are the only ones it reorders: dim16 0_2_3/3_2_3, dim32
# 2_2_2/1_3_1/0_2_3, dim63/96 2_2_2/1_3_1; benchmark/probes/
# probe_korder_changes.jl), TCCG at 8/16/24 (unchanged plans, regression
# guard; includes intensli_7_dim16) and the :network cases (TRG is the shape
# the model must NOT flip; the decision reads the core's L2 share, 512 KB on
# Rome). Compare against benchmark/results/to_suite-71120{88,89,90}-*.
#
# Part 2: benchmark/probes/probe_stage_breakdown.jl for intensli_7 and
# intensli_6 (Float64): pack B / pack A / kernel+store split. The suite has
# intensli_7_dim16 18x slower than intensli_6_dim16 on Rome (2.6x on Genoa,
# nothing on Intel), and intensli_7 at dims 12 and 24 fine on all three, so
# the variants discriminate between causes:
#   dim 12 / 24     non-power-of-two controls (dim 24 has 8x the pages of 16)
#   --pad 1         dim 16 extents, every A/C axis padded to 17 in the parent:
#                   no stride is a multiple of 4 KB (set aliasing). On
#                   ccqlin038 this is THE lever: intensli_7_dim24 pack A 52 ->
#                   6.3 ns/elem, 0.42 -> 0.08 s (dims 24 and 32 alias there,
#                   16 does not -- the trigger dims are machine-specific)
#   --nothp 1       A/C on 4 KB pages only (TLB reach; THP state printed below)
#   --msplit-pos 2  8 elements of A's unit-stride axis enumerated right after
#                   C's leading run: pack A reuses each line from L1 instead of
#                   after 4096 M rows (line locality / DRAM traffic)
#SBATCH --job-name=qs-scrambled-verify
#SBATCH --partition=ccq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=05:00:00
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
    --categories contract,network \
    --sources synthetic,tccg \
    --dtypes Float64,ComplexF64 \
    --synthetic-sizes 16,32,63,96 \
    --tccg-sizes 8,16,24 \
    --mps-bonddims 128,256 --ctmrg-chis 32,64 --trg-chis 16,32 \
    --reps 21 --time-budget 5 \
    --outdir "$OUTDIR"

echo "=== part 2: intensli stage breakdown (Float64) ==="
stage() {
    echo "--- $*"
    "$JULIA" --project=benchmark benchmark/probes/probe_stage_breakdown.jl \
        --dtype Float64 --reps 21 "$@" 2>&1 | grep -v "^WARNING"
}
{
    for case in intensli_7 intensli_6; do
        stage --case "$case" --dim 16
        stage --case "$case" --dim 16 --pad 1
        stage --case "$case" --dim 16 --nothp 1
        stage --case "$case" --dim 16 --pad 1 --nothp 1
        stage --case "$case" --dim 16 --msplit-pos 2 --msplit-li 8
        stage --case "$case" --dim 12
        stage --case "$case" --dim 24
    done
    stage --case intensli_7 --dim 32
    stage --case intensli_7 --dim 32 --pad 1
} | tee "$OUTDIR/intensli_stage_breakdown.txt"

echo "=== part 3: pack-A walk alone, aligned vs padded strides, sliver vs K-outer order ==="
for dim in 16 20 24 28 32; do
    for pad in 0 1; do
        "$JULIA" --project=benchmark benchmark/probes/probe_pack_order.jl \
            --case intensli_7 --dim "$dim" --pad "$pad" --reps 5 2>&1 | grep -v "^WARNING"
    done
done | tee "$OUTDIR/intensli_pack_order.txt"

echo "=== node check ==="
echo "SLURM_JOB_NODELIST = ${SLURM_JOB_NODELIST:-?}"
lscpu | grep -i "model name\|cache" || true
cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || true
