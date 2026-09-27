#!/bin/bash
# Slurm job verifying the degenerate/small-shape paths (unpacked B, dot, outer)
# on one node: benchmark/bench_degenerate.jl (each path vs the nest, same
# tree) and the bench_to_suite.jl :contract subset those paths target. Single
# core, single node. One job per microarchitecture, from the repo root -- the
# paths pick different lane widths per ISA (AVX2 on rome: W = 4 Float64):
#
#   sbatch --reservation=rocky8 --constraint=icelake benchmark/submit_degenerate.sh
#   sbatch --reservation=rocky8 --constraint=genoa   benchmark/submit_degenerate.sh
#   sbatch --reservation=rocky8 --constraint=rome    benchmark/submit_degenerate.sh
#
# See https://wiki.flatironinstitute.org/SCC/Software/Slurm.
#SBATCH --job-name=qs-degenerate
#SBATCH --partition=ccq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=benchmark/results/slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set -- run this script via sbatch, not directly}"

# Same julia resolution as submit_bench_to_suite.sh (module julia is broken
# on the rocky8 reservation; fall back to $QS_JULIA or juliaup 1.12).
module load modules/2.5-beta1 2>/dev/null || true
module load julia/1.12.6 2>/dev/null || true
MODJULIA=$(command -v julia 2>/dev/null || true)
case "$MODJULIA" in /mnt/sw/*) ;; *) MODJULIA="" ;; esac
JULIA=${QS_JULIA:-${MODJULIA:-$(ls -d "$HOME"/.julia/juliaup/julia-1.12*/bin/julia 2>/dev/null | tail -1)}}
[ -x "${JULIA:-}" ] || { echo "no julia found; set QS_JULIA" >&2; exit 127; }
echo "julia = $JULIA"

export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"$JULIA" --project=benchmark -e 'using Pkg; Pkg.instantiate()'

OUTDIR="benchmark/results/degenerate-${SLURM_JOB_ID}-$(hostname -s)"
mkdir -p "$OUTDIR"
"$JULIA" --project=benchmark benchmark/bench_degenerate.jl 31 | tee "$OUTDIR/bench_degenerate.txt"
"$JULIA" --project=benchmark benchmark/bench_to_suite.jl \
    --categories contract --sources synthetic,batched,tccg \
    --synthetic-sizes 6,8,12,16,32,63,128 --batched-sizes 16,32,64 --tccg-sizes 8,16 \
    --dtypes Float64,ComplexF64 --outdir "$OUTDIR"

echo "=== node check ==="
echo "SLURM_JOB_NODELIST = ${SLURM_JOB_NODELIST:-?}"
lscpu | grep -i "model name\|cache\|flags" | cut -c1-200 || true
