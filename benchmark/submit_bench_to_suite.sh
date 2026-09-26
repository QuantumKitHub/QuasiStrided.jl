#!/bin/bash
# Slurm job for a full benchmark/bench_to_suite.jl run: StridedBLAS vs
# QuasiStridedBackend over upstream TensorOperationsBenchmarks' :contract
# (synthetic layouts, TCCG, batched) and :network (mps, ctmrg, trg) cases,
# Float64 and ComplexF64 (the dtypes plot_bench_to_suite.jl plots). Single
# core, single node -- no internal parallelism, so plain sbatch (not disBatch).
# One job per microarchitecture, from the repo root:
#
#   sbatch --reservation=rocky8 --constraint=icelake benchmark/submit_bench_to_suite.sh
#   sbatch --reservation=rocky8 --constraint=genoa   benchmark/submit_bench_to_suite.sh
#   sbatch --reservation=rocky8 --constraint=rome    benchmark/submit_bench_to_suite.sh
#
# See https://wiki.flatironinstitute.org/SCC/Software/Slurm.
#
# Size sweeps go past what is affordable on purpose: --max-flops (2e11, ~10 s
# per call at the slowest rates seen) and --max-bytes (2 GiB) prune the
# largest combinations per shape, so every shape runs up to the largest size
# that fits instead of a hand-tuned ceiling per shape. --time-budget 5 cuts the
# rep count (to no fewer than 5) for the long cases.
#
# trg chi <= 48: StridedNative is not run (see bench_to_suite.jl), and chi > 48
# costs chi^6 per call, which --max-flops would drop anyway.
#SBATCH --job-name=qs-bench-to-suite
#SBATCH --partition=ccq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=benchmark/results/slurm-%j.out

set -euo pipefail
# Slurm runs a spooled copy of this script; SLURM_SUBMIT_DIR is the repo root.
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set -- run this script via sbatch, not directly}"

# See submit_blocking_model.sh: on Rocky 8 nodes (--reservation=rocky8) the
# module system's binaries need a newer glibc than the node has, so fall back
# to $QS_JULIA or a juliaup-installed 1.12 on the shared home.
module load modules/2.5-beta1 2>/dev/null || true
module load julia/1.12.6 2>/dev/null || true
MODJULIA=$(command -v julia 2>/dev/null || true)
case "$MODJULIA" in /mnt/sw/*) ;; *) MODJULIA="" ;; esac  # only a module-provided one
JULIA=${QS_JULIA:-${MODJULIA:-$(ls -d "$HOME"/.julia/juliaup/julia-1.12*/bin/julia 2>/dev/null | tail -1)}}
[ -x "${JULIA:-}" ] || { echo "no julia found; set QS_JULIA" >&2; exit 127; }
echo "julia = $JULIA"

export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"$JULIA" --project=benchmark -e 'using Pkg; Pkg.instantiate()'

# Own results dir per job: same-day reruns on one node would collide on
# benchmark/results/<host>-<date>.
OUTDIR="benchmark/results/to_suite-${SLURM_JOB_ID}-$(hostname -s)"

"$JULIA" --project=benchmark benchmark/bench_to_suite.jl \
    --categories contract,network \
    --dtypes Float64,ComplexF64 \
    --synthetic-sizes 4,6,8,12,16,24,32,63,96,128,256 \
    --tccg-sizes 6,8,12,16,24 \
    --batched-sizes 4,8,16,32,64,128 \
    --mps-bonddims 32,64,100,128,256,512 \
    --ctmrg-chis 16,32,64,100 \
    --trg-chis 16,24,32,48 \
    --reps 21 --time-budget 5 \
    --outdir "$OUTDIR"

"$JULIA" --project=benchmark benchmark/plot_bench_to_suite.jl "$OUTDIR/bench_to_suite.csv"

echo "=== node check ==="
echo "SLURM_JOB_NODELIST = ${SLURM_JOB_NODELIST:-?}"
lscpu | grep -i "model name\|cache" || true
