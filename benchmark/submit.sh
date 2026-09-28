#!/bin/bash
# Slurm job running one benchmark script on a single core, from the repo root:
#
#   sbatch [--reservation=rocky8] [--constraint=icelake|genoa|rome] [--time=..] \
#       benchmark/submit.sh <script.jl> [script args...]
#
# e.g. `sbatch --constraint=genoa benchmark/submit.sh bench_kernels.jl --dtypes Float64`.
# The script gets `--outdir benchmark/results/<script>-<jobid>-<host>`, so
# reruns never collide. Pick the node class with --constraint, one job per
# microarchitecture. See https://wiki.flatironinstitute.org/SCC/Software/Slurm.
#SBATCH --job-name=qs-bench
#SBATCH --partition=ccq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=benchmark/results/slurm-%j.out

set -euo pipefail
# Slurm runs a spooled copy of this script; SLURM_SUBMIT_DIR is the repo root.
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set -- run this script via sbatch, not directly}"
SCRIPT=${1:?usage: sbatch benchmark/submit.sh <script.jl> [args...]}
shift

# On Rocky 8 nodes (--reservation=rocky8) the module system's binaries need a
# newer glibc than the node has, so `module load` yields no julia; fall back to
# $QS_JULIA or a juliaup-installed 1.12 (official builds need only glibc 2.17).
module load modules/2.5-beta1 2>/dev/null || true
module load julia/1.12.6 2>/dev/null || true
MODJULIA=$(command -v julia 2>/dev/null || true)
case "$MODJULIA" in /mnt/sw/*) ;; *) MODJULIA="" ;; esac # only a module-provided one
JULIA=${QS_JULIA:-${MODJULIA:-$(ls -d "$HOME"/.julia/juliaup/julia-1.12*/bin/julia 2>/dev/null | tail -1)}}
[ -x "${JULIA:-}" ] || { echo "no julia found; set QS_JULIA" >&2; exit 127; }
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# BLAS escaping the one-core allocation shows up as impossible StridedBLAS
# rates, so record the CPU set the job is confined to.
echo "julia = $JULIA  node = ${SLURM_JOB_NODELIST:-?}  nproc = $(nproc)"
taskset -cp $$ || true
lscpu | grep -i "model name\|cache" || true

"$JULIA" --project=benchmark -e 'using Pkg; Pkg.instantiate()'
OUTDIR="benchmark/results/$(basename "$SCRIPT" .jl)-${SLURM_JOB_ID}-$(hostname -s)"
mkdir -p "$OUTDIR"
"$JULIA" --project=benchmark "benchmark/$SCRIPT" --outdir "$OUTDIR" "$@"
