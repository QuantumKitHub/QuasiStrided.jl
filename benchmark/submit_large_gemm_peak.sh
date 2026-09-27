#!/bin/bash
# Slurm job validating the AVX-512 real register-tile change (`_rule_mv`,
# src/planning/kernel_selection.jl: MV = 4, i.e. (32,6,8) Float64 / (64,6,16)
# Float32, stepping down to the MV = 2 sibling for short M) on the other node
# classes. One job per microarchitecture, from the repo root:
#
#   sbatch --reservation=rocky8 --constraint=icelake benchmark/submit_large_gemm_peak.sh
#   sbatch --reservation=rocky8 --constraint=genoa   benchmark/submit_large_gemm_peak.sh
#   sbatch --reservation=rocky8 --constraint=rome    benchmark/submit_large_gemm_peak.sh
#
# What it runs, single core, single thread:
#
#   1. benchmark/bench_large_gemm_peak.jl -- the engine's default kernel vs
#      every real SIMDKernel menu shape vs OpenBLAS at large square GEMMs and
#      at the short-M step-down boundary, Float64 and Float32, interleaved
#      rounds; plus a kernel-only (L1-resident packed panels) ceiling per
#      shape and an (mc, kc) arm for the shipped shape. On rome (AVX2, 16
#      registers) the derived shape is unchanged, (8,6,4)/(16,6,8); the run
#      there is the regression check.
#   2. benchmark/bench_to_suite.jl on the large :contract/:network cases the
#      change targets (synthetic 24/32/63, mps D = 256/512), Float64 and
#      ComplexF64, so the before/after is on the same cases as the local
#      ccqlin038 run.
#
# The job also prints the CPU set it was confined to and BLAS's thread count
# right before timing: the 2026-09-24 icelake run (worker6199) reported
# StridedBLAS rates up to 8x a single Ice Lake core's peak (797 GF/s at
# dim63_2_1_2, K = 63) while the QuasiStrided rows of the same run were
# ordinary, which is only possible if BLAS was not confined to one core there.
#
# See https://wiki.flatironinstitute.org/SCC/Software/Slurm.
#SBATCH --job-name=qs-large-gemm-peak
#SBATCH --partition=ccq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=03:00:00
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

echo "=== confinement check ==="
echo "SLURM_JOB_NODELIST = ${SLURM_JOB_NODELIST:-?}  SLURM_CPUS_ON_NODE = ${SLURM_CPUS_ON_NODE:-?}  SLURM_CPUS_PER_TASK = ${SLURM_CPUS_PER_TASK:-?}"
echo "nproc (visible to this job) = $(nproc)"
taskset -cp $$ || true
lscpu | grep -i "model name\|^CPU(s)\|cache" || true

"$JULIA" --project=benchmark -e 'using Pkg; Pkg.instantiate()'

OUTDIR="benchmark/results/large-gemm-peak-${SLURM_JOB_ID}-$(hostname -s)"
mkdir -p "$OUTDIR"

"$JULIA" --project=benchmark benchmark/bench_large_gemm_peak.jl --outdir "$OUTDIR"

"$JULIA" --project=benchmark benchmark/bench_to_suite.jl \
    --categories contract,network \
    --sources synthetic --synthetic-sizes 24,32,63 \
    --topics mps --mps-bonddims 256,512 \
    --dtypes Float64,ComplexF64 \
    --reps 11 --time-budget 5 \
    --outdir "$OUTDIR/to_suite"

echo "=== node check ==="
lscpu | grep -i "model name\|cache" || true
