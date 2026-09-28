#!/bin/bash
# Slurm job checking the MV = 4 selection fixes (src/planning/kernel_selection.jl:
# `_store_shape`, the C-run step-down, and `_profile_mv`, AMD AVX-512 keeping
# MV = 2) on the node classes they were decided for. One job per
# microarchitecture, from the repo root:
#
#   sbatch --reservation=rocky8 --constraint=genoa   benchmark/submit_mv4_fix.sh
#   sbatch --reservation=rocky8 --constraint=icelake benchmark/submit_mv4_fix.sh
#
# What it runs, single core, single thread:
#
#   1. benchmark/probes/probe_mv4_plans.jl on the cases the MV = 4 rule
#      regressed (tccg ccsd_t_2/4, ao2mo_2 at dim 16/24; mps D64; the
#      genoa-only dim12/16_2_2_2_gemm_ready and trg chi16/24) in four
#      selections, three interleaved rounds each: `mv2` (main), `mv4,nostore`
#      (PR #10 head), `mv4` (this branch's C-run step-down without the AMD
#      exclusion) and `fix` (this branch). On icelake `mv4` and `fix` are the
#      same selection (a noise check). Per-plan dumps show which kernel each
#      pairwise contraction ran.
#   2. benchmark/bench_large_gemm_peak.jl: the engine at 2048^3 / 3969^3 per
#      named menu shape (MV = 2 vs MV = 4 on this node, whatever the default
#      is) and the kernel-only ceilings -- on genoa, the evidence for or
#      against `_profile_mv`.
#
# See https://wiki.flatironinstitute.org/SCC/Software/Slurm.
#SBATCH --job-name=qs-mv4-fix
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
echo "SLURM_JOB_NODELIST = ${SLURM_JOB_NODELIST:-?}  SLURM_CPUS_ON_NODE = ${SLURM_CPUS_ON_NODE:-?}"
echo "nproc (visible to this job) = $(nproc)"
taskset -cp $$ || true
lscpu | grep -i "model name\|cache" || true

"$JULIA" --project=benchmark -e 'using Pkg; Pkg.instantiate()'

OUTDIR="benchmark/results/mv4-fix-${SLURM_JOB_ID}-$(hostname -s)"
mkdir -p "$OUTDIR"

CASES=ccsd_t_2_dim16,ccsd_t_4_dim16,ao2mo_2_dim16,ccsd_t_2_dim24,ao2mo_2_dim24,mps_1site_D64,mps_2site_D64,dim12_2_2_2_gemm_ready,dim16_2_2_2_gemm_ready,trg_plaquette_chi16,trg_plaquette_chi24
for round in 1 2 3; do
    for mode in mv2 mv4,nostore mv4 fix; do
        echo "=== plans round $round mode $mode"
        "$JULIA" --project=benchmark benchmark/probes/probe_mv4_plans.jl \
            --mode "$mode" --cases "$CASES" --dtype Float64 \
            | tee -a "$OUTDIR/plans_Float64.txt"
    done
done

"$JULIA" --project=benchmark benchmark/bench_large_gemm_peak.jl --outdir "$OUTDIR"

echo "=== node check ==="
lscpu | grep -i "model name\|cache" || true
