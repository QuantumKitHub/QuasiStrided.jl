#!/bin/bash
# Full bench_to_suite.jl run (StridedBLAS vs QuasiStridedBackend on the upstream
# :contract and :network cases, Float64 and ComplexF64) plus its plots:
#
#   sbatch [--reservation=rocky8] --constraint=icelake benchmark/submit_bench_to_suite.sh
#
# The size sweeps overshoot on purpose: --max-flops (default 2e11) and
# --max-bytes prune each shape at the largest size that fits.
#SBATCH --job-name=qs-bench-to-suite
#SBATCH --partition=ccq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=benchmark/results/slurm-%j.out

source "${SLURM_SUBMIT_DIR:?run via sbatch}/benchmark/submit.sh" bench_to_suite.jl \
    --categories contract,network \
    --dtypes Float64,ComplexF64 \
    --synthetic-sizes 4,6,8,12,16,24,32,63,96,128,256 \
    --tccg-sizes 6,8,12,16,24 \
    --batched-sizes 4,8,16,32,64,128 \
    --mps-bonddims 32,64,100,128,256,512 \
    --ctmrg-chis 16,32,64,100 \
    --trg-chis 16,24,32,48 \
    --reps 21 --time-budget 5
"$JULIA" --project=benchmark benchmark/plot_bench_to_suite.jl "$OUTDIR/bench_to_suite.csv"
