#!/bin/bash
# Same-node A/B of two git revisions on one benchmark script, from the repo root:
#
#   sbatch [--reservation=rocky8] [--constraint=icelake|genoa|rome] [--time=..] \
#       benchmark/submit_ab.sh <rev_a> <rev_b> <script.jl> [script args...]
#
# e.g. `sbatch --constraint=genoa benchmark/submit_ab.sh main HEAD bench_degenerate.jl`.
# Each revision runs from its own temporary worktree with its own copy of the
# script, alternately a, b, a, b, ... for $QS_AB_ROUNDS rounds (default 2), so
# slow drift on the node hits both alike. Outputs go to
# benchmark/results/<script>-ab-<jobid>-<host>/<a|b>-r<round>/. See
# https://wiki.flatironinstitute.org/SCC/Software/Slurm.
#SBATCH --job-name=qs-bench-ab
#SBATCH --partition=ccq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=benchmark/results/slurm-%j.out

set -euo pipefail
# Slurm runs a spooled copy of this script; SLURM_SUBMIT_DIR is the repo root.
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set -- run this script via sbatch, not directly}"
USAGE="usage: sbatch benchmark/submit_ab.sh <rev_a> <rev_b> <script.jl> [args...]"
REV_A=${1:?$USAGE}
REV_B=${2:?$USAGE}
SCRIPT=${3:?$USAGE}
shift 3
ROUNDS=${QS_AB_ROUNDS:-2}
SHA_A=$(git rev-parse --verify "$REV_A^{commit}")
SHA_B=$(git rev-parse --verify "$REV_B^{commit}")

# Same julia lookup as submit.sh.
module load modules/2.5-beta1 2>/dev/null || true
module load julia/1.12.6 2>/dev/null || true
MODJULIA=$(command -v julia 2>/dev/null || true)
case "$MODJULIA" in /mnt/sw/*) ;; *) MODJULIA="" ;; esac # only a module-provided one
JULIA=${QS_JULIA:-${MODJULIA:-$(ls -d "$HOME"/.julia/juliaup/julia-1.12*/bin/julia 2>/dev/null | tail -1)}}
[ -x "${JULIA:-}" ] || { echo "no julia found; set QS_JULIA" >&2; exit 127; }
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

echo "julia = $JULIA  node = ${SLURM_JOB_NODELIST:-?}  nproc = $(nproc)"
taskset -cp $$ || true
lscpu | grep -i "model name\|cache" || true

WORK=$(mktemp -d "${TMPDIR:-/tmp}/qs-ab-${SLURM_JOB_ID}-XXXX")
cleanup() {
    git worktree remove --force "$WORK/a" 2>/dev/null || true
    git worktree remove --force "$WORK/b" 2>/dev/null || true
    rm -rf "$WORK"
}
trap cleanup EXIT

OUTDIR="benchmark/results/$(basename "$SCRIPT" .jl)-ab-${SLURM_JOB_ID}-$(hostname -s)"
mkdir -p "$OUTDIR"
OUTDIR=$(cd "$OUTDIR" && pwd)
printf 'a %s %s\nb %s %s\n' "$REV_A" "$SHA_A" "$REV_B" "$SHA_B" | tee "$OUTDIR/revisions.txt"

for side in a b; do
    sha=$SHA_A
    [ "$side" = b ] && sha=$SHA_B
    git worktree add --detach "$WORK/$side" "$sha"
    # The submitting checkout's resolved versions, when it has them, so both
    # sides differ only in QuasiStrided.
    [ -f benchmark/Manifest.toml ] && cp benchmark/Manifest.toml "$WORK/$side/benchmark/"
    "$JULIA" --project="$WORK/$side/benchmark" -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
done

for round in $(seq 1 "$ROUNDS"); do
    for side in a b; do
        dir="$OUTDIR/$side-r$round"
        mkdir -p "$dir"
        echo "=== round $round: $(grep "^$side " "$OUTDIR/revisions.txt")"
        "$JULIA" --project="$WORK/$side/benchmark" "$WORK/$side/benchmark/$SCRIPT" --outdir "$dir" "$@" |
            tee "$dir/stdout.txt"
    done
done
