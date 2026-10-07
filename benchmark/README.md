# Benchmarks

Single-core scripts; each script's header has its options and outputs. All run
from the repo root with `julia --project=benchmark benchmark/<script> [args]`,
and every `bench_*.jl` takes `--smoke` (one rep on a few cases) to check that it runs.

| Script | Use it to |
|:-|:-|
| `bench_to_suite.jl` | compare QuasiStridedBackend with StridedBLAS on the upstream TensorOperations suite (`:contract`, `:network`); the overall picture after any change |
| `plot_bench_to_suite.jl` | plot a `bench_to_suite.csv`: throughput vs arithmetic intensity and the time ratio vs size, per dtype, with `--dtypes`/`--categories`/`--tags` filters |
| `bench_kernels.jl` | measure every register tile of the kernel menu against the default choice and OpenBLAS, after kernel or kernel-selection changes |
| `bench_blocking_model.jl` | check the cache-blocking model against a sweep of `(m_block, k_block, n_block)`, after blocking changes |
| `bench_mixed.jl` | compare the mixed real/complex kernels with promotion and the real GEMM of equal FMA count |
| `bench_degenerate.jl` | compare the dot, outer-product and unpacked-B paths with the five-loop nest and BLAS |
| `probes/probe_ttfx.jl` | time to first call (load, first plans, `@tensor`, `contract!`); one fresh process per run |
| `probes/probe_call_floor.jl` | per-call time and bytes on tiny problems (`tensorcontract!`, `contract!`, reused-plan `execute!`) |
| `probes/probe_specialisations.jl` | count compiled specialisations of the hot functions after a fixed workload |

On the cluster, one script per job, one job per microarchitecture:

    sbatch --reservation=rocky8 --constraint=<icelake|genoa|rome> benchmark/submit.sh <script> [args]

A same-node A/B of two revisions, alternating them over `$QS_AB_ROUNDS` rounds:

    sbatch --reservation=rocky8 --constraint=<icelake|genoa|rome> benchmark/submit_ab.sh <rev_a> <rev_b> <script> [args]

`benchmark/submit_bench_to_suite.sh` runs the full suite sweep plus its plots.
Results land in `benchmark/results/` (gitignored). Timings on a shared
workstation are indicative only.
