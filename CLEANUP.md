# Guided cleanup: decisions and status

Working notes for the `cleanup` branch; delete before merging.

## Process

Per chunk: explain purpose, reading order and design decisions; review
together; apply the agreed changes as one commit; run that chunk's tests (and
TTFX when codegen is touched).

## Chunks

| # | Chunk | Files | Status |
|---|---|---|---|
| 0 | Tour | `QuasiStrided.jl`, `plan.jl`, `execute.jl` call path | done |
| 1 | Hardware | `hardware/target.jl` | done |
| 2 | Axis groups | `layout/axis_group.jl`, `pair_group.jl` | done |
| 3 | Tiles | `layout/tiles.jl` | done |
| 4 | Packing formats | `packing/format.jl`, `panel.jl` (`transposed.jl` → chunk 11) | done |
| 5 | Packers | `packing/pack.jl`, `pack_contiguous.jl` | done |
| 6 | Kernel interface | `microkernels/interface.jl` (+ ScalarKernel) | done |
| 7 | Vector kernels | now `kernels.jl`, `vecops.jl`, `steps.jl`, `stores.jl` | done |
| 8 | Complex/mixed kernels | (merged into chunk 7's files) | done |
| 9 | Labels | `planning/labels.jl` (`conjugation.jl` → plan.jl) | done |
| 10 | Kernel selection | `planning/kernel_selection.jl` | done |
| 11 | Blocking | `blocking.jl`, line packing (now `packing/line_packing.jl`) | done |
| 12 | Workspace | `execution/workspace.jl`, `barrier.jl` | done |
| 13 | The plan | `planning/plan.jl` (+ `test_plan_contract.jl`, `test_per_call_overhead.jl`) | |
| 14 | Five-loop nest | `execution/macrokernel.jl`, `execute.jl` | |
| 15 | Alternative paths | `oracle.jl`, `unpackedb.jl` | |
| 16 | Degenerate paths | `dot.jl`, `outer.jl` | |
| 17 | TensorOperations backend | `integrations/tensoroperations.jl` | |
| 18 | Test infrastructure | `runtests.jl`, `helpers.jl`, `forced_isa_runner.jl`, `quality/` | |
| 19 | Benchmarks (optional) | `benchmark/` | |

## Decisions

### D1. Descriptive M/N/K names (applied: 564a740, 58b98e8)

Pattern `<m|n|k>_<level>_<what>`; M/N/K stay the organizing letters.

| Old | New |
|---|---|
| `Qm, Qn, Qk` | `m_length, n_length, k_length` |
| `mc, kc, nc` (`Blocking` fields, `plan_contract` kwargs) | `m_block, k_block, n_block` |
| `ic, pc, jc` | `m_block_start, k_block_start, n_block_start` |
| `mblock, kblock, nblock`, `kc_len` | `m_block_length, k_block_length, n_block_length` |
| `mr(k), nr(k)` | `tile_size(k) -> (m_tile, n_tile)`, `tile_size(k, i)` (like `size`) |
| `MRk, NRk` | `m_tile, n_tile` |
| `packed_a_per_k(k), packed_b_per_k(k)` | `sliver_width(k) -> (a_sliver_width, b_sliver_width)`, `sliver_width(k, i)` |
| `MRp, NRp` | `a_sliver_width, b_sliver_width` |
| `m_slivers, n_slivers` | `m_tiles, n_tiles` |
| `r, s`, `rfirst, sfirst` | `m_tile_index, n_tile_index`, `m_tile_start, n_tile_start` |
| `loop 5: jc` comments | `loop over N blocks` etc. |

Public names are renamed too (0.1, unregistered). Deferred: the `MR`/`NR`
type parameters and names inside the microkernels' generated code, reviewed
in chunks 6–8.

### D2. No leading underscores (per chunk)

Drop the `_` prefix from internal names as each chunk is reviewed; rename
where the underscore meant "internal helper of `foo`" (known clashes:
`_pack_a!`/`_pack_b!` vs `pack_a!`/`pack_b!`, `_plan_contract` vs
`plan_contract`). No submodules: nothing else clashes with Base, Core,
LinearAlgebra, SIMD or the package's own names. Explicit imports instead of
`using LinearAlgebra`, so later shadowing is an error (applied up front).

### D3. Dispatch barriers (chunks 12–14)

Keep barrier #1 (kernel choice). Remove barrier #2 (execution path) by
deciding the path at plan time and making it a `ContractPlan` type parameter;
this deletes `_path_hint`/`_continue`/`_execute_hinted!`/`_hint_holds`/
`_splits`/`_split_path`. To verify first: the path is fully fixed at plan
time, including dot-path workspace capacity under a reused grow-only
workspace. Measured: one crossing ~13 ns; a 2×2 `tensorcontract!` ~320 ns.

### D4. Precompile workload (later, separate)

Add a PrecompileTools workload for common cases (ranks ≤ 3, `Vector`
storage, the four eltypes) on top of the barriers, not instead of them.
Measure precompile time and pkgimage size first.

### D5. Hardware detection (chunk 1, applied)

Own implementation kept: CPUSummary/HostCPUFeatures detect at precompile
time, CpuId is x86-only without cache sharing, Hwloc needs a C library and
~1 s per process. The ISA is the CPUID probe's (name table dropped);
`CacheLevel` lost `ways`, `TargetProfile` lost `arch` and derives
`vector_bytes`/`nregisters` from `isa`; `core_bytes(profile, level)` replaces
three copies of the per-core share.

### D6. Axis groups (chunk 2, applied)

`normalize_group` and `offsets` leave `src` (`offsets` is a test helper);
`map_ramp_step` moves from dot.jl and `affine_ramp` is built on it;
`Base.setindex`/`prod`/`Base.Checked` replace hand-written versions;
`BlockDescriptor` moves to tiles.jl; `pair_group.jl` becomes the
`AxisGroup(labels, (ind1, v1), (ind2, v2))` constructor; `fill_offsets!`
drops its alias check. Kept: generic `P` (a batch label in A, B and C would
be a `P = 3` group) and unconditional range checks in `fill_offsets!`
(`@boundscheck`/`@propagate_inbounds` only help when inlined).

### D7. Tiles (chunk 3, applied)

Addresses stay zero-based (storage index = address + 1), as do `AxisGroup`
coordinates and block starts; coordinates within a tile, sliver or packed
panel are one-based. Tile axes are `AbstractVector{Int}`s of offsets:
`AffineAxis` (zero stride allowed, so not a `StepRange`) and `ScatterAxis`,
a borrowed pointer. The pointer is kept because the nest holds a
`Union{AffineAxis, ScatterAxis}`, which is only unboxed when both members are
`isbits`; a `view` there allocated per call, and splitting by hand needed
nested closures. One `Tile` type with `getindex`/`setindex!`/`size` replaces
`QSTile`/`SourceTile`/`DestinationTile`/`tile_load`/`tile_store!`.

### D8. Packing formats (chunk 4, applied)

Panels are only `PackedPanel` (the oracle and tests too; the `AbstractVector`
panel methods go); one `Descriptor` constructor plus a `RealDescriptor`
alias; one `packed_a_offset(d, i, p, plane = 0)` (and B);
`transposed.jl` merges into `panel.jl` and is reviewed with `pack_split.jl`.

### D9. Packers (chunk 5, applied)

A and B on the same footing: one `pack!(panel, tile, sliver_spec(kernel, i), transform)`
with lanes along the tile's rows; B is packed as `transpose(tile)`. A kernel
needing asymmetric packing gets it through its per-operand spec (format,
extent), or overloads `pack!` for its own spec. Fast paths are symmetric, so
real B with unit-stride columns takes the vector path (measured before
keeping). One `emit!` per format taking the values to store; padding passes
literal zeros. Vector-path predicates (`dense_lanes`, `is_unit_stride`,
`complex_fastpath_isa_eligible`) live in pack_contiguous.jl.

### D10. Kernel interface (chunk 6, applied)

`add_tile` (not an extension of `Base.accumulate`); `Microkernel` and
`KernelMethod` replace `DescriptorKernel`/`ComplexMethod`; each method
declares `pack_formats`, from which the reals per packed element follow. One
`execute_tile!` and one `pack!`, whose per-tile storage check is a
`@boundscheck` the nest skips with `@inbounds` (the macro block is span-checked;
under `--check-bounds=yes`, as in the tests, the checks run). `ScalarKernel`
uses a tuple accumulator (allocation-free) and branches on `beta` once per
tile. The contract lives in docstrings.

### D11. Vector kernels (chunk 7, decided; applied together with chunk 8)

One abstract `VectorKernel{MR, NR, T, W}` with generic `add_tile`,
`store_tile!` and scattered store; each kernel supplies its K step, its
full-block vector store and a generator-time accumulator-index function. All
kernel structs in one types file before the implementations, so layout
methods dispatch instead of `<:` tests (a generator only sees methods defined
before it). Kept as is: `@generated` (not `ntuple`); `muladd(-a, b, c)` (it
already compiles to one `vfnmadd`, no separate negation); the per-block `beta` tests in
the vector stores are left for now (LLVM does not unswitch them: about 4
compares per block). Hoisting is decided after chunk 8, leaning towards doing
it: each executed variant is smaller and branch-free (less code fetched per
store), at the cost of compile time and code size; measure both. `b_step_load*` hooks get descriptive
names.

### D12. Kernels = K step + accumulator layout (chunks 7–8, applied)

Three K steps (real, planar, fmaddsub) and three accumulator layouts (real,
split re/im, lane pairs). `accumulator_layout(kernel)` selects one generic
`store_tile!` (vector or scalar store, shared generated block/tail loop; each
layout supplies its full-block store and lane value); one generic `add_tile`
over `accumulate_step` with `k_steps` (1m: 2k). `inner(k)` is computed, not a
field (no `KI` parameter). One generic constructor without `W`; generic
`complex_method`/`lanewidth`. 1m gets the lane-pair vector store. The `beta`
hoist (cc78fd1) is kept: small K 19–27% faster (Float64) and 3–10%
(ComplexF64), large cases unchanged, vector stores ~35–40% larger, first call
Float64 3.1 → 3.7 s and ComplexF64 2.0 → 2.4 s, full suite ~18 → ~28 min.
Files, by concern: `interface.jl` (contract), `kernels.jl` (all kernel types
and their traits; `AccumulatorLayout`/`KernelMethod` are Holy traits),
`vecops.jl` (shared SIMD ops), `steps.jl` (K steps, `add_tile`), `stores.jl`
(layouts' stores).

### D13. Labels (chunk 9, applied)

`TupleTools.sortperm`/`getindices` (its merge sort is stable; relied on and
tested) replace the insertion sort; one K-order function; `l2_core_bytes`
moves to defaults.jl; `isconj`/`op_conjugates` move into plan.jl. The K-order
model, free-label order and swap rule are unchanged; labels.jl stays one
file.

### D14. Kernel selection (chunk 10, applied)

One `select_shape` pipeline (host shape → extent → small M → `fit_to_run`,
which merges the two C-run rules); NEON shapes pinned by a test instead of
override rows. The kernel type (unparameterised, e.g. `PlanarKernel`) replaces
`KernelMethod` as the shape-free identifier of a scheme; it crosses the plan
barrier as `Val{K}()` (a bare `Type` argument misses the dispatch fast path,
+350–430 ns per call).

### D15. Blocking, host defaults, line packing (chunk 11, applied)

`default_blocking(kernel, profile)` evaluates the analytical model with the
kernel's own tile and sliver bytes (complex and named kernels now get their
own optimum; real defaults unchanged); fixed rows only when caches are
undetected. No host-defaults cache: shape rules branch on `profile.isa`, and
`TargetProfile` derives `l2_share`/`l3_share`/`double_pumped` at
construction (computing them per plan cost 25–57 ns). The detected cache-line
size is used (`line_bytes`). The line-packing feature lives in
`packing/line_packing.jl`; `pack_split` takes the `SliverSpec`; the packer is
`pack_block_by_lines!`.

### D16. Workspace (chunk 12, applied)

No workspace pooling or reuse API: a plan builds its workspace through the
TensorOperations allocator; reusing a plan is the way to reuse buffers
(allocation-free). Measured cost under `DefaultAllocator`: per-call floor
~3x, 128³–256³ +13–36%; none under Bumper. If this matters, design an
explicit reuse mechanism rather than a cache. The tile-wise oracle is gone
(tests compare against the brute-force reference). Offsets and descriptors
are bundled per group (`ws.m`, `ws.n`, `ws.k`), not per operand: a group's
two maps are filled together and share a lifetime. Plans are single-task.

### Complex blocking benchmark (after chunk 11)

Same-node A/B, `bench_complex_blocking.jl`, jobs 7160147–9: the kernel's own
model is faster on Icelake for large sizes (0.58–0.92; the old scaled row was
rounded *up* to MR past the L2 budget), but 1–10% slower on Genoa/Rome and at
small M, following the smaller `k_block`. Rerun with a third "hybrid"
variant (jobs 7160543–5, commit d0100e6): hybrid/old 0.99–1.00 on all three
nodes (worst single case 1.02), new/old up to 1.10 at small M on AMD; the
Icelake 0.58 did not reproduce (noise). Decision: the hybrid — `k_block` from
the real default kernel's B sliver (`l1_k_block`), `m_block`/`n_block` from
the kernel's byte budget at that `k_block`, rounded down. Implemented in
blocking.jl (uncommitted until the full suite passes).

## Resume here

1. Chunk 13: `planning/plan.jl` (+ `test_plan_contract.jl`,
   `test_per_call_overhead.jl`), including D3 (execution path as a plan type
   parameter, removing barrier #2 and the hint machinery; kernel types cross
   the barrier as `Val`, `::Type{K}` everywhere after it) and the open items
   for chunks 13/14 below.
2. Workflow: present each chunk (purpose, reading order, design decisions,
   proposed fixes, questions), then hand the agreed changes to an Opus agent
   with the standard checks (Runic, full suite, per-call floor/allocations,
   TTFX, equivalence script where behaviour must not change); perf claims
   need Slurm runs (user submits; `--reservation=rocky8`), never poll Slurm.
   Only quote commit SHAs read from `git log`.


## Possible improvements

- Piecewise-affine block descriptions instead of block-sized offset buffers:
  https://github.com/lkdvos/QuasiStrided.jl/issues/13 (decide after chunks
  3, 5, 12, 14).
- One tile-axis type with a `regular` flag instead of a union of axis types:
  https://github.com/lkdvos/QuasiStrided.jl/issues/14.
- `PackedPanel` without a raw pointer:
  https://github.com/lkdvos/QuasiStrided.jl/issues/15.

- The A/B swap compares tile heights with `select_shape(...; k_length = ∞)`,
  so it never sees the small-K run demotion. Benchmark whether letting it see
  that step picks better orientations for small K.
- Analytical shape selection over a bounded search space instead of the menu:
  https://github.com/lkdvos/QuasiStrided.jl/issues/17.
- M and N are both ordered by their C strides. Only the M side matters for the
  vector store; for N the order mainly decides ramp detection and C locality,
  while B's pack (and the unpacked-B path) would prefer B's strides. Measure on
  suite layouts where B's and C's N strides disagree.

## Open items (to revisit in their chunk)

- One `pack_block!(panel, block, spec, transform, order)` for per-sliver and
  by-lines packing, so the nest makes one call per block (chunk 14).

- Internal constants still use `_UPPER` names; drop the underscores in one pass
  at the end (D2).

- Detected `l1d.line` is unused: `_K_LINE_BYTES = 64` in `labels.jl` (K-order
  cost model) and `pack_split.jl`; thread the detected line size through
  planning (chunks 9/11).
- 16-argument `ContractPlan(...)` spelled out five times, `_PlanRequest`
  twice more: one "copy with fields replaced" helper (chunk 13/14).
- `execute.jl` mixes public API, barrier/hint machinery, the C-panel path and
  the nest; `_c_panel_needed` lives there but is used by `plan.jl` (chunk 14).
- `_select_path` has 12 positional arguments, two optional (chunk 14).
- `oracle = true` default makes every user plan allocate test-path buffers;
  consider defaulting to `false` (chunk 13/15).
- `workspace.jl`/`barrier.jl` live in `execution/` but are included in the
  planning section (chunk 12).
