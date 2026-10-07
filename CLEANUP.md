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
| 13 | The plan | `planning/plan.jl` (+ `test_plan_contract.jl`, `test_per_call_overhead.jl`) | done |
| 14 | Five-loop nest | `execution/macrokernel.jl` (now `nest.jl`), `execute.jl`, `c_panel.jl` | done (ramp flags pending) |
| 15 | Alternative paths | `unpackedb.jl` (oracle gone in chunk 12) | done |
| 16 | Degenerate paths | `dot.jl`, `outer.jl` | done |
| 17 | TensorOperations backend | `integrations/tensoroperations.jl` | done |
| 18 | Test infrastructure | `runtests.jl`, `helpers.jl`, `forced_isa_runner.jl`, `quality/` | done |
| 19 | Benchmarks (optional) | `benchmark/` | done (plots under review) |

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

### D3. Dispatch barriers (chunk 13, applied: 308c07b)

One barrier, at plan time. Blocking (including line packing and the C panel)
and the execution path are resolved before the kernel barrier, from the kernel
type and shape; the barrier crosses once with `Val{Kern}()` and the path, and
the path is a `ContractPlan` field/type parameter, so `execute!` dispatches on
it statically. Deletes the hint family, `_strip_storage`/`_with_storage` and
the workspace's `SlotCache`. The TensorOperations route keeps its continuation
(one crossing per call); a reused plan has none. Accepted: a named-kernel
`plan_contract` is no longer inferable. The dot/outer/unpacked-B mode switches
become an internal `plan_contract` keyword instead of global `Ref`s. Also: a
keyword copy method for `ContractPlan`, blocking resolution as its own
function, underscores dropped, `barrier.jl` → `paths.jl`. Tests stay where they
are until chunk 18. As applied: named kernels cross as their (singleton)
instance; `pack_split` takes the pack format instead of a `SliverSpec` (whose
`MR` is runtime before the barrier); the dot-path fit is checked at plan time
from `workspace_sizes(K, …)`. Results and chosen paths bitwise identical to
before on 172 cases; per-call floor, TTFX and nest specialisations unchanged
within noise.

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

### D17. Five-loop nest (chunk 14, applied: 5c08bdf; ramp-flag benchmark pending)

Behaviour-preserving first: `EmptyPath`/`ScalePath` for empty M/N and K = 0
(decided at plan time; `execute!` keeps only the `alpha == 0` check); the
nest reads its parameters from the plan; one `pack_block!` per operand
(per-sliver or by lines); ramp decisions from the path's static flags; the C
panel path loops over N blocks itself (copy in, nest on the panel, copy
out), so the nest has no `target`/hooks; `execute.jl` split into
`execute.jl`, `nest.jl` (absorbs `macrokernel.jl`) and `c_panel.jl`.
Then, measured separately (Slurm A/B): drop the six `NestPath` ramp flags in
favour of the runtime `axis_of` choice; if they don't pay off, `NestPath`
shrinks to `{UNPACKED_B, SPLIT}` and the 512-entry table goes, and callable
paths get reconsidered.
As applied: `nest!(plan, alphaT, betaT, path, n_range)`; `pack_block!` lives in
`nest.jl` (needs the plan and group buffers); N's ramp under unpacked B stays
a runtime test of B's map. 408 equivalence cases bitwise identical; the extra
nests of degenerate plans are gone. Ramp-flag A/B: `benchmark/bench_ramp_flags.jl`
(flags cleared also disables the closed-form ramp descriptors, so it bounds
the cost of dropping them). Workstation, indicative: large cases equal, 24³
and 64³ 20–50% slower without flags.
Slurm round 1 (jobs 7183225–7, Icelake/Genoa/Rome; Icelake and Rome ran on a
tree with chunk-15 edits in progress, ratios are within-process): off/on
1.07–1.35 at 24³, up to 1.10 at 64³, 1.00 at 512³, ccsd(t) up to 1.07. But
off is *faster* in three cases: C64 12×512×512 on Icelake (0.52), F64
12×512×512 (0.75) and 64³ (0.91) on Genoa, all unpacked-B. Next: explain
that anomaly, then a round with a runtime-closed-form-ramp variant.
Cause found: in the unpacked-B tile loop with both C flags static, LLVM's SLP
vectoriser vectorises the per-column B addresses and recomputes them per K
step, costing a register: the ComplexF64 kernel spills 12 accumulators per
step. On Genoa codegen the runtime-stride K loop is also unrolled 8×, spilling
GPRs. Fix: dense B loads via per-column pointers with one shared K index, and
no unrolling of the unpacked-B K loop (ea97e6d; −20% on C64 12×512×512 locally;
Slurm A/B 7184013–5 confirmed: C64 small M 0.51 on Icelake, F64 small M /
64³ / 24³ 0.73 / 0.83 / 0.84 on Genoa, Rome neutral.
With the fix, flags-cleared costs 7–35% at 24³ and 5–10% at 64³. Round 2
(A/B 7184157–9, `ec3d30a` vs experiment `3b2e48e` on branch
`exp/runtime-axis-types`): closed-form descriptors with runtime axis types
match the static flags within noise (±4%, from the identical flags-cleared
column) except F64 partial-ramp 1.07 and ccsd(t) 1.05 on Icelake.
Decision: `NestPath` shrinks to `{UNPACKED_B, SPLIT}`; axis types are a
runtime `axis_of` choice; per-group ramp detection runs at runtime once per
call, keeping the closed-form descriptors; the 512-entry table goes. Then
reconsider callable paths. Applied after chunk 16.
Applied as b77152b and reverted (dde6d28): with the ramp decision at runtime,
dense plans also compile the scatter branches of the micro-tile, store and
pack code (execute_tile! 43 → 160, store_tile! 32 → 116 specialisations),
TTFX +40–90%, suite 22 → 32 min, no runtime gain. Final: the six static
flags stay (they pay for themselves in compile time), so paths stay structs.
Precompilation would not change this: it covers only the workload's types.

### D18. Unpacked B (chunk 15, applied: 337dc4a)

`UnpackedBView` is a B source next to `PackedPanel` (`packing/unpacked_b.jl`);
one tile loop in `nest.jl` takes its B sliver from either, by the path's
`UNPACKED_B`; the rule moves to `plan.jl`; eligibility is a kernel trait
(`reads_b_by_element`) defined beside each kernel. `unpackedb.jl` goes. The
`M <= 256` cutoff stays until measured (see possible improvements).

### D19. Degenerate paths (chunk 16, applied: 030d4d0)

`DotPath{MATB}` dispatched statically, with a `dot_operands(plan, Val(MATB))`
picking matrix/vector sides; the workspace is sized per path (dot: its vector
buffer, no panels; outer: no panels), deleting `dot_fits`; applicability
rules move to `plan.jl`; `_is_conj` → `op_conjugates`, `_buffer_range` →
`extrema(view(...))`, one `vector_lanes(profile, R)`. The β cases become
VectorInterface's `Zero()`/`One()` (direct dependency): one `axpby` defined by
dispatch for the vector/scalar stores, `axpby_at!`, the outer path and the
panel load; β stays a runtime number through the nest and `static_beta`
converts it only at the existing branch points (no 3× specialisation).
As applied: helpers take a β case (applying `static_beta` inside them cost
0.4 s of complex TTFX); the panel load keeps `iszero(beta)`; two named
complex 1×2 kernels at M = 1 now take the dot path (agree to rounding). Dot
plans allocate 8.5 KB instead of 150 KB; specialisation counts and TTFX
unchanged.

### D20. TensorOperations backend (chunk 17, applied: e2bf2ec)

No continuation: α/β become ordinary arguments of `planned` and fields of the
request (`nothing` = build the plan only); after the barrier `build_plan`
returns the plan, or executes and releases it. `Execute` and `f` go;
`contract!` and `tensorcontract!` pass α/β (one dynamic crossing each).
Conjugated-output check only in `planned`; the aliasing check moves into
`planned` so `plan_contract`/`contract!` get it too. Label translation stays
ours (TO's `contract_labels` builds `Vector{Char}`s, allocating and
non-inferable); `tensoradd!`/`tensortrace!` stay forwarded to StridedNative.
As applied: `planned`'s options are keywords (free); `contract!` 530 → 455 ns
at 2×2. The aliasing check (`Base.mightalias` on StridedViews) rejects any
output sharing a parent with A or B, even disjoint slices: deferred to
https://github.com/lkdvos/QuasiStrided.jl/issues/18; `build_plan`
specialises separately for plan-only and executing requests (200 → 304 on a
mixed workload, nothing below it grows).

### D21. Test infrastructure (chunk 18, applied)

Data: the suite is ~98% compilation (10m22s outside `Pkg.test`; 19–22 min
under `Pkg.test`'s `--check-bounds=yes`, which stays the default; document
`julia_args = ["--check-bounds=auto"]` for a fast local run). Pass 1:
ParallelTestRunner.jl (each file in its own process/module); one helper file
per folder plus shared fixtures in `test/helpers.jl`, no test file depending
on another; test files mirror `src/` (`test_per_call_overhead.jl` split by
concern, label-order tests to `test_labels.jl`, panel tests to
`test_c_panel.jl`); Aqua ambiguities for our package only. Pass 2: an audit
of what is tested more than once (low-level tests subsumed by high-level ones
or vice versa) and of compile-heavy loops, then agreed trims, checked against
the set of kernel/path/store instantiations the suite compiles.
Pass 1 applied (5920edb, be8b025): `Pkg.test` 21m53s → 3m44s (30 workers),
2m36s with `--check-bounds=auto`; 32163 tests kept (+1 Aqua). Audit (report
in the session scratchpad): line coverage saturated; low-level tests check
what end-to-end tests cannot (bitwise layouts, NaN in C at β = 0,
allocations, errors); the cost is extra compiled combinations, spread over
~15 testsets. Pass 2 applies the audit's trims 1–9 (≈100 s of per-file
compile, no src line lost); item 10 (single-precision kernel menus) is not
taken, so every shipped kernel shape stays compiled somewhere.
Pass 2 applied (6b2c50d, plus 1m pack coverage in the next commit): 32164 →
30888 tests, `Pkg.test` 218 s → 173 s; 0 src lines lost; every eltype ×
family and every path-flag value still reaches the nest. Slowest file is now
`test_unpackedb` (148 s; audit item 11, not taken). Pre-existing gap: eight
mixed-domain menu shapes (ComplexReal/RealComplex) are compiled by no test.

### D22. Benchmarks (chunk 19, applied: 0a1c746, b627c94)

Cleanup only (the outer/unpacked-B benchmark ideas stay recorded). Delete the
answered `bench_complex_blocking.jl`/`bench_ramp_flags.jl`; the probes become
the standard checks (`probe_ttfx.jl`, `probe_call_floor.jl`, new
`probe_specialisations.jl`), usable directly and through `submit_ab.sh`;
trim unused harness helpers; a uniform `--smoke` flag, every script
smoke-run once; a short `benchmark/README.md`. Suite plots: two overview
figures per dtype panel, throughput vs arithmetic intensity (colour = total
size, marker = backend) and time ratio vs total size (colour = intensity,
marker = category), with geomean/faster-count annotations and the most
extreme ratios labelled by case; tag filters (dtype, category, source/topic)
instead of per-group detail figures; violins dropped.

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

1. Chunk 19 (optional): benchmarks (`benchmark/`). Then the final passes: PrecompileTools workload (D4), delete this file before merging.
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

- Unpacked-B cutoff `M <= 256` is a constant; a cutoff from the reuse a
  packed B sliver would get (about `m_length / MR` M slivers) may be more
  principled. Benchmark before changing.

- The outer-product path is real only; a complex one would remove the
  asymmetry. Benchmark complex K = 1 contractions (nest vs a complex outer
  path) before deciding.

## Open items (to revisit in their chunk)



- Path types carry compile-time parameters (lane width, nest flags), which is
  why they are structs, not functions. Measure whether the six `NestPath` ramp
  flags pay off (regular vs scattered layouts, specialisation count); if they
  go, consider callable paths instead of `execute_path!` (chunk 14).
- `plan_request` is spelled out twice in `planned` (swapped and not).
- `workspace.jl`/`barrier.jl` live in `execution/` but are included in the
  planning section (chunk 12).
