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
| 1 | Hardware | `hardware/target.jl` | |
| 2 | Axis groups | `layout/axis_group.jl`, `pair_group.jl` | |
| 3 | Tiles | `layout/tiles.jl` | |
| 4 | Packing formats | `packing/format.jl`, `panel.jl`, `transposed.jl` | |
| 5 | Packers | `packing/pack.jl`, `pack_contiguous.jl` | |
| 6 | Kernel interface | `microkernels/interface.jl`, `scalar.jl` | |
| 7 | SIMD kernels | `simd.jl`, `planar.jl` | |
| 8 | Complex/mixed kernels | `onem.jl`, `fmaddsub.jl`, `mixed.jl` | |
| 9 | Labels | `planning/labels.jl`, `conjugation.jl` | |
| 10 | Kernel selection | `planning/kernel_selection.jl` | |
| 11 | Blocking | `blocking.jl`, `defaults.jl`, `pack_split.jl` | |
| 12 | Workspace | `execution/workspace.jl`, `barrier.jl` | |
| 13 | The plan | `planning/plan.jl` (+ `test_plan_contract.jl`, `test_per_call_overhead.jl`) | |
| 14 | Five-loop nest | `execution/macrokernel.jl`, `execute.jl` | |
| 15 | Alternative paths | `oracle.jl`, `unpackedb.jl` | |
| 16 | Degenerate paths | `dot.jl`, `outer.jl` | |
| 17 | TensorOperations backend | `integrations/tensoroperations.jl` | |
| 18 | Test infrastructure | `runtests.jl`, `helpers.jl`, `forced_isa_runner.jl`, `quality/` | |
| 19 | Benchmarks (optional) | `benchmark/` | |

## Decisions

### D1. Descriptive M/N/K names (applied up front, codebase-wide)

Pattern `<m|n|k>_<level>_<what>`; M/N/K stay the organizing letters.

| Old | New |
|---|---|
| `Qm, Qn, Qk` | `m_length, n_length, k_length` |
| `mc, kc, nc` (`Blocking` fields, `plan_contract` kwargs) | `m_block, k_block, n_block` |
| `ic, pc, jc` | `m_block_start, k_block_start, n_block_start` |
| `mblock, kblock, nblock`, `kc_len` | `m_block_length, k_block_length, n_block_length` |
| `mr(k), nr(k)` | `tile_size(k) -> (m_tile, n_tile)` |
| `MRk, NRk` | `m_tile, n_tile` |
| `packed_a_per_k(k), packed_b_per_k(k)` | `sliver_widths(k) -> (a_sliver_width, b_sliver_width)` |
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

## Open items noted in the tour (to revisit in their chunk)

- 16-argument `ContractPlan(...)` spelled out five times, `_PlanRequest`
  twice more: one "copy with fields replaced" helper (chunk 13/14).
- `execute.jl` mixes public API, barrier/hint machinery, the C-panel path and
  the nest; `_c_panel_needed` lives there but is used by `plan.jl` (chunk 14).
- `_select_path` has 12 positional arguments, two optional (chunk 14).
- `oracle = true` default makes every user plan allocate test-path buffers;
  consider defaulting to `false` (chunk 13/15).
- `workspace.jl`/`barrier.jl` live in `execution/` but are included in the
  planning section (chunk 12).
