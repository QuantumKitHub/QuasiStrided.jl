# QuasiStrided.jl

[![CI](https://github.com/lkdvos/QuasiStrided.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/lkdvos/QuasiStrided.jl/actions/workflows/CI.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A native-Julia dense tensor contraction engine for `StridedView`s
(`StridedViews.jl`): grouped-axis / block-scatter indexing, packing, and
fixed-shape SIMD microkernels, in the style of block-scatter-matrix tensor
contraction (BSMTC, Matthews arXiv:1607.00291). It ships as a
[TensorOperations.jl](https://github.com/QuantumKitHub/TensorOperations.jl)
backend, `QuasiStridedBackend`.

**Status**: not registered; see [Status](#status).

## Install

```julia
using Pkg
Pkg.add(url="https://github.com/lkdvos/QuasiStrided.jl")
```

## Usage

Opt into the engine through TensorOperations' `@tensor`/`ncon`:

```julia
using TensorOperations, QuasiStrided

A, B = randn(3, 5, 2), randn(5, 4)

# C[a,n,b] = sum_k A[a,k,b] * B[k,n]
@tensor backend = QuasiStridedBackend() C[a, n, b] := A[a, k, b] * B[k, n]

# equivalently, via ncon
C2 = ncon([A, B], [[-1, 1, -3], [1, -2]]; backend = QuasiStridedBackend())
```

`QuasiStridedBackend` is not registered with `TensorOperations.select_backend`,
so loading QuasiStrided never changes the behaviour of `@tensor` code that does
not name it.

### Engine interface

The engine behind the backend can also be called directly:

```julia
using QuasiStrided, StridedViews
using QuasiStrided: contract!, plan_contract, execute!, SIMDKernel

# C[a,n,b] = alpha * sum_k A[a,k,b] * B[k,n] + beta * C[a,n,b]
A, B, C = randn(3, 5, 2), randn(5, 4), zeros(3, 4, 2)

# A label shared by A and B but absent from indC is contracted.
contract!(StridedView(C), 1.0,
          StridedView(A), (1, 2, 3),   # a, k, b
          StridedView(B), (2, 4),      # k, n
          0.0,
          (1, 4, 3))                   # a, n, b

# Reuse a plan and its workspace across calls, optionally naming the kernel:
plan = plan_contract(StridedView(C), StridedView(A), (1, 2, 3),
                     StridedView(B), (2, 4), (1, 4, 3);
                     kernel = SIMDKernel(Val(8), Val(6), Float64))
execute!(plan, 1.0, 0.0)
```

`plan_contract` also takes `workspace=`/`allocator=` keywords (e.g. for
Bumper-backed buffers); see its docstring.

## API

The only export is `QuasiStridedBackend`. The following names are `public`
(Julia >= 1.11) but unexported, to avoid collisions with TensorOperations
(both define e.g. `scalartype`):

| Kind | Names |
| --- | --- |
| Contraction | `contract!`, `plan_contract`, `execute!`, `ContractPlan`, `ContractWorkspace` |
| Blocking | `Blocking`, `default_blocking` |
| Kernels | `ScalarKernel`, `SIMDKernel`, `PlanarKernel`, `OneMKernel`, `FMAddSubKernel` |
| Hardware | `target_profile`, `cache_topology`, `TargetProfile`, `CacheLevel` |

Everything else is internal.

## Code map

`plan_contract` resolves labels into M/N/K axis groups and picks a kernel,
blocking and workspace; `execute!` runs a BLIS five-loop nest that packs `A`
and `B` into panels and drives a microkernel over register tiles of `C`.

| Folder | Contents |
| --- | --- |
| `src/hardware/` | ISA, vector width, register count and cache topology, detected once per process |
| `src/layout/` | `AxisGroup` (a grouped axis as zero-based offsets), block descriptors, tile axes |
| `src/packing/` | packed-panel formats and the `pack_a!`/`pack_b!` packers with contiguous fast paths |
| `src/microkernels/` | the kernel interface and the scalar, SIMD, planar, 1m and fmaddsub kernels |
| `src/planning/` | label classification, conjugation, kernel selection, cache blocking, `plan_contract` |
| `src/execution/` | the workspace, the five-loop nest, the specialised paths and the tile-by-tile oracle |
| `src/integrations/` | `QuasiStridedBackend` for TensorOperations.jl |

`test/` mirrors this layout.

## Status

Implemented:

- Grouped-axis (block-scatter) indexing over `StridedView`s, with packing into
  kernel-specific panel formats and contiguous fast paths.
- Microkernels: `ScalarKernel` (reference), `SIMDKernel` (default for real
  eltypes), `PlanarKernel` (split-complex, default for complex eltypes),
  `OneMKernel` (the 1m method) and `FMAddSubKernel` (interleaved complex, used
  for small-M complex contractions on AVX-512). Any of them can be named via
  `plan_contract(...; kernel = ...)`.
- Register shapes derived from the detected ISA, demoted when `M` is too short
  for a full tile or when a smaller tile keeps stores into `C` unit-stride.
- A BLIS five-loop (`NC`/`KC`/`MC`) nest reusing packed panels, with blocking
  from an analytical model of the detected cache sizes (fixed constants when
  they are undetected); plus automatically selected paths for dot-product,
  outer-product and read-B-in-place shapes.
- `Float32`/`Float64`/`ComplexF32`/`ComplexF64`, mixed freely (a complex input
  needs a complex output). The compute type is the promoted eltype, or the
  precision chosen with `accumulator = Float32`/`Float64`
  (`plan_contract(...; accumulator)`, `QuasiStridedBackend(; accumulator)`);
  the result is rounded to `eltype(C)` once. A real operand of a complex
  contraction is promoted at pack time. `conjA`/`conjB` and each operand's
  `StridedView.op` are applied during packing.
- Zero steady-state allocation on Julia >= 1.11 for `execute!` on a reused
  plan and for `tensorcontract!` through the backend (which pools workspaces
  per task). On Julia 1.10 `SIMDKernel`'s accumulator is not kept in
  registers, so calls allocate; results are unaffected.

Not implemented:

- Anything but contraction: `tensoradd!`/`tensortrace!` under this backend are
  forwarded to `StridedNative()`, so whole `@tensor` networks still run.
- Other eltypes, a complex input with a real output, non-strided operands, an
  output aliased with an input, and a conjugated output view: `tensorcontract!`
  throws an `ArgumentError` instead of falling back to another backend.
- Dedicated real-times-complex kernels.
- Being the default backend: `StridedBLAS()` is faster on most shapes.
- Batch axes, traces/diagonals, threading, GPU execution, the 3m complex
  method, autotuning.

`benchmark/bench_to_suite.jl` compares `QuasiStridedBackend()` with
`StridedBLAS()` over TensorOperations.jl's upstream benchmark suite.
