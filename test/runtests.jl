using Test
using Random
using QuasiStrided
# QuasiStrided.jl un-exports its internal/public-unexported tiers; this test
# suite still references all of them unqualified across its included files,
# so restore them here once rather than editing each file individually.
using QuasiStrided: AxisGroup, axis_length, offsets, fill_offsets!, BlockDescriptor,
    describe_block, block_descriptors!, normalize_group, KernelDescriptor, mr, nr,
    scalartype, packed_a_offset, packed_b_offset, packed_a_length, packed_b_length,
    AffineAxis, ScatterAxis, SourceTile, DestinationTile, axis_from_descriptor, nrows,
    ncols, axis_offset_range, checked_tile_storage_bounds, pack_a!, pack_b!,
    zero_accumulator, accumulate, scale_tile!, store_tile!, execute_tile!, lanewidth,
    avecs_per_column, contract!, Blocking, default_blocking, ScalarKernel, SIMDKernel
# plan_contract, execute! and ContractPlan are intentionally NOT restored here:
# helpers.jl introduces its own unqualified `plan_contract`/`execute!`/
# `ContractPlan` bindings via `const ... = QuasiStrided.<name>`, and
# execution/test_macro_blocking.jl deliberately writes `QuasiStrided.<name>` to
# avoid colliding with those (see its header comment).

# Every file below is included into this one scope, in pipeline-stage order;
# helper names must therefore be unique across files.
@testset "QuasiStrided.jl" begin
    include("helpers.jl")

    include("hardware/test_target.jl")

    include("layout/test_axis_group.jl")
    include("layout/test_stridedviews_axisgroup.jl")
    include("layout/test_tiles.jl")

    include("packing/test_kernel_descriptor.jl")
    include("packing/test_pack_real.jl")
    include("packing/test_pack_complex.jl")
    include("packing/test_pack_complex_contiguous.jl")

    include("microkernels/test_scalar_kernel.jl")
    include("microkernels/test_simd_kernel.jl")
    include("microkernels/test_planar_kernel.jl")
    include("microkernels/test_planar_store_fastpath.jl")
    include("microkernels/test_onem_kernel.jl")
    include("microkernels/test_fmaddsub_kernel.jl")
    # After test_planar_store_fastpath.jl: reuses its `ref_axpby` reference.
    include("microkernels/test_fmaddsub_store_fastpath.jl")

    include("planning/test_kernel_selection.jl")
    include("planning/test_plan_contract.jl")
    include("planning/test_per_call_overhead.jl")

    include("execution/test_manual_pipeline.jl")
    include("execution/test_execute.jl")
    include("execution/test_workspace.jl")
    include("execution/test_scalar_vs_simd.jl")
    include("execution/test_macro_blocking.jl")
    include("execution/test_direct.jl")
    include("execution/test_halfpack.jl")

    # Last among the functional tests: it does a bare `using TensorOperations`,
    # which exports its own `scalartype` and would otherwise conflict with the
    # `scalartype` restored from QuasiStrided above for any file included
    # after it.
    include("integrations/test_tensoroperations.jl")
    # Quality gate (Aqua): runs last since it audits the *whole* loaded
    # package/dependency graph after every other testset has already forced
    # everything to load, rather than gating functional tests on it.
    include("quality/test_aqua.jl")
end
