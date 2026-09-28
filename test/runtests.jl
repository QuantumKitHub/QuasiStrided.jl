using Test
using Random
using QuasiStrided
# Internal names the test files use unqualified.
using QuasiStrided: AxisGroup, axis_length, offsets, fill_offsets!, BlockDescriptor,
    describe_block, block_descriptors!, normalize_group, KernelDescriptor, mr, nr,
    scalartype, packed_a_offset, packed_b_offset, packed_a_length, packed_b_length,
    AffineAxis, ScatterAxis, SourceTile, DestinationTile, axis_from_descriptor, nrows,
    ncols, axis_offset_range, checked_tile_storage_bounds, pack_a!, pack_b!,
    zero_accumulator, accumulate, scale_tile!, store_tile!, execute_tile!, lanewidth,
    avecs_per_column, contract!, Blocking, default_blocking, ScalarKernel, SIMDKernel
# plan_contract, execute! and ContractPlan are bound in helpers.jl instead.

# All files share one scope, so helper names must be unique across files.
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
    # All three use test_halfpack.jl's `_hp_ref` reference (and `_HPDenseMat`).
    include("execution/test_unpackedb.jl")
    include("execution/test_dot.jl")
    include("execution/test_outer.jl")

    # Last: its `using TensorOperations` makes `scalartype` ambiguous for later files.
    include("integrations/test_tensoroperations.jl")
    include("quality/test_aqua.jl")
end
