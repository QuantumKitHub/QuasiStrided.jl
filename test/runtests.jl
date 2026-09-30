using Test
using Random
using QuasiStrided
# Internal names the test files use unqualified.
using QuasiStrided: AxisGroup, axis_length, offsets, fill_offsets!, BlockDescriptor,
    describe_block, block_descriptors!, normalize_group, KernelDescriptor, mr, nr,
    scalartype, packed_a_offset, packed_b_offset, packed_a_length, packed_b_length,
    AffineAxis, ScatterAxis, SourceTile, DestinationTile, axis_from_descriptor, nrows,
    ncols, axis_offset_range, checked_tile_storage_bounds, pack_a!, pack_b!,
    zero_accumulator, accumulate, store_tile!, execute_tile!, lanewidth, contract!,
    Blocking, default_blocking, ScalarKernel, SIMDKernel, TargetProfile, CacheLevel,
    target_profile, cache_topology, unknown_target, _detect_isa, _detect_target,
    _derived_shape, _fallback_shape, _shape_override, _kernel_for, _default_kernel,
    _fallback_blocking, kernel_shapes, _parse_size, _count_cpu_list, NR_DEFAULT,
    _rule_applies, _isa_nregisters, packed_a_per_k, packed_b_per_k, realtype,
    complex_method, RealMethod, PlanarMethod, OneMMethod, accumulator_planes, a_reals,
    b_reals, FMAddSubMethod, _modelled_blocking, _scale_blocking, _real_blocking_row,
    _planar_pressure, _pack_split, _NestPath
# plan_contract, execute! and ContractPlan are bound in helpers.jl instead.

# All files share one scope, so helper names must be unique across files.
@testset "QuasiStrided.jl" begin
    include("helpers.jl")

    include("hardware/test_target.jl")

    include("layout/test_axis_group.jl")
    include("layout/test_stridedviews_axisgroup.jl")
    include("layout/test_tiles.jl")

    include("packing/test_kernel_descriptor.jl")
    # Defines `ref_pack`/`pack_fixture`, which the two complex packing files reuse.
    include("packing/test_pack_real.jl")
    include("packing/test_pack_complex.jl")
    include("packing/test_pack_complex_contiguous.jl")

    # Defines the `mk_*` kernel-contract helpers the other microkernel files use.
    include("microkernels/test_scalar_kernel.jl")
    include("microkernels/test_simd_kernel.jl")
    include("microkernels/test_planar_kernel.jl")
    include("microkernels/test_planar_store_fastpath.jl")
    include("microkernels/test_onem_kernel.jl")
    include("microkernels/test_fmaddsub_kernel.jl")
    include("microkernels/test_mixed_kernel.jl")

    include("planning/test_kernel_selection.jl")
    include("planning/test_plan_contract.jl")
    include("planning/test_per_call_overhead.jl")

    include("execution/test_manual_pipeline.jl")
    include("execution/test_execute.jl")
    include("execution/test_workspace.jl")
    include("execution/test_scalar_vs_simd.jl")
    include("execution/test_macro_blocking.jl")
    # These use test_execute.jl's shared helpers and brute-force reference.
    include("execution/test_unpackedb.jl")
    include("execution/test_dot.jl")
    include("execution/test_outer.jl")
    include("execution/test_mixed.jl")

    # Last: its `using TensorOperations` makes `scalartype` ambiguous for later files.
    include("integrations/test_tensoroperations.jl")
    include("quality/test_aqua.jl")
end
