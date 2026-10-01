module QuasiStrided

using StridedViews: StridedView, offset
using Base.Checked: checked_abs, checked_add, checked_mul

# TensorOperations names are always qualified: a bare `using` collides on `scalartype`.
import TensorOperations as TO

# `plan_contract` resolves labels into M/N/K axis groups, picks a microkernel,
# cache blocking and workspace; `execute!` runs a BLIS five-loop nest that packs
# A/B into panels and drives the microkernel over register tiles of C. Files are
# included in type-dependency order.

# --- Hardware: ISA and cache detection ---
include("hardware/target.jl")

# --- Layout: zero-based strided/scattered addressing ---
include("layout/axis_group.jl")
include("layout/tiles.jl")

# --- Packing: packed-panel formats and the packers that fill them ---
include("packing/format.jl")
include("packing/panel.jl")
include("packing/pack.jl")
include("packing/pack_contiguous.jl")
include("packing/transposed.jl")

# --- Microkernels: accumulate over one packed K panel, store into C ---
include("microkernels/interface.jl")
include("microkernels/scalar.jl")
include("microkernels/simd.jl")
include("microkernels/planar.jl")
include("microkernels/onem.jl")
include("microkernels/fmaddsub.jl")
include("microkernels/mixed.jl")

# --- Planning: labels, conjugation, kernel and blocking choice, the plan ---
include("planning/labels.jl")
include("planning/conjugation.jl")
include("planning/kernel_selection.jl")
include("planning/blocking.jl")
include("planning/defaults.jl")
include("planning/pack_split.jl")
include("execution/workspace.jl")
include("execution/barrier.jl")
include("planning/plan.jl")

# --- Execution: the five-loop nest, the tile-by-tile oracle and the
# specialised paths ---
include("execution/macrokernel.jl")
include("execution/execute.jl")
include("execution/oracle.jl")
include("execution/unpackedb.jl")
include("execution/dot.jl")
include("execution/outer.jl")

# --- Integrations ---
include("integrations/tensoroperations.jl")

export QuasiStridedBackend

# Detect the hardware per process, not at precompile time: a cached .ji may be
# loaded on a different CPU.
function __init__()
    init_target!()
    return nothing
end

@static if VERSION >= v"1.11"
    eval(
        Expr(
            :public, :contract!, :plan_contract, :execute!, :ContractPlan,
            :ContractWorkspace, :Blocking, :default_blocking, :tile_size, :sliver_width,
            :ScalarKernel, :SIMDKernel, :PlanarKernel, :OneMKernel, :FMAddSubKernel,
            :ComplexRealKernel, :RealComplexKernel,
            :target_profile, :cache_topology,
            :TargetProfile, :CacheLevel
        )
    )
end

end # module QuasiStrided
