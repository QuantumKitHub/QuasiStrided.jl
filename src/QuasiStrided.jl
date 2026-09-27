module QuasiStrided

using LinearAlgebra
using StridedViews: StridedView, offset

# Module-import convention, not to be changed: TensorOperations names are
# always qualified; a bare `using TensorOperations` would collide on
# `scalartype`.
import TensorOperations as TO

# A contraction C[indC] = alpha * A[indA] * B[indB] + beta * C runs through the
# stages below, in order. `plan_contract` resolves labels into M/N/K axis
# groups, picks a microkernel and cache blocking, and sizes a workspace;
# `execute!` runs a BLIS five-loop nest that packs A/B slivers into panels and
# drives the microkernel over register tiles of C.
#
# Include order follows type dependencies, which mostly coincide with the
# stage order; the exceptions are noted inline.

# --- Hardware: ISA and cache detection ---
include("hardware/target.jl")

# --- Layout: zero-based strided/scattered addressing ---
include("layout/axis_group.jl")
include("layout/pair_group.jl")
include("layout/tiles.jl")

# --- Packing: packed-panel formats and the packers that fill them ---
include("packing/format.jl")
include("packing/panel.jl")
include("packing/pack.jl")
include("packing/pack_contiguous.jl")

# --- Microkernels: accumulate over one packed K panel, store into C ---
include("microkernels/interface.jl")
include("microkernels/scalar.jl")
include("microkernels/simd.jl")
include("microkernels/planar.jl")
include("microkernels/onem.jl")
include("microkernels/fmaddsub.jl")

# --- Planning: labels, conjugation, kernel and blocking choice, the plan ---
include("planning/labels.jl")
include("planning/conjugation.jl")
include("planning/kernel_selection.jl")
include("planning/blocking.jl")
# `ContractPlan` holds a `ContractWorkspace`, so the workspace comes first.
include("execution/workspace.jl")
include("planning/plan.jl")

# --- Execution: the five-loop nest, the tile-by-tile oracle, the opt-in
# unpacked direct path for small contractions, the opt-in half-packed
# path (B packed, one-tile A read in place), and the automatically selected
# unpacked-B path (A packed, B read in place) ---
include("execution/macrokernel.jl")
include("execution/execute.jl")
include("execution/oracle.jl")
include("execution/direct.jl")
include("execution/halfpack.jl")
include("execution/unpackedb.jl")
include("execution/dot.jl")

# --- Integrations ---
include("integrations/tensoroperations.jl")

export QuasiStridedBackend

# Hardware detection runs once per process, never at precompile time: a .ji
# cached on one node class of a shared depot must not carry another node's
# feature set (src/hardware/target.jl).
function __init__()
    _init_target!()
    return nothing
end

@static if VERSION >= v"1.11"
    eval(
        Expr(
            :public, :contract!, :plan_contract, :execute!, :ContractPlan,
            :ContractWorkspace, :Blocking, :default_blocking,
            :ScalarKernel, :SIMDKernel,
            # `PlanarKernel` is the default for a complex
            # element type, and `OneMKernel` is reachable ONLY by naming it in
            # `plan_contract(...; kernel = ...)` -- the engine never picks it,
            # deliberately. A selection mechanism whose only handle is an
            # internal name is not a selection mechanism, so both belong in
            # this tier alongside `ScalarKernel`/`SIMDKernel`.
            :PlanarKernel, :OneMKernel, :FMAddSubKernel,
            :target_profile, :cache_topology,
            :TargetProfile, :CacheLevel
        )
    )
end

end # module QuasiStrided
