# Dispatch barriers: how a contraction reaches the ONE kernel and the ONE
# execution path it runs, compiling nothing else, while allocating nothing.
#
# The problem. The kernel (a menu shape, chosen from the host and the extents)
# and the execution path (the five-loop nest with B packed or read in place,
# the dot path at one of four lane widths, the outer-product path at one of
# four) are runtime choices, and everything below them is specialised on
# them. A static branch over every choice -- an unrolled ladder with a
# concrete call in each arm -- is allocation-free and cheap to run, but Julia
# infers and compiles every arm the first time the ladder itself is compiled:
# the whole menu (ten kernels for ComplexF64) times both B paths times the
# degenerate paths, and for a complex eltype times the four `conj`/`identity`
# transform pairs. Measured on ccqlin038 (Julia 1.12.7, SnoopCompile, the
# first `@tensor` call of benchmark/probes/probe_ttfx.jl): 40 `execute!`
# instances and 320 `_dot_nest!` instances, 85 s of inference, for ONE
# ComplexF64 contraction; 6 plans and 26 s for Float64.
#
# The fix is a dynamic call -- inference stops at it, and only the callee the
# call actually reaches is ever compiled -- arranged so that it boxes nothing.
# `jl_apply_generic` passes every argument as a boxed pointer, so a call
# allocates exactly for the arguments that are not already heap objects or
# singletons. Every barrier below therefore passes:
#
#   * the static choices (kernel shape, method, transforms, path) as
#     SINGLETON values -- a `Val` or an empty struct each;
#   * the operand storages (`parent` of each `StridedView`, a `Memory` or
#     `Vector`) as themselves, which are already heap objects;
#   * everything else -- groups, bases, blocking, scalars, the workspace --
#     through a typed slot, a `Base.RefValue{P}` owned by the workspace's
#     `_SlotCache` and reused across calls: written just before the call,
#     read back as the callee's first action.
#
# The payload never holds an operand array (they travel as arguments), so a
# pooled, task-lifetime workspace retains no user data between calls; it
# may hold the workspace itself, which is a cycle within the workspace, not a
# retention. Measured in isolation (Julia 1.12.7): the barrier -- slot write
# plus the dynamic call, method-cache hit -- costs ~15 ns and 0 bytes over
# the equivalent static branch.
#
# Reentrancy/threading: a slot belongs to one workspace, and a workspace to
# one task at a time (the pool is task-local; an explicitly shared workspace
# was never safe to use concurrently). The callee copies the payload out
# before doing anything that could run another contraction.

# The typed slot for payload type `P` in `ws`, created on first use. The
# common case, the same payload type as the previous barrier crossing with
# this workspace, is a single type-tag test.
@inline function _barrier_slot!(ws::ContractWorkspace, ::Type{P}) where {P}
    cache = ws.slots
    slot = cache.last
    slot isa Base.RefValue{P} && return slot
    return _barrier_slot_slow!(cache, P)
end

# No workspace yet (`plan_contract(...; workspace = nothing)`, which builds a
# fresh one after the kernel is known): a one-off slot, which allocates, on a
# path that allocates a whole workspace anyway.
@inline _barrier_slot!(::Nothing, ::Type{P}) where {P} = Base.RefValue{P}()

@noinline function _barrier_slot_slow!(cache::_SlotCache, ::Type{P}) where {P}
    slot = get!(() -> Base.RefValue{P}(), cache.slots, P)::Base.RefValue{P}
    cache.last = slot
    return slot
end

# ----------------------------------------------------------------------------
# Execution paths
# ----------------------------------------------------------------------------
#
# What `execute!` runs once the empty-output and beta-only short-circuits are
# past, as a singleton value, so that it can cross a barrier for free and
# select exactly one `_execute_path!` method. Chosen by `_select_path`
# (src/execution/execute.jl).

# The five-loop nest (src/execution/execute.jl). `UNPACKED_B`: B read in
# place by the microkernel (`_use_unpacked_b`, src/execution/unpackedb.jl)
# rather than packed. `AFF`: which of the six sliver axes the nest builds are
# statically `AffineAxis` rather than `Union{AffineAxis,PtrScatterAxis}`, one
# `Bool` per map in the order M-in-A, M-in-C, N-in-B, N-in-C, K-in-A, K-in-B
# (each group's maps in their own order) -- true where that map of its
# composite is an affine ramp (`_map_ramp_step`), so that every sliver
# descriptor of it is regular. Each `Union` axis is split at every pack and
# tile call it reaches, so a plan whose maps are all ramps (the common case)
# compiles one variant of each where the `Union` compiled two to four.
struct _NestPath{UNPACKED_B, AFF} end

@inline _is_ramp_map(g::AxisGroup, p::Int) = _map_ramp_step(g, p) !== nothing

# The nest path for a plan with these groups. N's B map is left `false` when B
# is read in place: that path builds no B sliver axis (`_unpacked_b_view`
# reads the descriptor itself), so the flag would only split specialisations.
@inline _nest_path(unpacked_b::Bool, mgroup::AxisGroup, ngroup::AxisGroup, kgroup::AxisGroup) =
    _nest_path_from_flags(
    unpacked_b,
    _is_ramp_map(mgroup, 1), _is_ramp_map(mgroup, 2),
    !unpacked_b && _is_ramp_map(ngroup, 1), _is_ramp_map(ngroup, 2),
    _is_ramp_map(kgroup, 1), _is_ramp_map(kgroup, 2)
)

# Seven runtime `Bool`s to the `_NestPath` singleton they name, looked up in a
# table of all 128: one index computation and one load. The value is typed
# `Any` -- it is only ever passed through a barrier, which dispatches on it,
# or tested with `isa` -- so neither a runtime `_NestPath{u, aff}()` (which
# applies the type dynamically, ~100 ns) nor a 128-leaf literal decision
# tree (0.27 s of inference on the first call) is needed.
@inline function _nest_path_from_flags(flags::Vararg{Bool, 7})
    index = 1
    for i in 1:7
        index += flags[i] << (i - 1)
    end
    return @inbounds _NEST_PATHS[index]
end

const _NEST_PATHS = let
    table = Vector{Any}(undef, 128)
    for index in 1:128
        bits = ntuple(i -> ((index - 1) >> (i - 1)) & 1 == 1, 7)
        table[index] = _NestPath{bits[1], bits[2:end]}()
    end
    table
end

# The opt-in half-packed nest (`execute_half_packed!`, src/execution/halfpack.jl),
# with `_NestPath`'s six static affine-axis flags. Its M-in-A flag is always
# `false`: that path builds no A sliver axis (A is read in place).
struct _HalfPackPath{AFF} end

@inline _half_pack_path(mgroup::AxisGroup, ngroup::AxisGroup, kgroup::AxisGroup) =
    @inbounds _HALF_PACK_PATHS[
    1 + (_is_ramp_map(mgroup, 2) << 1) +
        (_is_ramp_map(ngroup, 1) << 2) + (_is_ramp_map(ngroup, 2) << 3) +
        (_is_ramp_map(kgroup, 1) << 4) + (_is_ramp_map(kgroup, 2) << 5),
]

const _HALF_PACK_PATHS = let
    table = Vector{Any}(undef, 64)
    for index in 1:64
        table[index] = _HalfPackPath{ntuple(i -> ((index - 1) >> (i - 1)) & 1 == 1, 6)}()
    end
    table
end

# The K-vectorized dot path (src/execution/dot.jl) at lane width `W`; `MATB`:
# the matrix operand is B (`Qm == 1`), else A (`Qn == 1`).
struct _DotPath{MATB, W} end

# The streaming outer-product path (src/execution/outer.jl) at lane width `W`.
struct _OuterPath{W} end

# `Val(W)`'s singleton for the lane widths the degenerate paths compile,
# from a runtime `W` (`_dot_lanewidth`), as a literal per arm.
@inline function _lane_path(::Type{P}, W::Int) where {P}
    W == 16 && return P{16}()
    W == 8 && return P{8}()
    W == 4 && return P{4}()
    return P{2}()
end
