# Dispatch barriers. The kernel and the execution path are runtime choices
# that everything below them is specialised on. A static branch over every
# choice would make Julia infer and compile every arm (the whole kernel menu x
# every path x, for complex, four transform pairs) on the first call. A
# dynamic call instead compiles only the callee actually reached.
#
# `jl_apply_generic` boxes every argument that is not already a heap object
# or a singleton, so each barrier passes:
#   * static choices (kernel shape, method, transforms, path) as singletons;
#   * operand storages (already heap objects) as themselves;
#   * everything else through a typed slot, a `Base.RefValue{P}` owned by the
#     workspace's `_SlotCache`: written just before the call, read back first
#     thing in the callee.
# The payload never holds an operand array, so a pooled workspace retains no
# user data between calls. A workspace belongs to one task at a time, and
# the callee copies the payload out before it could run another contraction.

# The typed slot for payload type `P`; the common case (same payload type as
# the previous crossing) is one type-tag test.
@inline function _barrier_slot!(ws::ContractWorkspace, ::Type{P}) where {P}
    cache = ws.slots
    slot = cache.last
    slot isa Base.RefValue{P} && return slot
    return _barrier_slot_slow!(cache, P)
end

# No workspace yet: a one-off slot, on a path that allocates a workspace anyway.
@inline _barrier_slot!(::Nothing, ::Type{P}) where {P} = Base.RefValue{P}()

@noinline function _barrier_slot_slow!(cache::_SlotCache, ::Type{P}) where {P}
    slot = get!(() -> Base.RefValue{P}(), cache.slots, P)::Base.RefValue{P}
    cache.last = slot
    return slot
end

# ----------------------------------------------------------------------------
# Execution paths: singletons that select one `_execute_path!` method.
# ----------------------------------------------------------------------------

# The five-loop nest. `UNPACKED_B`: B read in place instead of packed. `AFF`:
# per map, in the order M-in-A, M-in-C, N-in-B, N-in-C, K-in-A, K-in-B,
# whether it is an affine ramp, so that its sliver axes are statically
# `AffineAxis` and the `PtrScatterAxis` arm is never compiled. `SPLIT`: whether
# A and B are packed line by line (`plan.mpack`/`plan.npack`).
struct _NestPath{UNPACKED_B, AFF, SPLIT} end

@inline _is_ramp_map(g::AxisGroup, p::Int) = _map_ramp_step(g, p) !== nothing

# N's B flag stays `false` when B is read in place: that path builds no B
# sliver axis, so the flag would only split specialisations. A split group is
# enumerated in another order, so its ramp flags are `false`. `panel`: C is
# `_PanelPath`'s dense panel, whose M and N maps are ramps.
@inline function _nest_path(
        unpacked_b::Bool, mgroup::AxisGroup, ngroup::AxisGroup, kgroup::AxisGroup,
        split_a::Bool, split_b::Bool, panel::Bool = false
    )
    split_b &= !unpacked_b
    return _nest_path_from_flags(
        panel ? _PANEL_PATHS : _NEST_PATHS, unpacked_b,
        !split_a && _is_ramp_map(mgroup, 1), !split_a && (panel || _is_ramp_map(mgroup, 2)),
        !unpacked_b && !split_b && _is_ramp_map(ngroup, 1), !split_b && (panel || _is_ramp_map(ngroup, 2)),
        _is_ramp_map(kgroup, 1), _is_ramp_map(kgroup, 2), split_a, split_b
    )
end

# The nest run against a dense panel of C in the compute type.
struct _PanelPath{P <: _NestPath} end

# Table lookup rather than `_NestPath{u, aff, split}()` built at runtime (a
# dynamic type application) or a 512-leaf branch tree (slow to infer).
@inline function _nest_path_from_flags(table::Vector{Any}, flags::Vararg{Bool, 9})
    index = 1
    for i in 1:9
        index += flags[i] << (i - 1)
    end
    return @inbounds table[index]
end

const _NEST_PATHS = let
    table = Vector{Any}(undef, 512)
    for index in 1:512
        bits = ntuple(i -> ((index - 1) >> (i - 1)) & 1 == 1, 9)
        table[index] = _NestPath{bits[1], bits[2:7], bits[8:9]}()
    end
    table
end

const _PANEL_PATHS = Any[_PanelPath{typeof(p)}() for p in _NEST_PATHS]

# The dot path at lane width `W`; `MATB`: the matrix operand is B (`Qm == 1`).
struct _DotPath{MATB, W} end

# The outer-product path at lane width `W`.
struct _OuterPath{W} end

# A runtime lane width to its path singleton, one literal per arm.
@inline function _lane_path(::Type{P}, W::Int) where {P}
    W == 16 && return P{16}()
    W == 8 && return P{8}()
    W == 4 && return P{4}()
    return P{2}()
end
