# Execution paths: singletons that select one `execute_path!` method.
# `plan_contract` decides the path (`select_path`) and stores it in the plan,
# so `execute!` reaches the path's code statically.

# Nothing to compute: M or N is empty.
struct EmptyPath end

# `C *= beta`: K is empty.
struct ScalePath end

# The five-loop nest. `UNPACKED_B`: B read in place instead of packed. `AFF`:
# per map, in the order M-in-A, M-in-C, N-in-B, N-in-C, K-in-A, K-in-B,
# whether it is an affine ramp, so that its sliver axes are statically
# `AffineAxis` and the offset-buffer `view` arm is never compiled. `SPLIT`: whether
# A and B are packed line by line (`plan.mpack`/`plan.npack`).
struct NestPath{UNPACKED_B, AFF, SPLIT} end

@inline is_ramp_map(g::AxisGroup, p::Int) = map_ramp_step(g, p) !== nothing

# N's B flag stays `false` when B is read in place: that path builds no B
# sliver axis, so the flag would only split specialisations. A split group is
# enumerated in another order, so its ramp flags are `false`. `panel`: C is
# `PanelPath`'s dense panel, whose M and N maps are ramps.
@inline function nest_path(
        unpacked_b::Bool, mgroup::AxisGroup, ngroup::AxisGroup, kgroup::AxisGroup,
        split_a::Bool, split_b::Bool, panel::Bool = false
    )
    split_b &= !unpacked_b
    return nest_path_from_flags(
        panel ? _PANEL_PATHS : _NEST_PATHS, unpacked_b,
        !split_a && is_ramp_map(mgroup, 1), !split_a && (panel || is_ramp_map(mgroup, 2)),
        !unpacked_b && !split_b && is_ramp_map(ngroup, 1), !split_b && (panel || is_ramp_map(ngroup, 2)),
        is_ramp_map(kgroup, 1), is_ramp_map(kgroup, 2), split_a, split_b
    )
end

# The nest run against a dense panel of C in the compute type.
struct PanelPath{P <: NestPath} end

# Table lookup rather than `NestPath{u, aff, split}()` built at runtime (a
# dynamic type application) or a 512-leaf branch tree (slow to infer).
@inline function nest_path_from_flags(table::Vector{Any}, flags::Vararg{Bool, 9})
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
        table[index] = NestPath{bits[1], bits[2:7], bits[8:9]}()
    end
    table
end

const _PANEL_PATHS = Any[PanelPath{typeof(p)}() for p in _NEST_PATHS]

# The dot path at lane width `W`; `MATB`: the matrix operand is B
# (`m_length == 1`).
struct DotPath{MATB, W} end

# The outer-product path at lane width `W`.
struct OuterPath{W} end

# A runtime lane width to its path singleton, one literal per arm.
@inline function lane_path(::Type{P}, W::Int) where {P}
    W == 16 && return P{16}()
    W == 8 && return P{8}()
    W == 4 && return P{4}()
    return P{2}()
end
