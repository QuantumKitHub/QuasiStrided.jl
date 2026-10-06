# Execution paths: singletons that select one `execute_path!` method.
# `plan_contract` decides the path (`select_path`) and stores it in the plan,
# so `execute!` reaches the path's code statically.

# Nothing to compute: M or N is empty.
struct EmptyPath end

# `C *= beta`: K is empty.
struct ScalePath end

# The five-loop nest. `UNPACKED_B`: B read in place instead of packed. `SPLIT`:
# whether A and B are packed line by line (`plan.mpack`/`plan.npack`).
struct NestPath{UNPACKED_B, SPLIT} end

# The nest run against a dense panel of C in the compute type.
struct PanelPath{P <: NestPath} end

# B read in place is never split. One literal per arm, so that no type is
# built at runtime.
@inline function nest_path(unpacked_b::Bool, split_a::Bool, split_b::Bool, panel::Bool = false)
    if unpacked_b
        return split_a ? nest_or_panel(NestPath{true, (true, false)}, panel) :
            nest_or_panel(NestPath{true, (false, false)}, panel)
    elseif split_a
        return split_b ? nest_or_panel(NestPath{false, (true, true)}, panel) :
            nest_or_panel(NestPath{false, (true, false)}, panel)
    else
        return split_b ? nest_or_panel(NestPath{false, (false, true)}, panel) :
            nest_or_panel(NestPath{false, (false, false)}, panel)
    end
end

@inline nest_or_panel(::Type{P}, panel::Bool) where {P <: NestPath} = panel ? PanelPath{P}() : P()

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
