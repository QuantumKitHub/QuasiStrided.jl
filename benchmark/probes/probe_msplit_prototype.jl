# Prototype: does splitting A's unit-stride M axis into (inner, outer) and
# moving the inner part forward in the M enumeration fix the A-pack line
# amplification of the TCCG intensli cases, and at what store cost?
#
# Builds the plan normally, then rebuilds it with a hand-made M AxisGroup:
#   :asis  -> the planner's order (C-stride order, A's fastest axis last)
#   :pos2  -> C's fastest axis first, then `li` of A's fastest axis, then the rest
#   :pos1  -> `li` of A's fastest axis first, then the C order
#
#   julia -t 1 --project=benchmark benchmark/probes/probe_msplit_prototype.jl --case intensli_7 --dim 24 --dtype Float64 --li 8

using QuasiStrided
using QuasiStrided: plan_contract, execute!, mr, nr, axis_length, AxisGroup, ContractPlan,
    _default_kernel, _kernel_from_shape, kernel_shapes, complex_method
using StridedViews
using Random
using Statistics: median
using Printf
using LinearAlgebra

include(joinpath(@__DIR__, "..", "harness.jl"))

const KNOWN = Dict(
    "intensli_6" => "abcde bf>dcfea",
    "intensli_7" => "abcde df>ecbfa",
    "intensli_8" => "abcde fb>dfcea",
    "ccsd_t_3" => "abcd cefg>fgdabe",
)
const CASE = argopt("case", "intensli_7")
const EXPR = something(argval("expr"), get(KNOWN, CASE, nothing))
const DIM = argopt("dim", 24)
const T = parse_dtypes(argopt("dtype", "Float64"))[1]
const REPS = argopt("reps", 7)
const LI = argopt("li", 8)

function parse_expr(expr::AbstractString)
    lhs, rhs = split(expr, '>')
    a, b = split(strip(lhs), ' ')
    return collect(a), collect(b), collect(strip(rhs))
end
const LA, LB, LC = parse_expr(EXPR)
const LETTERS = unique(vcat(LA, LB, LC))
label(c) = findfirst(==(c), LETTERS)
const indA = Tuple(label.(LA)); const indB = Tuple(label.(LB)); const indC = Tuple(label.(LC))

rng = MersenneTwister(1234)
A = randn(rng, T, ntuple(_ -> DIM, length(LA))...)
B = randn(rng, T, ntuple(_ -> DIM, length(LB))...)
C = zeros(T, ntuple(_ -> DIM, length(LC))...)
Av, Bv, Cv = StridedView(A), StridedView(B), StridedView(C)

plan0 = plan_contract(Cv, Av, indA, Bv, indB, indC; oracle = false)
g = plan0.mgroup
println("case = $CASE dim = $DIM T = $T kernel = ", typeof(plan0.kernel).name.name, " MR=", mr(plan0.kernel), " blocking = ", plan0.blocking)
println("as-is M group: lengths=", g.lengths, " strides=", g.strides)
flops = 2.0 * axis_length(plan0.mgroup) * axis_length(plan0.ngroup) * axis_length(plan0.kgroup)

# Split dim `du` (1-based, in the group's own order) into (li, L/li) and place
# the inner part at `pos`, the outer part where `du` was.
function split_group(g::AxisGroup{D, P}, du::Int, li::Int, pos::Int) where {D, P}
    L = g.lengths[du]
    @assert L % li == 0
    dims = [(g.lengths[d], ntuple(p -> g.strides[p][d], P)) for d in 1:D]
    inner = (li, ntuple(p -> g.strides[p][du], P))
    outer = (L ÷ li, ntuple(p -> li * g.strides[p][du], P))
    dims[du] = outer
    insert!(dims, pos, inner)
    lengths = ntuple(i -> dims[i][1], D + 1)
    strides = ntuple(p -> ntuple(i -> dims[i][2][p], D + 1), P)
    return AxisGroup(lengths, strides)
end

function with_mgroup(plan::ContractPlan, mg; kernel = plan.kernel)
    return ContractPlan(
        kernel, mg, plan.ngroup, plan.kgroup, plan.blocking,
        plan.Astorage, plan.Abase, plan.Bstorage, plan.Bbase, plan.Cstorage, plan.Cbase,
        plan.atransform, plan.btransform, plan.workspace
    )
end

du = argmin([g.lengths[d] > 1 ? abs(g.strides[1][d]) : typemax(Int) for d in 1:length(g.lengths)])
println("A's fastest M axis is group dim $du (A stride $(g.strides[1][du]), length $(g.lengths[du]))")

variants = Pair{String, Any}[]
push!(variants, "asis" => plan0)
if du != 1
    g2 = split_group(g, du, LI, 2)
    g1 = split_group(g, du, LI, 1)
    push!(variants, "pos2(li=$LI)" => with_mgroup(plan0, g2))
    push!(variants, "pos1(li=$LI)" => with_mgroup(plan0, g1))
    # pos2 with the run-demoted kernel (what `_demote_for_run` would pick for run = L_1)
    function demoted_shape(run, kernel)
        best = nothing
        for shape in kernel_shapes(T, complex_method(kernel))
            if run % shape[1] == 0 && (best === nothing || shape[1] > best[1])
                best = shape
            end
        end
        return best
    end
    if T <: Real
        best = demoted_shape(g.lengths[1], plan0.kernel)
        if best !== nothing && best[1] != mr(plan0.kernel)
            k2 = _kernel_from_shape(best, T, complex_method(plan0.kernel))
            p2 = plan_contract(Cv, Av, indA, Bv, indB, indC; oracle = false, kernel = k2)
            push!(variants, "pos2(li=$LI)+MR$(best[1])" => with_mgroup(p2, g2))
        end
    end
end

execute!(plan0, one(T), zero(T)); Cref = copy(C)
for (name, p) in variants
    println("  ", name, ": M group lengths=", p.mgroup.lengths, " strides=", p.mgroup.strides)
end
for round in 1:2
    for (name, p) in variants
        fill!(C, zero(T))
        execute!(p, one(T), zero(T))
        @assert C ≈ Cref "$name wrong result"
        t = median_time_s(() -> execute!(p, one(T), zero(T)); reps = REPS)
        @printf("round %d  %-22s  %9.4f s  %7.2f GF/s\n", round, name, t, flops / t / 1e9)
    end
end
