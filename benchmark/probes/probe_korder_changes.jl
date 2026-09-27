# Which upstream-suite :contract cases does `_order_contract_labels`
# (src/planning/labels.jl) actually reorder? For every synthetic/TCCG/batched
# case at one small leg dim, plan it and compare the K composite against the
# pre-rule order (`indA` order, i.e. `_classify_labels`' output). Cases whose
# K group is unchanged run the identical code path before and after the rule
# and cannot have moved in a before/after benchmark except by noise.
#
# The cost model reads the operand sizes (L2 fit, page-sized walks), so run it
# at the leg dims actually benchmarked.
#
#   julia --project=benchmark benchmark/probes/probe_korder_changes.jl \
#       [--dim 4] [--sources synthetic,tccg,batched]

using TensorOperations
using TensorOperationsBenchmarks
using TensorOperationsBenchmarks: ContractSpec, BatchedContractSpec
using QuasiStrided
using QuasiStrided: plan_contract, _classify_labels, _build_pair_group
using StridedViews
using Random

include(joinpath(@__DIR__, "..", "harness.jl"))

const TOB = TensorOperationsBenchmarks
const DIM = argopt("dim", 4)
const SOURCES = split(argopt("sources", "synthetic,tccg,batched"), ',')

gens = Dict("synthetic" => TOB._synthetic_contract_cases, "tccg" => TOB._tccg_cases, "batched" => TOB._batched_contract_cases)
cases = reduce(vcat, [gens[s]((DIM,)) for s in SOURCES])
changed = String[]
unchanged = 0
for case in cases
    spec = case.spec
    spec isa Union{ContractSpec, BatchedContractSpec} || continue
    dimsA = ntuple(i -> spec.dims[spec.IA[i]], length(spec.IA))
    dimsB = ntuple(i -> spec.dims[spec.IB[i]], length(spec.IB))
    dimsC = ntuple(i -> spec.dims[spec.IC[i]], length(spec.IC))
    A = StridedView(Array{Float64}(undef, dimsA...))  # planning never reads values
    B = StridedView(Array{Float64}(undef, dimsB...))
    C = StridedView(Array{Float64}(undef, dimsC...))
    pA, pB, pAB = TensorOperations.contract_indices(spec.IA, spec.IB, spec.IC)
    indA, indB, indC = QuasiStrided._qs_labels(pA, pB, pAB)
    plan = plan_contract(C, A, indA, B, indB, indC; oracle = false)
    _, _, klabels = _classify_labels(indA, indB, indC)
    old = _build_pair_group(klabels, indA, A, indB, B)  # pre-rule K group, (A, B) maps
    swapped = plan.Astorage === parent(B)
    new = plan.kgroup
    newstrides = swapped ? (new.strides[2], new.strides[1]) : new.strides
    if (new.lengths, newstrides) == (old.lengths, old.strides)
        global unchanged += 1
    else
        push!(changed, "$(case.id)  K old=$(klabels) A/B strides $(old.strides) -> new A/B strides $(newstrides)")
    end
end
println("dim = $DIM: $(length(cases)) cases, $(unchanged) with an unchanged K group, $(length(changed)) reordered:")
foreach(println, changed)
