# Which pairwise contractions inside an upstream :network case does the K
# order rule (`_order_contract_labels`, src/planning/labels.jl) change, and
# what does each change cost? Runs the case through `ncon` with
# QuasiStridedBackend, logging every K-order decision (labels, both operands'
# K strides and sizes, the chosen operand), then times the whole network with
# the rule ON and OFF (interleaved), and each changed pairwise contraction on
# its own, rebuilt from the logged shapes.
#
#   julia -t 1 --project=benchmark benchmark/probes/probe_network_korder.jl \
#       --case mps_1site --size 32 --dtype ComplexF64 --reps 21
#
# The rule is switched by redefining `_order_contract_labels` here with a
# mode flag -- a probe-only hack, never to be copied into src/.

using TensorOperations
using TensorOperations: StridedBLAS
using TensorOperationsBenchmarks
using TensorOperationsBenchmarks: ArrayProvider, randtensor, NetworkSpec
using QuasiStrided
using QuasiStrided: QuasiStridedBackend, _choose_k_order, _k_order_cost
using StridedViews
using Random
using Statistics: median
using Printf
using LinearAlgebra

include(joinpath(@__DIR__, "..", "harness.jl"))

const TOB = TensorOperationsBenchmarks
const CASE = argopt("case", "mps_1site")
const SIZE = argopt("size", 32)
const T = parse_dtypes(argopt("dtype", "ComplexF64"))[1]
const REPS = argopt("reps", 21)

const MODE = Ref(:new)  # :new (cost model), :old (indA order), :simple (12ae2b4's rule)

# 12ae2b4's rule: sort by stride in the operand with the smaller minimum K stride.
function simple_rule(klabels, indA, A, indB, B, Qm, Qn)
    minst(ind, v) = minimum((abs(Base.strides(v)[findfirst(==(l), ind)]) for l in klabels if size(v, findfirst(==(l), ind)) > 1); init = typemax(Int))
    mA = minst(indA, A); mB = minst(indB, B)
    by_b = mB < mA || (mB == mA && Qn > Qm)
    return by_b ? QuasiStrided._sort_labels_by_stride(klabels, indB, B) : QuasiStrided._sort_labels_by_stride(klabels, indA, A)
end
const LOG = Vector{Any}()

# Replaces the entry point without an explicit `l2bytes` (the one
# `plan_contract` calls); the label lists are NTuples (src/planning/labels.jl).
function QuasiStrided._order_contract_labels(
        klabels::NTuple{DK, Int}, indA::NTuple{NA, Int}, A::StridedView, morder::NTuple{DM, Int},
        indB::NTuple{NB, Int}, B::StridedView, norder::NTuple{DN, Int}, Qm::Int, Qn::Int
    ) where {DK, NA, NB, DM, DN}
    DK <= 1 && return klabels
    l2bytes = QuasiStrided._l2_core_bytes()
    new = _choose_k_order(klabels, indA, A, morder, indB, B, norder, Qm, Qn, l2bytes)
    stA = Base.strides(A); stB = Base.strides(B)
    kA = Tuple(stA[findfirst(==(l), indA)] for l in klabels)
    kB = Tuple(stB[findfirst(==(l), indB)] for l in klabels)
    kL = Tuple(size(A, findfirst(==(l), indA)) for l in klabels)
    costs = Tuple((_k_order_cost(o, indA, A, morder, Qm, l2bytes), _k_order_cost(o, indB, B, norder, Qn, l2bytes)) for o in (klabels, new))
    simple = simple_rule(klabels, indA, A, indB, B, Qm, Qn)
    push!(LOG, (; klabels, kL, kA, kB, sizeA = size(A), stA, indA, sizeB = size(B), stB, indB, Qm, Qn, costs, changed = new != klabels || simple != klabels, new, simple))
    return MODE[] === :new ? new : MODE[] === :simple ? simple : klabels
end

gens = Dict("mps_1site" => TOB._mps_cases, "mps_2site" => TOB._mps_cases, "trg_plaquette" => TOB._trg_cases, "ctmrg_corner" => TOB._ctmrg_cases)
cases = gens[CASE]((SIZE,))
case = cases[findfirst(c -> startswith(c.id, CASE), cases)]
spec = case.spec::NetworkSpec
provider = ArrayProvider{T}()
tensors = map(spec.indexlists) do il
    dims = ntuple(i -> spec.dims[abs(il[i])], length(il))
    randtensor(provider, il, dims, T)
end
run(backend) = TensorOperations.ncon(tensors, spec.indexlists, spec.conjlist; order = spec.order, output = spec.output, backend = backend)

println("case = $(case.id)  T = $T  indexlists = $(spec.indexlists)  order = $(spec.order)")
empty!(LOG); MODE[] = :new
Cnew = run(QuasiStridedBackend())
decisions = unique(e -> (e.klabels, e.kA, e.kB, e.indA, e.indB, e.sizeA, e.sizeB), LOG)
println("pairwise contractions with >1 K label: $(length(decisions)) distinct (of $(length(LOG)) planned)")
for e in decisions
    println(
        "  K=", e.klabels, " lengths=", e.kL, "  A: size=", e.sizeA, " ind=", e.indA, " Kstrides=", e.kA, "  B: size=", e.sizeB, " ind=", e.indB, " Kstrides=", e.kB,
        "  Qm=", e.Qm, " Qn=", e.Qn, "  cost(A,B) old=", e.costs[1], " new=", e.costs[2], e.changed ? "  CHANGED: new=$(e.new) simple=$(e.simple)" : "  (unchanged)"
    )
end
MODE[] = :old
Cold = run(QuasiStridedBackend())
Cblas = run(StridedBLAS())
println("max|new-old| = ", maximum(abs, Cnew .- Cold), "  max|new-blas| = ", maximum(abs, Cnew .- Cblas))

# Whole network, interleaved.
function tmed(f)
    f()
    ts = [(t0 = time_ns(); f(); (time_ns() - t0) / 1.0e9) for _ in 1:REPS]
    return median(ts)
end
for round in 1:2
    MODE[] = :old; told = tmed(() -> run(QuasiStridedBackend()))
    MODE[] = :new; tnew = tmed(() -> run(QuasiStridedBackend()))
    MODE[] = :simple; tsim = tmed(() -> run(QuasiStridedBackend()))
    @printf("round %d  network: simple rule %.3e s (simple/old %.3f)\n", round, tsim, tsim / told)
    tblas = tmed(() -> run(StridedBLAS()))
    @printf("round %d  network: QS old %.3e s  QS new %.3e s  (new/old %.3f)  BLAS %.3e s\n", round, told, tnew, tnew / told, tblas)
end

# Each CHANGED pairwise contraction on its own, rebuilt dense at the logged
# shapes (the network's intermediates are dense, so this is the same layout).
for e in decisions
    e.changed || continue
    A = randn(T, e.sizeA...); B = randn(T, e.sizeB...)
    mlab = [l for l in e.indA if !(l in e.klabels)]
    nlab = [l for l in e.indB if !(l in e.klabels)]
    indC = Tuple(vcat(mlab, nlab))
    C = zeros(T, Tuple(l in e.indA ? size(A, findfirst(==(l), e.indA)) : size(B, findfirst(==(l), e.indB)) for l in indC)...)
    Av, Bv, Cv = StridedView(A), StridedView(B), StridedView(C)
    f() = QuasiStrided.contract!(Cv, one(T), Av, e.indA, Bv, e.indB, zero(T), indC)
    MODE[] = :old; told = tmed(f)
    MODE[] = :new; tnew = tmed(f)
    MODE[] = :old; told2 = tmed(f)
    MODE[] = :new; tnew2 = tmed(f)
    MODE[] = :simple; tsim = tmed(f); MODE[] = :simple; tsim2 = tmed(f)
    @printf("  pairwise K=%s: simple %.3e,%.3e s  simple/old %.3f\n", e.klabels, tsim, tsim2, min(tsim, tsim2) / min(told, told2))
    flops = 2.0 * prod(size(C)) * prod(e.kL)
    @printf(
        "  pairwise K=%s A%s B%s: old %.3e,%.3e s  new %.3e,%.3e s  new/old %.3f  (%.1f GF/s new)\n",
        e.klabels, e.sizeA, e.sizeB, told, told2, tnew, tnew2, min(tnew, tnew2) / min(told, told2), flops / min(tnew, tnew2) / 1.0e9
    )
end
