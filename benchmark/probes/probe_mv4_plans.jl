# Plan dump for the MV = 4 regression cases (probe_mv4_regress.jl's list, or
# `--cases id1,id2`): every pairwise plan a case builds (register shape,
# whether the operand roles were swapped, blocking, extents, C's M/N strides,
# the M composite's unit-stride run), then the case's median time.
#
#   julia -t 1 --project=benchmark benchmark/probes/probe_mv4_plans.jl --mode fix
#
# `--mode`: `fix` (this tree as is), `nostore` (the `_store_shape` step-down
# disabled: the PR #10 head's selection), `mv2` (the real AVX-512 rule at
# MV = 2: main's selection). The modes redefine package internals, so run
# one mode per process.

using TensorOperations, QuasiStrided, TensorOperationsBenchmarks, Statistics, Printf
using QuasiStrided: ContractPlan, mr, nr, axis_length, RealMethod
const TOB = TensorOperationsBenchmarks
TensorOperations.LinearAlgebra.BLAS.set_num_threads(1)
include(joinpath(@__DIR__, "..", "harness.jl"))

const MODE = argopt("mode", "fix")
if MODE == "nostore"
    @eval QuasiStrided @inline _store_shape(
        shape::NTuple{3, Int}, ::Type{T}, method::RealMethod, Qm::Int, run::Int
    ) where {T} = shape
elseif MODE == "mv2"
    @eval QuasiStrided _rule_mv(::Val{:avx512}, ::RealMethod) = 2
    QuasiStrided._DEFAULTS_F64[] = nothing
    QuasiStrided._DEFAULTS_F32[] = nothing
elseif MODE != "fix"
    error("unknown --mode $MODE")
end

const LOG = Ref(false)
const SEEN = Set{String}()
function log_plan(plan)
    g = plan.mgroup; h = plan.ngroup
    s = @sprintf(
        "    MR=%-3d Qm=%-6d Qn=%-6d Qk=%-5d blk=(%d,%d,%d)  M(C)=%s N(C)=%s K(A,B)=%s",
        mr(plan.kernel), axis_length(g), axis_length(h), axis_length(plan.kgroup),
        plan.blocking.mc, plan.blocking.kc, plan.blocking.nc,
        string(g.strides[2]), string(h.strides[2]), string(plan.kgroup.strides)
    )
    s in SEEN || (push!(SEEN, s); println(s))
    return nothing
end

@eval QuasiStrided @inline function _continue(e::_Execute{T}, plan::ContractPlan{T}, hint) where {T}
    $(LOG)[] && $(log_plan)(plan)
    return (_execute_hinted!(plan, e.alpha, e.beta, hint); nothing)
end

ids = split(argopt("cases", "ao2mo_2_dim16,ao2mo_2_dim24,ccsd_t_2_dim16,ccsd_t_4_dim16,ccsd_t_2_dim24,mps_1site_D64,mps_2site_D64"), ',')
dtype = parse_dtypes(argopt("dtype", "Float64"))[1]
cases = filter(c -> c.id in ids, vcat(TOB._tccg_cases((8, 16, 24)), TOB._mps_cases((64, 128, 256))))
println("mode = $MODE  T = $dtype  default shape = ", QuasiStrided._resolved_defaults(dtype).shape)
for c in cases
    s = c.spec
    if s isa TOB.ContractSpec
        dims(I) = ntuple(i -> s.dims[I[i]], length(I))
        A = randn(dtype, dims(s.IA)); B = randn(dtype, dims(s.IB)); C = zeros(dtype, dims(s.IC))
        pA, pB, pAB = TensorOperations.contract_indices(s.IA, s.IB, s.IC)
        f = () -> TensorOperations.tensorcontract!(C, A, pA, false, B, pB, false, pAB, one(dtype), zero(dtype), QuasiStridedBackend())
    else
        ts = [randn(dtype, ntuple(i -> s.dims[abs(il[i])], length(il))...) for il in s.indexlists]
        f = () -> ncon(ts, s.indexlists, s.conjlist; order = s.order, output = s.output, backend = QuasiStridedBackend())
    end
    println(c.id)
    f()
    LOG[] = true; empty!(SEEN); f(); LOG[] = false
    ts_ = [(@elapsed f()) for _ in 1:21]
    @printf("    median %.1f us  (min %.1f, max %.1f)\n", median(ts_) * 1.0e6, minimum(ts_) * 1.0e6, maximum(ts_) * 1.0e6)
end
