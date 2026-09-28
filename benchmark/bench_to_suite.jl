# StridedBLAS() vs QuasiStridedBackend() on the upstream TensorOperations.jl
# benchmark suite (TensorOperationsBenchmarks), :contract and :network
# categories. :permute/:trace are not run: QuasiStridedBackend forwards
# tensoradd!/tensortrace! to StridedNative, so they would time the same code
# twice. StridedNative itself is left out: it is very slow on :network's trg.
#
#   julia --project=benchmark benchmark/bench_to_suite.jl [options]
#
# Options (all optional):
#   --categories contract,network
#   --sources synthetic,tccg,batched   # :contract subsets (upstream's params.source)
#   --topics mps,ctmrg,trg             # :network subsets (upstream's params.topic)
#   --dtypes Float64,Float32
#   --synthetic-sizes 4,8,16,32,63     # leg dim of the synthetic contract shapes
#   --tccg-sizes 8,16                  # leg dim applied to every TCCG index
#   --batched-sizes 4,8,16,32,64       # matrix dim of each batched slice
#   --mps-bonddims 32,64,128           # MPS/MPO effective-Hamiltonian bond dim D
#   --ctmrg-chis 16,32,64              # CTMRG environment bond dim chi
#   --trg-chis 16,32,48                # TRG plaquette bond dim chi
#   --reps 21                          # upper bound on timed reps per backend
#   --time-budget 10                   # seconds per (case, backend); reps shrink
#                                      #   (to no fewer than 5) to fit it
#   --max-bytes 2147483648             # per-case skip ceiling
#   --max-flops 200000000000           # per-case skip ceiling
#   --outdir <path>                    # default benchmark/results/<host>-<date>
#
# Each source/topic generator gets its own size sweep (upstream shares one per
# category), since a leg dim sensible for a rank-4 synthetic shape is far too
# large for a rank-6 CCSD(T) equation, and likewise trg's chi^6 vs mps's D^3.
# :network cases run through `ncon`, which allocates its output, so their
# timing includes that allocation (as upstream's does). Batched cases time
# `batch` separate tensorcontract! calls (per-call overhead is the point).
#
# Writes bench_to_suite.csv (per-rep GF/s min/std next to the median-based
# rate, for the plot's violins), canary_to_suite.csv, summary_to_suite.txt,
# mismatches_to_suite.txt and PROVENANCE_to_suite.txt.

using TensorOperations
using TensorOperations: StridedBLAS
using TensorOperationsBenchmarks
using TensorOperationsBenchmarks: BenchmarkCase, ContractSpec, BatchedContractSpec,
    NetworkSpec, flops, isblasequivalent, ArrayProvider, randtensor
using QuasiStrided
using QuasiStrided: QuasiStridedBackend
using Statistics: std
import Pkg

include(joinpath(@__DIR__, "harness.jl"))

const TOB = TensorOperationsBenchmarks

const REPS = argopt("reps", 21)
const MIN_REPS = 5
const TIME_BUDGET = parse(Float64, argopt("time-budget", "10"))
const RUN_DTYPES = parse_dtypes(argopt("dtypes", "Float64,Float32"))
const CATEGORIES = Symbol.(split(argopt("categories", "contract"), ','))
const SOURCES = Symbol.(split(argopt("sources", "synthetic,tccg,batched"), ','))
const TOPICS = Symbol.(split(argopt("topics", "mps,ctmrg,trg"), ','))
const SYNTHETIC_SIZES = parse_ints(argopt("synthetic-sizes", "4,8,16,32,63"))
const TCCG_SIZES = parse_ints(argopt("tccg-sizes", "8,16"))
const BATCHED_SIZES = parse_ints(argopt("batched-sizes", "4,8,16,32,64"))
const MPS_BONDDIMS = parse_ints(argopt("mps-bonddims", "32,64,128"))
const CTMRG_CHIS = parse_ints(argopt("ctmrg-chis", "16,32,64"))
const TRG_CHIS = parse_ints(argopt("trg-chis", "16,32,48"))
const MAX_CASE_BYTES = argopt("max-bytes", 2 * 2^30)
const MAX_CASE_FLOPS = parse(Float64, argopt("max-flops", "2e11"))

const BACKENDS = (
    StridedBLAS = StridedBLAS(),
    QuasiStrided = QuasiStridedBackend(),
)

# The per-source/per-topic generators are called directly (bypassing
# `build_suite`/BenchmarkTools); these asserts make an upstream reshuffle of
# the categories fail loudly.
@assert TOB.REGISTRY[:contract] === TOB._contract_cases
@assert TOB.REGISTRY[:network] === TOB._network_cases
let probe = (8,)
    @assert length(TOB._contract_cases(probe)) == length(TOB._synthetic_contract_cases(probe)) +
        length(TOB._tccg_cases(probe)) + length(TOB._batched_contract_cases(probe))
    @assert length(TOB._network_cases(probe)) == length(TOB._mps_cases(probe)) +
        length(TOB._ctmrg_cases(probe)) + length(TOB._trg_cases(probe))
end

const CONTRACT_GENERATORS = (
    synthetic = (TOB._synthetic_contract_cases, SYNTHETIC_SIZES),
    tccg = (TOB._tccg_cases, TCCG_SIZES),
    batched = (TOB._batched_contract_cases, BATCHED_SIZES),
)
const NETWORK_GENERATORS = (
    mps = (TOB._mps_cases, MPS_BONDDIMS),
    ctmrg = (TOB._ctmrg_cases, CTMRG_CHIS),
    trg = (TOB._trg_cases, TRG_CHIS),
)

function _generate(generators, selected, category)
    cases = BenchmarkCase[]
    for (name, (gen, sizes)) in pairs(generators)
        name in selected || continue
        for case in gen(sizes)
            @assert case.category === category
            push!(cases, case)
        end
    end
    return cases
end

const CASES = vcat(
    :contract in CATEGORIES ? _generate(CONTRACT_GENERATORS, SOURCES, :contract) : BenchmarkCase[],
    :network in CATEGORIES ? _generate(NETWORK_GENERATORS, TOPICS, :network) : BenchmarkCase[],
)

_nelem(spec::Union{ContractSpec, BatchedContractSpec}, I) = prod((spec.dims[l] for l in I); init = 1)
_nelem_network(spec::NetworkSpec, il) = prod((spec.dims[abs(l)] for l in il); init = 1)

# Upstream's `bytes(spec)` assumes Float64 elements; this is dtype-generic.
function case_bytes(spec::ContractSpec, ::Type{T}) where {T}
    n = _nelem(spec, spec.IA) + _nelem(spec, spec.IB) + _nelem(spec, spec.IC)
    return n * sizeof(T)
end
function case_bytes(spec::BatchedContractSpec, ::Type{T}) where {T}
    n = _nelem(spec, spec.IA) + _nelem(spec, spec.IB) + _nelem(spec, spec.IC)
    return spec.batch * n * sizeof(T)
end

# All input tensors plus the (fresh, `ncon`-allocated) output.
function case_bytes(spec::NetworkSpec, ::Type{T}) where {T}
    n = sum(_nelem_network(spec, il) for il in spec.indexlists)
    n += _nelem_network(spec, spec.output)
    return n * sizeof(T)
end

params_string(params::NamedTuple) =
    join(("$k=$(getfield(params, k))" for k in keys(params)), ";")

# The sweep parameter: `dim` (:contract), `D` (mps) or `chi` (ctmrg/trg).
function case_sweepparam(case::BenchmarkCase)
    p = case.params
    hasproperty(p, :dim) && return p.dim
    hasproperty(p, :D) && return p.D
    hasproperty(p, :chi) && return p.chi
    error("case $(case.category)/$(case.id) has no known sweep-parameter field (dim/D/chi)")
end

# Upstream tags :contract cases by `source` and :network cases by `topic`.
case_group(case::BenchmarkCase) =
    hasproperty(case.params, :source) ? case.params.source :
    hasproperty(case.params, :topic) ? case.params.topic : :none
case_layout(case::BenchmarkCase) =
    hasproperty(case.params, :layout) ? case.params.layout : Symbol("")

# Compact einsum-style label, one letter per distinct label in order of first
# appearance: `[a1,c1,a2,c2]*[c1,c2,b1]` -> "acbd cde>abe" (comma-free for the CSV).
function _compact(labelsets)
    letters = Dict{Any, Char}()
    next = Ref('a')
    tochar(l) = get!(letters, l) do
        c = next[]
        next[] = c == 'z' ? 'A' : c + 1
        c
    end
    return map(ls -> join(tochar.(ls)), labelsets)
end
function case_expr(spec::Union{ContractSpec, BatchedContractSpec})
    a, b, c = _compact((spec.IA, spec.IB, spec.IC))
    prefix = spec isa BatchedContractSpec ? "$(spec.batch)x " : ""
    return "$prefix$a $b>$c"
end
function case_expr(spec::NetworkSpec)
    parts = _compact((abs.(il) for il in (spec.indexlists..., spec.output)))
    return join(parts[1:(end - 1)], " ") * ">" * parts[end]
end

function build_case(spec::ContractSpec, provider, ::Type{T}) where {T}
    dimsA = ntuple(i -> spec.dims[spec.IA[i]], length(spec.IA))
    dimsB = ntuple(i -> spec.dims[spec.IB[i]], length(spec.IB))
    dimsC = ntuple(i -> spec.dims[spec.IC[i]], length(spec.IC))
    A = randtensor(provider, spec.IA, dimsA, T)
    B = randtensor(provider, spec.IB, dimsB, T)
    pA, pB, pAB = TensorOperations.contract_indices(spec.IA, spec.IB, spec.IC)
    return (; A, B, pA, pB, pAB, dimsC)
end

function build_case(spec::BatchedContractSpec, provider, ::Type{T}) where {T}
    dimsA = ntuple(i -> spec.dims[spec.IA[i]], length(spec.IA))
    dimsB = ntuple(i -> spec.dims[spec.IB[i]], length(spec.IB))
    dimsC = ntuple(i -> spec.dims[spec.IC[i]], length(spec.IC))
    As = [randtensor(provider, spec.IA, dimsA, T) for _ in 1:spec.batch]
    Bs = [randtensor(provider, spec.IB, dimsB, T) for _ in 1:spec.batch]
    pA, pB, pAB = TensorOperations.contract_indices(spec.IA, spec.IB, spec.IC)
    return (; As, Bs, pA, pB, pAB, dimsC)
end

function build_case(spec::NetworkSpec, provider, ::Type{T}) where {T}
    tensors = map(spec.indexlists) do il
        dims = ntuple(i -> spec.dims[abs(il[i])], length(il))
        return randtensor(provider, il, dims, T)
    end
    return (; tensors)
end

alloc_output(spec::ContractSpec, ctx, ::Type{T}) where {T} = zeros(T, ctx.dimsC)
alloc_output(spec::BatchedContractSpec, ctx, ::Type{T}) where {T} =
    [zeros(T, ctx.dimsC) for _ in 1:spec.batch]
# `ncon` allocates its own output.
alloc_output(::NetworkSpec, ctx, ::Type{T}) where {T} = nothing

function run_case!(backend, spec::ContractSpec, ctx, C)
    return TensorOperations.tensorcontract!(
        C, ctx.A, ctx.pA, spec.conjA, ctx.B, ctx.pB, spec.conjB, ctx.pAB,
        one(eltype(C)), zero(eltype(C)), backend
    )
end

function run_case!(backend, spec::BatchedContractSpec, ctx, Cs)
    for b in eachindex(Cs)
        TensorOperations.tensorcontract!(
            Cs[b], ctx.As[b], ctx.pA, spec.conjA, ctx.Bs[b], ctx.pB, spec.conjB, ctx.pAB,
            one(eltype(Cs[b])), zero(eltype(Cs[b])), backend
        )
    end
    return Cs
end

function run_case!(backend, spec::NetworkSpec, ctx, C)
    return TensorOperations.ncon(
        ctx.tensors, spec.indexlists, spec.conjlist;
        order = spec.order, output = spec.output, backend = backend
    )
end

# A batched case's result is a vector of per-slice outputs; compare it flat.
_flat(x::AbstractArray{<:Number}) = vec(x)
_flat(xs::AbstractVector{<:AbstractArray}) = reduce(vcat, map(vec, xs))

const OUTDIR = outdir()
const CSV_PATH = joinpath(OUTDIR, "bench_to_suite.csv")
const CANARY_PATH = joinpath(OUTDIR, "canary_to_suite.csv")
const SUMMARY_PATH = joinpath(OUTDIR, "summary_to_suite.txt")
const MISMATCH_PATH = joinpath(OUTDIR, "mismatches_to_suite.txt")
const PROVENANCE_PATH = joinpath(OUTDIR, "PROVENANCE_to_suite.txt")

csv_io = open(CSV_PATH, "w")
println(
    csv_io,
    "backend,dtype,category,case_id,dim,params,reps,median_seconds,gflops,gbytes,min_gflops,std_gflops,",
    "group,layout,blas,intensity,expr"
)
function log_row(backend_name, T, case::BenchmarkCase, reps, t, gf, gb, min_gf, std_gf, intensity)
    println(
        csv_io,
        "$backend_name,$T,$(case.category),$(case.id),$(case_sweepparam(case)),",
        params_string(case.params), ",$reps,",
        @sprintf("%.9f,%.4f,%.4f,%.4f,%.4f", t, gf, gb, min_gf, std_gf), ",",
        case_group(case), ",", case_layout(case), ",", isblasequivalent(case.spec), ",",
        @sprintf("%.4g", intensity), ",", case_expr(case.spec)
    )
    return flush(csv_io)
end

# Like `median_time_s`, but returns every sample. The warm-up call's time sets
# the rep count: up to `reps`, cut to fit `budget` seconds, at least `minreps`.
function timed_samples_s(f!::Function; reps::Int, budget::Float64 = Inf, minreps::Int = 1)
    t0 = time_ns()
    f!()  # warm-up, discarded
    twarm = (time_ns() - t0) / 1.0e9
    n = clamp(floor(Int, budget / max(twarm, 1.0e-9)), min(minreps, reps), reps)
    ts = Vector{Float64}(undef, n)
    for r in 1:n
        t0 = time_ns()
        f!()
        t1 = time_ns()
        ts[r] = (t1 - t0) / 1.0e9
    end
    return ts
end

print_env_header(stdout, "bench_to_suite.jl")
println("cases = ", length(CASES), " per dtype (before per-dtype byte skips)")

# Drift canary: a 64^3 Float64 StridedBLAS @tensor at start/middle/end.
function canary_contract!(backend, C, A, B)
    @tensor backend = backend C[i, j] = A[i, k] * B[k, j]
    return C
end

function run_blas_canary(rng, label::String)
    A = randn(rng, Float64, 64, 64)
    B = randn(rng, Float64, 64, 64)
    C = zeros(Float64, 64, 64)
    t = median_time_s(() -> canary_contract!(StridedBLAS(), C, A, B); reps = 15)
    println("canary[$label] median = $(t) s")
    return t
end

canary_rng = MersenneTwister(0xB3_C4_0003)
canary_results = Float64[]
push!(canary_results, run_blas_canary(canary_rng, "A (start)"))

raw = Vector{NamedTuple}()          # successful timings
mismatches = Vector{NamedTuple}()   # QuasiStrided result != StridedBLAS result
failures = Vector{NamedTuple}()     # a backend threw
skipped = Vector{NamedTuple}()      # over --max-bytes/--max-flops

const MIDPOINT = cld(length(CASES) * length(RUN_DTYPES), 2)
progress = 0

for T in RUN_DTYPES
    provider = ArrayProvider{T}()
    rtol = T === Float64 ? 1.0e-10 : 1.0e-5
    for case in CASES
        global progress += 1
        spec = case.spec
        cb = case_bytes(spec, T)
        fl = flops(spec)
        if cb > MAX_CASE_BYTES || fl > MAX_CASE_FLOPS
            push!(skipped, (dtype = T, category = case.category, id = case.id, bytes = cb, flops = fl))
            @info "skipped (over byte/flop ceiling)" dtype = T id = case.id bytes = cb flops = fl
            continue
        end

        ctx = build_case(spec, provider, T)

        # Correctness gate before any timing: StridedBLAS is the reference,
        # QuasiStrided must match to `rtol` or its timing is skipped.
        results = Dict{Symbol, Any}()
        for (bname, backend) in pairs(BACKENDS)
            C = alloc_output(spec, ctx, T)
            try
                results[bname] = run_case!(backend, spec, ctx, C)
            catch e
                msg = sprint(showerror, e)
                push!(
                    failures,
                    (
                        dtype = T, category = case.category, id = case.id,
                        backend = String(bname), message = first(msg, 400),
                    )
                )
                @warn "backend threw" dtype = T id = case.id backend = bname msg
            end
        end

        ref = get(results, :StridedBLAS, nothing)
        qs = get(results, :QuasiStrided, nothing)
        qs_ok = true
        if ref !== nothing && qs !== nothing
            if !isapprox(_flat(qs), _flat(ref); rtol = rtol)
                nref = norm(_flat(ref))
                disc = nref == 0 ? norm(_flat(qs)) : norm(_flat(qs) - _flat(ref)) / nref
                qs_ok = false
                push!(
                    mismatches,
                    (
                        dtype = T, category = case.category, id = case.id,
                        rtol = rtol, discrepancy = disc,
                    )
                )
                @error "MISMATCH: QuasiStrided != StridedBLAS -- timing skipped" dtype = T id = case.id discrepancy = disc
            end
        end

        for (bname, backend) in pairs(BACKENDS)
            haskey(results, bname) || continue           # threw above
            bname === :QuasiStrided && !qs_ok && continue # mismatched above
            C = alloc_output(spec, ctx, T)
            times = timed_samples_s(
                () -> run_case!(backend, spec, ctx, C);
                reps = REPS, budget = TIME_BUDGET, minreps = MIN_REPS
            )
            t = median(times)
            gf = fl / t / 1.0e9
            gb = cb / t / 1.0e9
            gflops_samples = (fl ./ times) ./ 1.0e9
            min_gf = minimum(gflops_samples)
            std_gf = std(gflops_samples)
            log_row(bname, T, case, length(times), t, gf, gb, min_gf, std_gf, fl / cb)
            push!(
                raw,
                (
                    backend = String(bname), dtype = T, category = case.category,
                    group = case_group(case), layout = case_layout(case),
                    blas = isblasequivalent(spec), expr = case_expr(spec),
                    id = case.id, dim = case_sweepparam(case), t = t, gflops = gf, gbytes = gb,
                    min_gflops = min_gf, std_gflops = std_gf, reps = length(times),
                )
            )
        end

        if progress == MIDPOINT
            push!(canary_results, run_blas_canary(canary_rng, "B (middle)"))
        end
    end
    @info "dtype done" dtype = T rows = length(raw)
end

push!(canary_results, run_blas_canary(canary_rng, "A' (end)"))
close(csv_io)

canary_spread = relative_spread(canary_results)
open(CANARY_PATH, "w") do io
    println(io, "label,median_seconds")
    labels = length(canary_results) == 3 ?
        ("A_start", "B_middle", "Aprime_end") :
        Tuple("c$i" for i in 1:length(canary_results))
    for (lbl, t) in zip(labels, canary_results)
        println(io, "$lbl,", @sprintf("%.9f", t))
    end
    println(io, "# relative spread (max-min)/min = ", @sprintf("%.4f", canary_spread))
end
println("canary spread (max-min)/min = ", @sprintf("%.4f", canary_spread))

const NOISE_FLOOR = max(0.1, canary_spread)

# QS/BLAS time ratio per (dtype, case id); > 1 = QuasiStrided slower.
function qs_ratios(rows)
    out = NamedTuple[]
    for key in unique((r.dtype, r.id) for r in rows)
        rs = filter(r -> (r.dtype, r.id) == key, rows)
        ib = findfirst(r -> r.backend == "StridedBLAS", rs)
        iq = findfirst(r -> r.backend == "QuasiStrided", rs)
        (ib === nothing || iq === nothing) && continue
        b, q = rs[ib], rs[iq]
        push!(
            out, (;
                q.dtype, q.category, q.group, q.layout, q.blas, q.expr, q.id, q.dim,
                ratio = q.t / b.t, t_qs = q.t, t_blas = b.t, gf_qs = q.gflops, gf_blas = b.gflops,
            )
        )
    end
    return out
end

function print_geomeans(io, label, rs)
    isempty(rs) && return
    rv = [r.ratio for r in rs]
    return println(
        io, "  ", rpad(label, 34), @sprintf("geomean QS/BLAS = %6.3f", geomean(rv)),
        @sprintf("   worst %6.3f   n=%3d   slower(>1.1)=%d", maximum(rv), length(rv), count(>(1.1), rv))
    )
end

const RATIOS = qs_ratios(raw)
const TOP_SLOWEST = 40

open(SUMMARY_PATH, "w") do io
    println(io, "# TensorOperations upstream-suite backend benchmark summary")
    print_env_header(io, "bench_to_suite.jl")
    println(
        io, "reps <= ", REPS, " (median; one discarded warm-up; cut to fit ",
        TIME_BUDGET, " s per backend, min ", MIN_REPS, ")"
    )
    println(io, "canary median times (s): ", canary_results)
    println(io, "canary relative spread (max-min)/min: ", @sprintf("%.4f", canary_spread))
    println(
        io,
        "NOISE: read any difference smaller than max(10%, canary spread) = ",
        @sprintf("%.1f%%", 100 * NOISE_FLOOR), " as noise, not as a result."
    )
    if !isempty(mismatches)
        println(io, "\n!! ", length(mismatches), " CORRECTNESS MISMATCH(ES) -- see mismatches_to_suite.txt")
    end
    if !isempty(failures)
        println(io, "!! ", length(failures), " BACKEND REJECTION(S)/ERROR(S) -- see mismatches_to_suite.txt")
    end

    println(io, "\n===== geomean QS/BLAS time ratio (> 1 = QuasiStrided slower) =====")
    for T in RUN_DTYPES
        println(io, "\n", T, ":")
        rT = filter(r -> r.dtype == T, RATIOS)
        for cat in unique(r.category for r in rT), g in unique(r.group for r in rT if r.category == cat)
            rg = filter(r -> r.category == cat && r.group == g, rT)
            print_geomeans(io, "$cat/$g", rg)
            for l in unique(r.layout for r in rg)
                l === Symbol("") && continue
                print_geomeans(io, "    layout=$l", filter(r -> r.layout == l, rg))
            end
            if any(r -> r.blas, rg) && !all(r -> r.blas, rg)
                print_geomeans(io, "    blas-equivalent", filter(r -> r.blas, rg))
                print_geomeans(io, "    not blas-equivalent", filter(r -> !r.blas, rg))
            end
        end
    end

    for T in RUN_DTYPES
        rT = sort(filter(r -> r.dtype == T, RATIOS); by = r -> -r.ratio)
        println(io, "\n===== ", T, ": ", min(TOP_SLOWEST, length(rT)), " slowest cases for QuasiStrided =====")
        println(io, "  ratio   QS GF/s  BLAS GF/s   QS time     group/layout            case  [expr]")
        for r in first(rT, TOP_SLOWEST)
            gl = r.layout === Symbol("") ? string(r.group) : "$(r.group)/$(r.layout)"
            println(
                io, @sprintf("  %6.3f  %7.2f  %8.2f   %.3e  ", r.ratio, r.gf_qs, r.gf_blas, r.t_qs),
                rpad(gl, 24), r.id, "  [", r.expr, "]", r.blas ? "  (blas)" : ""
            )
        end
    end

    println(io, "\n===== per-case detail =====")
    for T in RUN_DTYPES, r in sort(filter(r -> r.dtype == T, RATIOS); by = r -> (string(r.group), r.id))
        println(
            io, "  ", rpad(string(T), 11), rpad("$(r.category)/$(r.id)", 52),
            @sprintf("QS %.4e s  BLAS %.4e s  QS/BLAS %.3f", r.t_qs, r.t_blas, r.ratio)
        )
    end
end
println(read(SUMMARY_PATH, String))

open(MISMATCH_PATH, "w") do io
    println(io, "# Correctness mismatches, backend rejections and skips")
    println(io, "# benchmark/bench_to_suite.jl -- ", gethostname(), " ", now())
    println(io, "\n## QuasiStrided-vs-StridedBLAS mismatches (timing NOT taken for these)")
    if isempty(mismatches)
        println(io, "none found")
    else
        for m in mismatches
            println(
                io, "  MISMATCH ", m.category, "/", m.id, " dtype=", m.dtype,
                " rtol=", m.rtol,
                @sprintf(" norm(diff)/norm(ref)=%.3e", m.discrepancy)
            )
        end
    end
    println(io, "\n## Backend errors / rejections (backend threw; other backends still timed)")
    if isempty(failures)
        println(io, "none found")
    else
        for f in failures
            println(
                io, "  ERROR ", f.category, "/", f.id, " dtype=", f.dtype,
                " backend=", f.backend, ": ", replace(f.message, "\n" => " | ")
            )
        end
    end
    println(io, "\n## Cases skipped by --max-bytes=", MAX_CASE_BYTES, " / --max-flops=", MAX_CASE_FLOPS)
    if isempty(skipped)
        println(io, "none (upstream's own within_memory_budget, Sys.total_memory() ÷ 64, ran first in the generators)")
    else
        for s in skipped
            println(io, "  SKIP ", s.category, "/", s.id, " dtype=", s.dtype, " bytes=", s.bytes, " flops=", s.flops)
        end
    end
end
println(read(MISMATCH_PATH, String))

to_version, tob_rev = try
    deps = Pkg.dependencies()
    tov = tobr = "unknown"
    for (_, info) in deps
        if info.name == "TensorOperations"
            tov = string(info.version)
        elseif info.name == "TensorOperationsBenchmarks"
            tobr = string(
                something(info.git_revision, "?"), " (tree ",
                something(info.tree_hash, "?"), ")"
            )
        end
    end
    tov, tobr
catch e
    "unknown ($(sprint(showerror, e)))", "unknown"
end

machine_load = try
    strip(read(`uptime`, String))
catch
    "unknown (uptime failed)"
end
top_procs = try
    strip(read(pipeline(`ps -eo pcpu,comm --sort=-pcpu`, `head -6`), String))
catch
    "unknown (ps failed)"
end

open(PROVENANCE_PATH, "w") do io
    println(io, "command = julia --project=benchmark benchmark/bench_to_suite.jl ", join(ARGS, " "))
    print_env_header(io, "bench_to_suite.jl")
    println(io, "logical_cpus = ", Sys.CPU_THREADS)
    println(io, "blas_config = ", LinearAlgebra.BLAS.get_config())
    println(io, "TensorOperations = ", to_version)
    println(io, "TensorOperationsBenchmarks = ", tob_rev)
    println(io, "backends = ", collect(keys(BACKENDS)), " (QuasiStrided = QuasiStridedBackend() directly)")
    println(io, "dtypes = ", collect(RUN_DTYPES))
    println(io, "reps <= ", REPS, " (median, one discarded warm-up, time budget ", TIME_BUDGET, " s, min ", MIN_REPS, ")")
    println(io, "max_bytes = ", MAX_CASE_BYTES, "  max_flops = ", MAX_CASE_FLOPS)
    println(io, "categories = ", CATEGORIES)
    println(io, "sources = ", SOURCES, "  topics = ", TOPICS)
    println(io, "synthetic sizes = ", SYNTHETIC_SIZES)
    println(io, "tccg sizes = ", TCCG_SIZES)
    println(io, "batched sizes = ", BATCHED_SIZES)
    println(io, "mps bonddims = ", MPS_BONDDIMS)
    println(io, "ctmrg chis = ", CTMRG_CHIS)
    println(io, "trg chis = ", TRG_CHIS)
    println(io, "cases generated = ", length(CASES), " per dtype")
    println(io, "cases actually timed = ", length(raw), " backend-rows total")
    println(io, "case list = ")
    for case in CASES
        println(io, "  ", case.category, "/", case.id, "  ", params_string(case.params))
    end
    println(io, "mismatches = ", length(mismatches), " (see mismatches_to_suite.txt)")
    for m in mismatches
        println(
            io, "  ", m.category, "/", m.id, " dtype=", m.dtype,
            @sprintf(" norm(diff)/norm(ref)=%.3e", m.discrepancy)
        )
    end
    println(io, "backend_rejections = ", length(failures), " (see mismatches_to_suite.txt)")
    for f in failures
        println(io, "  ", f.category, "/", f.id, " dtype=", f.dtype, " backend=", f.backend)
    end
    println(io, "skipped_cases = ", length(skipped))
    println(io, "canary_medians_s = ", canary_results)
    println(io, "canary_relative_spread = ", @sprintf("%.4f", canary_spread))
    println(io, "noise_floor_used = ", @sprintf("%.4f", NOISE_FLOOR))
    println(io, "machine_load_at_run = ", machine_load)
    println(io, "top_processes_at_run =")
    for line in split(top_procs, '\n')
        println(io, "  ", line)
    end
    println(io, "caveat = single machine ($(gethostname())), single measurement session.")
end

println("\nDone. Results in ", OUTDIR)
