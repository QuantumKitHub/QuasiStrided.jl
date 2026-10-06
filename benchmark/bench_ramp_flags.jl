# Same-node comparison of the nest with and without its static ramp flags,
# interleaved per case: `on` is the plan as built; `off` is the same plan with
# every `NestPath` ramp flag cleared, so every sliver axis takes the runtime
# affine/scatter choice and ramp composites go through the offset buffers. `off`
# is an upper bound on what dropping the flags costs: without them the ramp
# descriptors could still be computed in closed form at runtime.
#
#   julia --project=benchmark benchmark/bench_ramp_flags.jl
#       [--dtypes Float64,ComplexF64] [--rounds 3] [--reps 5] [--outdir DIR]

include(joinpath(@__DIR__, "harness.jl"))

using QuasiStrided: NestPath, ContractPlan

const RUN_DTYPES = parse_dtypes(argopt("dtypes", "Float64,ComplexF64"))
const ROUNDS = argopt("rounds", 3)
const REPS = argopt("reps", 5)
const OUTDIR = outdir()
const CSV_PATH = joinpath(OUTDIR, "bench_ramp_flags.csv")

# (name, size of each label, indA, indB, indC)
const CASES = [
    ("matmul_24^3", (24, 24, 24), (1, 2), (2, 3), (1, 3)),
    ("matmul_64^3", (64, 64, 64), (1, 2), (2, 3), (1, 3)),
    ("matmul_512^3", (512, 512, 512), (1, 2), (2, 3), (1, 3)),
    ("smallM_12x512x512", (12, 512, 512), (1, 2), (2, 3), (1, 3)),
    # Composites that are ramps in some maps only.
    ("partial_32", (32, 32, 32, 32), (1, 3, 4), (4, 2), (1, 2, 3)),
    ("ccsd_t_16", (16, 16, 16, 16, 16, 16, 16), (7, 4, 5, 6), (1, 2, 3, 7), (1, 2, 3, 4, 5, 6)),
]

# Median over `REPS` batches, each long enough (>= 1 ms) to time reliably.
function batch_time_s(f!)
    f!()
    n = max(1, ceil(Int, 1.0e-3 / max(@elapsed(f!()), 1.0e-9)))
    return median_time_s(
        () -> (
            for _ in 1:n
                f!()
            end
        ); reps = REPS
    ) / n
end

operand(T, sizes, ind, rng) = randn(rng, T, map(l -> sizes[l], ind)...)

function without_ramp_flags(plan)
    plan.path isa NestPath || return nothing
    U, _, S = typeof(plan.path).parameters
    return ContractPlan(plan; path = NestPath{U, ntuple(_ -> false, 6), S}())
end

function run(io)
    print_env_header(stdout, "bench_ramp_flags.jl")
    println(io, "dtype,case,path,round,t_on,t_off")
    rng = MersenneTwister(1)
    for T in RUN_DTYPES, (name, sizes, indA, indB, indC) in CASES
        A = operand(T, sizes, indA, rng)
        B = operand(T, sizes, indB, rng)
        C = operand(T, sizes, indC, rng)
        on = plan_contract(StridedView(C), StridedView(A), indA, StridedView(B), indB, indC)
        off = without_ramp_flags(on)
        if off === nothing || off.path === on.path
            println("skip $T $name: path $(typeof(on.path)) has no ramp flags set")
            continue
        end
        for r in 1:ROUNDS
            t_on = batch_time_s(() -> execute!(on, one(T), zero(T)))
            t_off = batch_time_s(() -> execute!(off, one(T), zero(T)))
            println(io, join((T, name, repr(string(typeof(on.path))), r, t_on, t_off), ','))
            flush(io)
            @printf("%s %s round %d: on %.4g s, off/on %.3f\n", T, name, r, t_on, t_off / t_on)
        end
    end
    return nothing
end

open(run, CSV_PATH, "w")
println("wrote ", CSV_PATH)
