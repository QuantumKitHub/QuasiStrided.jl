# Large-GEMM peak: the engine's default real kernel against every real
# SIMDKernel menu shape and against OpenBLAS, on the compute-bound regime the
# 2026-09-26 AVX-512 register-tile change targets (`_rule_mv`,
# src/planning/kernel_selection.jl).
#
#   julia --project=benchmark benchmark/bench_large_gemm_peak.jl [--outdir DIR] [--sizes 2048,3969] [--rounds 3]
#
# Four arms, Float64 and Float32, single core, interleaved rounds (median):
#
#   kernel   kernel-only ceiling per menu shape: `unsafe_execute_tile!` on one
#            L1-resident packed A sliver and B sliver at the model's `kc`,
#            i.e. flops/cycle of the register tile with nothing around it.
#   engine   `plan_contract`/`execute!` at n^3 for every menu shape (caller-named
#            kernel, the engine's own blocking) and for the DEFAULT kernel, plus
#            OpenBLAS `mul!` on the same operands.
#   stepdown the short-M boundary of `_extent_shape`: M in a sweep around one
#            and two tall tiles at K = N = 1024, default kernel vs the tall and
#            half-height shapes named explicitly.
#   blocking the default kernel at n^3 over a small (mc, kc) grid around the
#            model row (an L1/2 vs 3L1/4 `kc` check; the model's `nc`).
#
# Writes bench_large_gemm_peak.csv and summary_large_gemm_peak.txt to --outdir
# (default benchmark/results/<host>-<date>).

include(joinpath(@__DIR__, "harness.jl"))

using QuasiStrided: SIMDKernel, plan_contract, execute!, default_blocking, target_profile,
    _derived_shape, _default_kernel, kernel_shapes, packed_panel, unsafe_execute_tile!,
    DestinationTile, AffineAxis, _modelled_blocking

const OUTDIR = let d = argopt("outdir", "")
    isempty(d) ? results_dir() : d
end
mkpath(OUTDIR)
const SIZES = parse_ints(argopt("sizes", "2048,3969"))
const ROUNDS = argopt("rounds", 3)
const REAL_DTYPES = (Float64, Float32)

const CSV_PATH = joinpath(OUTDIR, "bench_large_gemm_peak.csv")
const SUMMARY_PATH = joinpath(OUTDIR, "summary_large_gemm_peak.txt")

csv_io = open(CSV_PATH, "w")
println(csv_io, "arm,dtype,case,shape,M,K,N,mc,kc,nc,round,seconds,gflops")
function log_row(arm, T, case, shape, M, K, N, blk, round, t)
    gf = 2.0 * M * K * N / t / 1.0e9
    println(
        csv_io, "$arm,$T,$case,$(shape),$M,$K,$N,$(blk === nothing ? "" : blk.mc),",
        "$(blk === nothing ? "" : blk.kc),$(blk === nothing ? "" : blk.nc),$round,",
        @sprintf("%.9f", t), ",", @sprintf("%.3f", gf)
    )
    flush(csv_io)
    return gf
end

shapestr(sh) = "$(sh[1])x$(sh[2])/W$(sh[3])"
shape_of(k) = (mr(k), nr(k), lanewidth(k))

rows = NamedTuple[]
function record!(arm, T, case, shape, M, K, N, blk, round, t)
    gf = log_row(arm, T, case, shape, M, K, N, blk, round, t)
    push!(rows, (arm = arm, dtype = T, case = case, shape = shape, gf = gf))
    return nothing
end

println("bench_large_gemm_peak.jl  cpu = ", Sys.CPU_NAME, "  profile = ", target_profile())
for T in REAL_DTYPES
    println(
        "  ", T, ": derived shape ", _derived_shape(target_profile(), T), "  default kernel ",
        shapestr(shape_of(_default_kernel(T))), "  blocking ", default_blocking(_default_kernel(T))
    )
end

# ---------------------------------------------------------------------------
# kernel arm
# ---------------------------------------------------------------------------

function kernel_hot!(kernel, C, apack, bpack, kc, reps)
    MR = mr(kernel); NR = nr(kernel)
    GC.@preserve apack bpack begin
        ap = packed_panel(apack, 1, length(apack)); bp = packed_panel(bpack, 1, length(bpack))
        dest = DestinationTile(C, 0, AffineAxis(0, 1, MR), AffineAxis(0, MR, NR))
        for _ in 1:reps
            unsafe_execute_tile!(kernel, dest, ap, bp, kc, one(eltype(C)), one(eltype(C)))
        end
    end
    return nothing
end

for T in REAL_DTYPES
    fixtures = []
    for sh in kernel_shapes(T)
        MR, NR, W = sh
        kernel = SIMDKernel(Val(MR), Val(NR), T, Val(W))
        kc = default_blocking(kernel).kc
        apack = rand(T, MR * kc); bpack = rand(T, NR * kc); C = zeros(T, MR * NR)
        kernel_hot!(kernel, C, apack, bpack, kc, 10)
        t1 = @elapsed kernel_hot!(kernel, C, apack, bpack, kc, 1000)
        reps = max(100, round(Int, 0.5 / t1 * 1000))
        push!(fixtures, (sh, kernel, C, apack, bpack, kc, reps))
    end
    for r in 1:ROUNDS, (sh, kernel, C, apack, bpack, kc, reps) in fixtures
        t = @elapsed kernel_hot!(kernel, C, apack, bpack, kc, reps)
        record!("kernel", T, "l1_panels", shapestr(sh), sh[1], kc * reps, sh[2], nothing, r, t)
    end
end
println("kernel arm done")

# ---------------------------------------------------------------------------
# engine arm
# ---------------------------------------------------------------------------

for T in REAL_DTYPES, n in SIZES
    rng = MersenneTwister(0x1a2b + n)
    A = randn(rng, T, n, n); B = randn(rng, T, n, n); C = zeros(T, n, n)
    plans = []
    for sh in kernel_shapes(T)
        kernel = SIMDKernel(Val(sh[1]), Val(sh[2]), T, Val(sh[3]))
        plan = plan_contract(StridedView(C), StridedView(A), (1, 2), StridedView(B), (2, 3), (1, 3); kernel = kernel)
        execute!(plan, one(T), zero(T))
        push!(plans, (shapestr(sh), plan))
    end
    pdef = plan_contract(StridedView(C), StridedView(A), (1, 2), StridedView(B), (2, 3), (1, 3))
    execute!(pdef, one(T), zero(T))
    mul!(C, A, B)
    for r in 1:ROUNDS
        for (name, plan) in plans
            t = @elapsed execute!(plan, one(T), zero(T))
            record!("engine", T, "n$n", name, n, n, n, plan.blocking, r, t)
        end
        t = @elapsed execute!(pdef, one(T), zero(T))
        record!("engine", T, "n$n", "default:" * shapestr(shape_of(pdef.kernel)), n, n, n, pdef.blocking, r, t)
        t = @elapsed mul!(C, A, B)
        record!("engine", T, "n$n", "openblas", n, n, n, nothing, r, t)
    end
    @info "engine arm" T n
end

# ---------------------------------------------------------------------------
# stepdown arm: M around one and two tall tiles, K = N = 1024
# ---------------------------------------------------------------------------

for T in REAL_DTYPES
    tall = _derived_shape(target_profile(), T)
    MR, NR, W = tall
    half = (MR ÷ 2, NR, W)
    half in kernel_shapes(T) || continue
    K = N = 1024
    Ms = unique([MR ÷ 2, MR - 1, MR, MR + 1, MR + W, MR + 2W, MR + 2W + 1, 2MR - 1, 2MR, 2MR + W, 3MR + W])
    for M in Ms
        rng = MersenneTwister(0x5d + M)
        A = randn(rng, T, M, K); B = randn(rng, T, K, N); C = zeros(T, M, N)
        plans = []
        for (label, sh) in (("tall", tall), ("half", half))
            kernel = SIMDKernel(Val(sh[1]), Val(sh[2]), T, Val(sh[3]))
            plan = plan_contract(StridedView(C), StridedView(A), (1, 2), StridedView(B), (2, 3), (1, 3); kernel = kernel)
            execute!(plan, one(T), zero(T))
            push!(plans, (label * ":" * shapestr(sh), plan))
        end
        pdef = plan_contract(StridedView(C), StridedView(A), (1, 2), StridedView(B), (2, 3), (1, 3))
        execute!(pdef, one(T), zero(T))
        push!(plans, ("default:" * shapestr(shape_of(pdef.kernel)), pdef))
        reps = max(3, round(Int, 0.2 / (2.0 * M * K * N / 30.0e9)))
        for r in 1:ROUNDS, (name, plan) in plans
            t = @elapsed for _ in 1:reps
                execute!(plan, one(T), zero(T))
            end
            record!("stepdown", T, "M$M", name, M, K * reps, N, plan.blocking, r, t)
        end
    end
    @info "stepdown arm" T
end

# ---------------------------------------------------------------------------
# blocking arm: default kernel, (mc, kc) grid around the model row
# ---------------------------------------------------------------------------

for T in REAL_DTYPES
    n = maximum(SIZES)
    rng = MersenneTwister(0x77)
    A = randn(rng, T, n, n); B = randn(rng, T, n, n); C = zeros(T, n, n)
    kernel = _default_kernel(T)
    MR = mr(kernel)
    model = default_blocking(kernel)
    kc34 = max(1, (target_profile().l1d.bytes * 3 ÷ 4) ÷ (nr(kernel) * sizeof(T)))
    grid = unique(
        [
            (model.mc, model.kc), (model.mc ÷ 2 ÷ MR * MR, model.kc), (model.mc * 2, model.kc),
            (max(MR, model.mc * model.kc ÷ kc34 ÷ MR * MR), kc34), (model.mc, kc34),
            (max(MR, model.mc ÷ 2 ÷ MR * MR), 2 * model.kc),
        ]
    )
    plans = []
    for (mc, kc) in grid
        (mc >= MR && kc >= 1) || continue
        plan = plan_contract(StridedView(C), StridedView(A), (1, 2), StridedView(B), (2, 3), (1, 3); kernel = kernel, mc = mc, kc = kc, nc = model.nc)
        execute!(plan, one(T), zero(T))
        push!(plans, ("mc$(mc)_kc$(kc)", plan))
    end
    for r in 1:ROUNDS, (name, plan) in plans
        t = @elapsed execute!(plan, one(T), zero(T))
        record!("blocking", T, "n$n", name, n, n, n, plan.blocking, r, t)
    end
    @info "blocking arm" T
end
close(csv_io)

# ---------------------------------------------------------------------------
# summary: medians per (arm, dtype, case, shape)
# ---------------------------------------------------------------------------

open(SUMMARY_PATH, "w") do io
    print_env_header(io, "bench_large_gemm_peak.jl")
    println(io, "profile = ", target_profile())
    for T in REAL_DTYPES
        println(
            io, T, ": derived shape ", _derived_shape(target_profile(), T),
            "  default kernel ", shapestr(shape_of(_default_kernel(T))),
            "  blocking ", default_blocking(_default_kernel(T))
        )
    end
    keys_ = unique([(r.arm, r.dtype, r.case, r.shape) for r in rows])
    lastarm = ""
    for key in keys_
        arm, T, case, shape = key
        if arm != lastarm
            println(io, "\n== ", arm, " (median GF/s over ", ROUNDS, " rounds; min..max) ==")
            lastarm = arm
        end
        v = [r.gf for r in rows if (r.arm, r.dtype, r.case, r.shape) == key]
        println(
            io, "  ", rpad(string(T), 8), rpad(case, 12), rpad(shape, 22),
            @sprintf("%7.1f", median(v)), "  (", @sprintf("%.1f", minimum(v)), "..", @sprintf("%.1f", maximum(v)), ")"
        )
    end
end
println(read(SUMMARY_PATH, String))
println("Done. Results in ", OUTDIR)
