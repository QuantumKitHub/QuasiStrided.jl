# (mc, kc, nc) plateau sweep: the analytical blocking model and the fallback
# row against a log-spaced grid, at the engine's default kernel.
#
#   julia --project=benchmark benchmark/bench_blocking_model.jl [--reps 21] [--smoke] [--outdir DIR]
#
# Per real dtype: the named rows (`model` = `_modelled_blocking`, `fallback` =
# `_fallback_blocking`) and the `wide` grid on every harness shape, plus 1-D
# `nc` and `mc` slices through the model row on a 2048x256x2048 shape (the only
# one large enough for `nc` to bind). Per complex dtype: the named rows scaled
# by `_scale_blocking`. Every configuration of a shape is timed back to back.
# The score of a configuration is the geomean over shapes of its time over the
# best time at that shape. `--smoke` runs 1 rep on a few points.

include(joinpath(@__DIR__, "harness.jl"))

using QuasiStrided: Blocking, target_profile, _default_kernel, _modelled_blocking,
    _fallback_blocking, _scale_blocking, complex_method

const SMOKE = hasflag("smoke")
const REPS = SMOKE ? 1 : argopt("reps", 21)
thin(v) = SMOKE ? v[1:min(end, 3)] : v

const GRID_SHAPES = vcat(MAIN_SHAPES, EXTRA_SHAPES, SMALL_SHAPES)
const BIG_SHAPE = ShapeSpec(2048, 256, 2048)
const WIDE = Dict(
    Float64 => full_grid((48, 96, 192, 384, 768), (64, 128, 192, 256, 384, 512, 768, 1024), (192, 768, 3072)),
    Float32 => full_grid((48, 96, 192, 384, 768), (128, 256, 384, 512, 768, 1024, 1536, 2048), (192, 768, 3072)),
)
const NSLICE = (48, 96, 192, 384, 768, 1536, 3072, 6144)
const MSLICE = (24, 48, 96, 192, 384, 768, 1536)

const PROFILE = target_profile()
const OUTDIR = outdir()
const CSV_PATH = joinpath(OUTDIR, "blocking_model.csv")
const SUMMARY_PATH = joinpath(OUTDIR, "summary_blocking_model.txt")

tup(b::Blocking) = (b.mc, b.kc, b.nc)
named_points(rows) = [(string(k), tup(v)) for (k, v) in pairs(rows) if v !== nothing]
named_rows(::Type{T}) where {T <: Real} = (model = _modelled_blocking(PROFILE, T), fallback = _fallback_blocking(T))

csv = open(CSV_PATH, "w")
println(csv, "set,dtype,shape,Ma,Ka,Na,mc,kc,nc,mc_eff,kc_eff,nc_eff,reps,median_seconds,gflops")

function sweep_shape!(raw, kernel, ::Type{T}, spec, points) where {T}
    fx = build_plain(T, spec, MersenneTwister(0xB10C))
    for (set, (mc, kc, nc)) in points
        plan = plan_contract(
            fx.Cv, fx.Av, fx.indA, fx.Bv, fx.indB, fx.indC; kernel = kernel, mc = mc, kc = kc, nc = nc
        )
        t = median_time_s(() -> execute!(plan, one(T), zero(T)); reps = REPS)
        b = plan.blocking
        println(
            csv, "$set,$T,$(spec.name),$(spec.Ma),$(spec.Ka),$(spec.Na),$mc,$kc,$nc,$(b.mc),$(b.kc),$(b.nc),$REPS,",
            @sprintf("%.9f,%.4f", t, gflops(T, spec.Ma, spec.Ka, spec.Na, t))
        )
        flush(csv)
        push!(raw, (; set, dtype = T, shape = spec.name, mc, kc, nc, t))
    end
    return nothing
end

# Geomean over `shapes` of each configuration's time over the best time any
# configuration reached at that shape; `key` groups rows into configurations.
function score(raw, T, shapes, key; sets = nothing)
    rows = filter(r -> r.dtype == T && r.shape in shapes, raw)
    best = Dict{String, Float64}()
    for r in rows
        best[r.shape] = min(get(best, r.shape, Inf), r.t)
    end
    g = Dict{Any, Vector{Float64}}()
    for r in rows
        (sets === nothing || r.set in sets) && push!(get!(g, key(r), Float64[]), r.t / best[r.shape])
    end
    return sort([(k, geomean(v)) for (k, v) in g]; by = last)
end

function summarize(io, raw, canaries)
    print_env_header(io, "bench_blocking_model.jl")
    println(io, "target = ", PROFILE, "\nreps = ", REPS)
    @printf(io, "canary spread = %.1f%%  %s\n", 100 * relative_spread(canaries), canaries)
    grid_names = [s.name for s in GRID_SHAPES]
    for T in DTYPES
        println(io, "\n## $T   ", named_rows(T))
        w = score(raw, T, grid_names, r -> (r.mc, r.kc, r.nc); sets = ("wide",))
        @printf(
            io, "wide grid: best %.4f %s  worst %.4f %s  spread %.1f%%\n",
            w[1][2], w[1][1], w[end][2], w[end][1], 100 * (w[end][2] / w[1][2] - 1)
        )
        within(tol) = count(x -> x[2] <= w[1][2] * (1 + tol), w)
        println(io, "points within 3%/6%/10% of best: ", within(0.03), "/", within(0.06), "/", within(0.1), " of ", length(w))
        println(io, "top 5: ", w[1:min(5, end)])
        for (set, g) in score(raw, T, grid_names, r -> r.set; sets = ("model", "fallback"))
            @printf(io, "  %-10s %.4f\n", set, g)
        end
        for (s, f) in (("nslice", r -> r.nc), ("mslice", r -> r.mc))
            pts = sort([r for r in raw if r.dtype == T && r.set == s]; by = f)
            isempty(pts) && continue
            tb = minimum(r.t for r in pts)
            println(io, "  $s @ $(BIG_SHAPE.name): ", join([@sprintf("%d:%.3f", f(r), r.t / tb) for r in pts], "  "))
        end
    end
    for T in CDTYPES
        println(io, "\n## $T (scaled named rows)")
        for (set, g) in score(raw, T, [s.name for s in MAIN_SHAPES], r -> r.set)
            @printf(io, "  %-10s %.4f\n", set, g)
        end
    end
    return nothing
end

print_env_header(stdout, "bench_blocking_model.jl")
println("target = ", PROFILE)
raw = NamedTuple[]
crng = MersenneTwister(0x0CA9A121)
canaries = [run_canary(crng, "start")]
for T in DTYPES
    kernel = _default_kernel(T)
    rows = named_rows(T)
    println("\n$T kernel $(mr(kernel))x$(nr(kernel))/W$(lanewidth(kernel))  ", rows)
    rows.model === nothing && @warn "no cache geometry detected: model undefined"
    grid = [("wide", c) for c in thin(WIDE[T])]
    for spec in thin(GRID_SHAPES)
        sweep_shape!(raw, kernel, T, spec, vcat(named_points(rows), grid))
    end
    m = something(rows.model, rows.fallback)
    slices = vcat(
        named_points(rows), [("nslice", (m.mc, m.kc, n)) for n in thin(NSLICE)],
        [("mslice", (x, m.kc, m.nc)) for x in thin(MSLICE)],
    )
    sweep_shape!(raw, kernel, T, BIG_SHAPE, slices)
    push!(canaries, run_canary(crng, "after-$T"))
end
for T in CDTYPES
    kernel = _default_kernel(T)
    rows = map(v -> v === nothing ? nothing : _scale_blocking(v, complex_method(kernel)), named_rows(real(T)))
    println("\n$T kernel $(mr(kernel))x$(nr(kernel))/W$(lanewidth(kernel))  ", rows)
    for spec in thin(MAIN_SHAPES)
        sweep_shape!(raw, kernel, T, spec, named_points(rows))
    end
end
push!(canaries, run_canary(crng, "end"))
close(csv)

open(io -> summarize(io, raw, canaries), SUMMARY_PATH, "w")
print(read(SUMMARY_PATH, String))
relative_spread(canaries) > 0.1 && @warn "canary spread exceeds 10%: re-run before believing any ranking."
println("wrote ", CSV_PATH)
