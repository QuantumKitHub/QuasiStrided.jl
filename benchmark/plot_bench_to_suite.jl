# Overview plot of bench_to_suite.jl's CSV: one row per dtype, columns
# sharing the arithmetic-intensity axis: QuasiStrided GFLOP/s, StridedBLAS
# GFLOP/s (same scale) and the QuasiStrided/StridedBLAS time ratio (< 1 =
# QuasiStrided faster, with the geomean and faster count). Colour = total
# work, marker = category.
#
#   julia --project=benchmark benchmark/plot_bench_to_suite.jl [csv_path]
#       [--dtypes ComplexF64] [--categories network] [--tags source=tccg,topic=mps]
#
# Without `csv_path`, uses the newest benchmark/results/*/bench_to_suite.csv.
# Without `--dtypes`, plots whichever of Float64 and ComplexF64 the CSV has,
# or every dtype it has if neither.
# `--tags` selects on the case params: a case is kept when it carries at least
# one of the named keys and matches one of the values given for each key it
# carries, so `source=tccg,topic=mps` keeps the TCCG and the MPS cases. Writes
# bench_to_suite_overview[_<filters>].png next to the CSV.

using CairoMakie
using Printf

include(joinpath(@__DIR__, "harness.jl"))

function latest_csv()
    root = joinpath(@__DIR__, "results")
    candidates = filter(isfile, [joinpath(d, "bench_to_suite.csv") for d in readdir(root; join = true)])
    isempty(candidates) && error("no benchmark/results/*/bench_to_suite.csv found")
    return candidates[argmax(mtime.(candidates))]
end

# The positional CSV path is the first argument that is neither an option nor an option's value.
const CSV_PATH = let i = findfirst(i -> !startswith(ARGS[i], "--") && (i == 1 || !startswith(ARGS[i - 1], "--")), eachindex(ARGS))
    i === nothing ? latest_csv() : ARGS[i]
end
splitlist(s) = s === nothing ? nothing : String.(strip.(split(s, ',')))
const DTYPES_FILTER = splitlist(argval("dtypes"))
const CATEGORIES_FILTER = splitlist(argval("categories"))
const TAGS_FILTER = let t = splitlist(argval("tags"))
    t === nothing ? nothing : [Pair(String.(split(kv, '='; limit = 2))...) for kv in t]
end

# "k1=v1;k2=v2;..." as written by bench_to_suite.jl.
parse_params(s) = isempty(s) ? Dict{String, String}() : Dict(Pair(String.(split(kv, '='; limit = 2))...) for kv in split(s, ';'))

function read_rows(path)
    lines = readlines(path)
    header = split(lines[1], ',')
    "expr" in header || error("$path is not a bench_to_suite.jl CSV (no expr column)")
    col = Dict(h => i for (i, h) in enumerate(header))
    return map(lines[2:end]) do line
        f = split(line, ',')
        t = parse(Float64, f[col["median_seconds"]])
        gf = parse(Float64, f[col["gflops"]])
        (
            backend = f[col["backend"]], dtype = f[col["dtype"]], category = f[col["category"]],
            id = f[col["case_id"]], params = parse_params(f[col["params"]]), t = t, gflops = gf,
            flops = gf * t * 1.0e9, intensity = parse(Float64, f[col["intensity"]]),
        )
    end
end

function tags_match(params)
    TAGS_FILTER === nothing && return true
    keys_present = filter(k -> haskey(params, k), unique(first.(TAGS_FILTER)))
    return !isempty(keys_present) &&
        all(k -> any(kv -> kv == (k => params[k]), TAGS_FILTER), keys_present)
end

const ALL_ROWS = read_rows(CSV_PATH)
const PLOT_DTYPES = something(
    DTYPES_FILTER, let present = unique(r.dtype for r in ALL_ROWS)
        default = filter(in(present), ["Float64", "ComplexF64"])
        isempty(default) ? present : default
    end
)

keep(r) = r.dtype in PLOT_DTYPES && (CATEGORIES_FILTER === nothing || r.category in CATEGORIES_FILTER) &&
    tags_match(r.params)

const ROWS = filter(keep, ALL_ROWS)
isempty(ROWS) && error("no rows left after filtering $CSV_PATH")

const SUFFIX = join(
    vcat(
        something(DTYPES_FILTER, String[]), something(CATEGORIES_FILTER, String[]),
        TAGS_FILTER === nothing ? String[] : ["$(k)-$(v)" for (k, v) in TAGS_FILTER],
    ), "_"
)
outpath(name) = joinpath(dirname(CSV_PATH), "bench_to_suite_$(name)$(isempty(SUFFIX) ? "" : "_" * SUFFIX).png")

# One entry per (dtype, category, case) timed by both backends.
const PAIRS = let byid = Dict{Tuple{String, String, String}, Dict{String, Any}}()
    for r in ROWS
        get!(byid, (r.dtype, r.category, r.id), Dict{String, Any}())[r.backend] = r
    end
    [
        (dtype = k[1], category = k[2], id = k[3], qs = v["QuasiStrided"], blas = v["StridedBLAS"])
            for (k, v) in byid if haskey(v, "QuasiStrided") && haskey(v, "StridedBLAS")
    ]
end
const DTYPE_ORDER = ["Float32", "Float64", "ComplexF32", "ComplexF64"]
const PANEL_DTYPES = sort(unique(r.dtype for r in ROWS); by = d -> something(findfirst(==(d), DTYPE_ORDER), 99))

# batlow without its darkest and palest ends, so both ends stay visible on white.
const CMAP = cgrad(:batlow)[0.08:0.01:0.85]
const GREY = RGBf(0.45, 0.45, 0.45)
decade_ticks(ms) = [m * 10.0^e for e in -3:4 for m in ms]
const LOG_TICKS = decade_ticks((1, 2, 5))
# Log-axis ticks with plain labels, as many per decade as the span allows.
log_ticks(lo, hi) = hi / lo > 1000 ? decade_ticks((1,)) : hi / lo > 10 ? LOG_TICKS : decade_ticks((1, 1.5, 2, 3, 5, 7))
# Ratios within a decade of each other get the steps in between.
ratio_ticks(lo, hi) = hi / lo < 10 ? sort(vcat(LOG_TICKS, [0.6, 0.7, 0.8, 0.9, 1.1, 1.25, 1.5, 3])) : LOG_TICKS
plainticks(vs) = [@sprintf("%g", v) for v in vs]
const TITLE = "$(basename(dirname(CSV_PATH)))$(isempty(SUFFIX) ? "" : "  [" * replace(SUFFIX, "_" => ", ") * "]")"
const MARKERS = (contract = (marker = :circle, markersize = 10), network = (marker = :utriangle, markersize = 13))
const CATEGORIES = filter(c -> any(r -> r.category == c, ROWS), ["contract", "network"])

function points!(ax, items, x, y, crange)
    for cat in CATEGORIES
        cs = filter(it -> it.category == cat, items)
        isempty(cs) && continue
        scatter!(
            ax, [Point2f(x(it), y(it)) for it in cs]; MARKERS[Symbol(cat)]...,
            color = [log10(work(it)) for it in cs], colormap = CMAP, colorrange = crange,
            alpha = 0.65, strokewidth = 0.4, strokecolor = (:black, 0.5)
        )
    end
    return
end
work(r) = hasproperty(r, :flops) ? r.flops : r.blas.flops

function overview_figure()
    nrow = length(PANEL_DTYPES)
    fig = Figure(size = (1700, 430 * nrow + 170))
    Label(fig[0, 1:3], TITLE; fontsize = 13, color = GREY)
    crange = extrema(log10(r.flops) for r in ROWS)
    xticks = log_ticks(extrema(r.intensity for r in ROWS)...)
    axes = Axis[]
    for (i, dtype) in enumerate(PANEL_DTYPES)
        last = i == nrow
        xlabel = last ? "arithmetic intensity (flop/byte)" : ""
        rate_axes = map(enumerate(("QuasiStrided", "StridedBLAS"))) do (j, backend)
            ax = Axis(
                fig[i, j]; xscale = log10, yscale = log10, xticks, xtickformat = plainticks, xlabel,
                yticks = log_ticks(extrema(r.gflops for r in ROWS if r.dtype == dtype)...), ytickformat = plainticks,
                title = i == 1 ? backend : "", titlesize = 16, ylabel = j == 1 ? "throughput (GFLOP/s)" : ""
            )
            points!(ax, filter(r -> r.dtype == dtype && r.backend == backend, ROWS), r -> r.intensity, r -> r.gflops, crange)
            ax
        end
        linkyaxes!(rate_axes...)
        ps = filter(p -> p.dtype == dtype, PAIRS)
        rv = [p.qs.t / p.blas.t for p in ps]
        ylim = isempty(ps) ? (0.5, 2.0) : extrema(rv) .* (0.8, 1.25)
        ax = Axis(
            fig[i, 3]; xscale = log10, yscale = log10, xticks, xtickformat = plainticks, xlabel,
            yticks = ratio_ticks(ylim...), ytickformat = plainticks,
            title = i == 1 ? "QuasiStrided / StridedBLAS time" : "", titlesize = 16,
            ylabel = "time ratio (< 1: QuasiStrided faster)"
        )
        hlines!(ax, [1.0]; color = :black, linewidth = 1, linestyle = :dash)
        points!(ax, ps, p -> p.blas.intensity, p -> p.qs.t / p.blas.t, crange)
        ylims!(ax, ylim)
        if !isempty(ps)
            textlabel!(
                ax, Point2f(0.98, 0.03); space = :relative, text_align = (:right, :bottom), fontsize = 13,
                text = @sprintf("geomean %.2f\nQuasiStrided faster in %d of %d", geomean(rv), count(<(1), rv), length(rv)),
                background_color = (:white, 0.85), strokecolor = (:black, 0.3), cornerradius = 3, justification = :left
            )
        end
        last || foreach(a -> hidexdecorations!(a; grid = false, minorgrid = false), (rate_axes..., ax))
        Label(fig[i, 0], dtype; rotation = pi / 2, fontsize = 16, font = :bold, tellheight = false)
        append!(axes, rate_axes)
        push!(axes, ax)
    end
    linkxaxes!(axes...)
    Colorbar(fig[1:nrow, 4]; colormap = CMAP, limits = crange, label = "total work (log10 flop)")
    Legend(
        fig[nrow + 1, 1:3],
        [MarkerElement(; MARKERS[Symbol(c)]..., color = GREY, strokewidth = 0.4, strokecolor = :black) for c in CATEGORIES],
        CATEGORIES; orientation = :horizontal, framevisible = false
    )
    return fig
end

path = outpath("overview")
save(path, overview_figure())
println("wrote ", path)
