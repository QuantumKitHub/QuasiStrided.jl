# Overview plots of bench_to_suite.jl's CSV, one panel per dtype:
#
#   throughput  GFLOP/s against arithmetic intensity, colour = total work,
#               filled = QuasiStrided, open = StridedBLAS, a faint segment
#               joining the two backends of a case.
#   relative    QuasiStrided/StridedBLAS time against total work (< 1 =
#               QuasiStrided faster), colour = arithmetic intensity, marker =
#               category; the geomean, the faster count and the most extreme
#               cases at both ends are annotated.
#
#   julia --project=benchmark benchmark/plot_bench_to_suite.jl [csv_path]
#       [--dtypes ComplexF64] [--categories network] [--tags source=tccg,topic=mps]
#
# Without `csv_path`, uses the newest benchmark/results/*/bench_to_suite.csv.
# `--tags` selects on the case params: a case is kept when it carries at least
# one of the named keys and matches one of the values given for each key it
# carries, so `source=tccg,topic=mps` keeps the TCCG and the MPS cases. Writes
# bench_to_suite_{throughput,relative}[_<filters>].png next to the CSV.

using CairoMakie
using Printf

include(joinpath(@__DIR__, "harness.jl"))

const NLABELLED = 3

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

keep(r) = (DTYPES_FILTER === nothing || r.dtype in DTYPES_FILTER) &&
    (CATEGORIES_FILTER === nothing || r.category in CATEGORIES_FILTER) && tags_match(r.params)

const ROWS = filter(keep, read_rows(CSV_PATH))
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

# Viridis/cividis without their palest end, which vanishes against white.
truncated(name) = cgrad(Makie.to_colormap(name)[1:220])
const GREY = RGBf(0.45, 0.45, 0.45)
const LOG_TICKS = [m * 10.0^e for e in -3:4 for m in (1, 2, 5)]
# Ratios within a decade of each other get the steps in between.
ratio_ticks(lo, hi) = hi / lo < 10 ? sort(vcat(LOG_TICKS, [0.6, 0.7, 0.8, 0.9, 1.1, 1.25, 1.5, 3])) : LOG_TICKS
plainticks(vs) = [@sprintf("%g", v) for v in vs]
const TITLE = "$(basename(dirname(CSV_PATH)))$(isempty(SUFFIX) ? "" : "  [" * replace(SUFFIX, "_" => ", ") * "]")"

function throughput_figure()
    fig = Figure(size = (620 * length(PANEL_DTYPES) + 120, 600))
    Label(fig[0, 1:length(PANEL_DTYPES)], TITLE; fontsize = 13, color = GREY)
    logwork = log10.(r.flops for r in ROWS)
    crange = extrema(logwork)
    cmap = truncated(:viridis)
    for (j, dtype) in enumerate(PANEL_DTYPES)
        ax = Axis(
            fig[1, j]; xscale = log10, yscale = log10, yticks = LOG_TICKS, ytickformat = plainticks, title = dtype,
            xlabel = "arithmetic intensity (flop/byte)", ylabel = j == 1 ? "throughput (GFLOP/s)" : ""
        )
        ps = filter(p -> p.dtype == dtype, PAIRS)
        segs = [Point2f(p.blas.intensity, p.blas.gflops) => Point2f(p.qs.intensity, p.qs.gflops) for p in ps]
        linesegments!(ax, segs; color = (:black, 0.12), linewidth = 1)
        for (backend, filled) in (("StridedBLAS", false), ("QuasiStrided", true))
            rs = filter(r -> r.dtype == dtype && r.backend == backend, ROWS)
            c = [log10(r.flops) for r in rs]
            pts = [Point2f(r.intensity, r.gflops) for r in rs]
            if filled
                scatter!(ax, pts; color = c, colormap = cmap, colorrange = crange, markersize = 7, strokewidth = 0.3, strokecolor = :white)
            else
                stroke = [get(cmap, (x - crange[1]) / (crange[2] - crange[1])) for x in c]
                scatter!(ax, pts; color = :transparent, strokecolor = stroke, markersize = 8, strokewidth = 1.1)
            end
        end
    end
    Colorbar(fig[1, length(PANEL_DTYPES) + 1]; colormap = cmap, limits = crange, label = "total work (log10 flop)")
    Legend(
        fig[2, 1:length(PANEL_DTYPES)],
        [
            MarkerElement(marker = :circle, color = GREY, markersize = 9),
            MarkerElement(marker = :circle, color = :transparent, strokecolor = GREY, strokewidth = 1.2, markersize = 9),
            LineElement(color = (:black, 0.3)),
        ],
        ["QuasiStrided", "StridedBLAS", "same case"]; orientation = :horizontal, framevisible = false
    )
    return fig
end

const CATEGORY_MARKERS = Dict("contract" => :circle, "network" => :utriangle)

function relative_figure()
    fig = Figure(size = (620 * length(PANEL_DTYPES) + 120, 700))
    Label(fig[0, 1:length(PANEL_DTYPES)], TITLE; fontsize = 13, color = GREY)
    loginten = log10.(p.blas.intensity for p in PAIRS)
    crange = extrema(loginten)
    cmap = truncated(:cividis)
    for (j, dtype) in enumerate(PANEL_DTYPES)
        ps = filter(p -> p.dtype == dtype, PAIRS)
        ratio(p) = p.qs.t / p.blas.t
        rv = ratio.(ps)
        ylim = isempty(ps) ? (0.5, 2.0) : extrema(rv) .* (0.7, 1.4)
        ax = Axis(
            fig[1, j]; xscale = log10, yscale = log10, yticks = ratio_ticks(ylim...), ytickformat = plainticks,
            title = isempty(ps) ? dtype : @sprintf("%s: geomean %.2f, QuasiStrided faster in %d of %d", dtype, geomean(rv), count(<(1), rv), length(rv)),
            xlabel = "total work (flop)", ylabel = j == 1 ? "time QuasiStrided / StridedBLAS\n(< 1: QuasiStrided faster)" : ""
        )
        hlines!(ax, [1.0]; color = :black, linewidth = 1, linestyle = :dash)
        isempty(ps) && continue
        for (cat, marker) in CATEGORY_MARKERS
            cs = filter(p -> p.category == cat, ps)
            isempty(cs) && continue
            scatter!(
                ax, [Point2f(p.blas.flops, ratio(p)) for p in cs]; marker, markersize = cat == "network" ? 11 : 7,
                color = [log10(p.blas.intensity) for p in cs], colormap = cmap, colorrange = crange,
                strokewidth = cat == "network" ? 0.8 : 0.3, strokecolor = cat == "network" ? :black : :white
            )
        end
        # The extreme cases are numbered at the point and named in a key below
        # the panel: their ids are too long to place beside clustered points.
        order = sortperm(rv)
        k = min(NLABELLED, length(ps) ÷ 2)
        ends = (fastest = order[1:k], slowest = reverse(order)[1:k])
        key = fig[2, j] = GridLayout()
        for (col, (name, idx)) in enumerate(pairs(ends))
            for (n, i) in enumerate(idx)
                tag = col == 1 ? string(n) : string('a' + n - 1)
                text!(
                    ax, Point2f(ps[i].blas.flops, rv[i]); text = tag, fontsize = 12, font = :bold,
                    align = (:center, col == 1 ? :top : :bottom), offset = (0, col == 1 ? -4 : 4)
                )
            end
            lines = [@sprintf("%s  %s  %.2f", col == 1 ? string(n) : string('a' + n - 1), ps[i].id, rv[i]) for (n, i) in enumerate(idx)]
            Label(key[1, col], "$name:\n" * join(lines, "\n"); fontsize = 11, justification = :left, halign = :left, valign = :top, tellwidth = false)
        end
        ylims!(ax, ylim)
    end
    Colorbar(fig[1, length(PANEL_DTYPES) + 1]; colormap = cmap, limits = crange, label = "arithmetic intensity (log10 flop/byte)")
    cats = filter(c -> any(p -> p.category == c, PAIRS), ["contract", "network"])
    Legend(
        fig[3, 1:length(PANEL_DTYPES)],
        [MarkerElement(marker = CATEGORY_MARKERS[c], color = GREY, markersize = c == "network" ? 11 : 9) for c in cats],
        cats; orientation = :horizontal, framevisible = false
    )
    return fig
end

for (name, f) in (("throughput", throughput_figure), ("relative", relative_figure))
    path = outpath(name)
    save(path, f())
    println("wrote ", path)
end
