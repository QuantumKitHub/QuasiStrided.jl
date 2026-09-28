# Plots bench_to_suite.jl's CSV, per (dtype, category, group) -- group =
# upstream's source/topic tag, split by sweep dim above MAX_PER_FIG cases: a
# GFLOP/s panel, a log-scaled QuasiStrided/StridedBLAS time ratio (< 1 =
# QuasiStrided faster), and a per-case throughput violin plot. Float64 and
# ComplexF64 only.
#
#   julia --project=benchmark benchmark/plot_bench_to_suite.jl [csv_path]
#
# Without `csv_path`, uses the newest benchmark/results/*/bench_to_suite.csv.
# Writes PNGs next to the CSV.
#
# The CSV keeps only median/min/std per case, so the violins are drawn from a
# normal(median, std) reflected at the observed min: their location and spread
# are real, their tail shape is illustrative.

using CairoMakie
using Printf
using Random

const PLOTTED_DTYPES = ("Float64", "ComplexF64")
const MAX_PER_FIG = 60

function latest_csv()
    root = joinpath(@__DIR__, "results")
    candidates = String[]
    for d in readdir(root; join = true)
        p = joinpath(d, "bench_to_suite.csv")
        isfile(p) && push!(candidates, p)
    end
    isempty(candidates) && error("no benchmark/results/*/bench_to_suite.csv found")
    return candidates[argmax(mtime.(candidates))]
end

const CSV_PATH = isempty(ARGS) ? latest_csv() : ARGS[1]
const OUTDIR = dirname(CSV_PATH)

struct Row
    backend::String
    dtype::String
    category::String
    case_id::String
    dim::Int
    params::Dict{String, String}
    reps::Int
    t::Float64
    gflops::Float64
    gbytes::Float64
    min_gflops::Float64
    std_gflops::Float64
    group::String
    blas::Bool
    intensity::Float64
    expr::String
end

# "k1=v1;k2=v2;..." as written by bench_to_suite.jl; values stay strings.
function parse_params(s::AbstractString)
    d = Dict{String, String}()
    isempty(s) && return d
    for kv in split(s, ';')
        k, v = split(kv, '=')
        d[k] = v
    end
    return d
end

function read_rows(path)
    lines = readlines(path)
    occursin(",expr", lines[1]) || error("$path is not a bench_to_suite.jl CSV (no expr column)")
    rows = Row[]
    for line in lines[2:end]
        f = split(line, ',')
        push!(
            rows, Row(
                f[1], f[2], f[3], f[4], parse(Int, f[5]), parse_params(f[6]), parse(Int, f[7]),
                parse(Float64, f[8]), parse(Float64, f[9]), parse(Float64, f[10]),
                parse(Float64, f[11]), parse(Float64, f[12]),
                f[13], f[15] == "true", parse(Float64, f[16]), f[17]
            )
        )
    end
    return rows
end

const ROWS = read_rows(CSV_PATH)

# Plotted on a log axis: ratios span orders of magnitude across cases.
function ratio_for(rows, case_id, num, den)
    t_num = only(r.t for r in rows if r.case_id == case_id && r.backend == num)
    t_den = only(r.t for r in rows if r.case_id == case_id && r.backend == den)
    return t_num / t_den
end

# A leading "*" marks upstream's `isblasequivalent` cases.
case_label(r::Row) = @sprintf("%s%s  %s  dim=%d  %.3g FLOP/B", r.blas ? "*" : "", r.expr, r.case_id, r.dim, r.intensity)

# Modeled samples (see the header), reflected rather than clipped at `min_v` so
# no mass piles up at the boundary; seeded from the inputs, so replots match.
function synth_samples(median_v::Float64, min_v::Float64, std_v::Float64; n::Int = 300)
    std_v <= 0 && return fill(median_v, n)
    rng = Random.Xoshiro(hash((median_v, min_v, std_v)))
    raw = median_v .+ std_v .* randn(rng, n)
    return map(x -> x < min_v ? 2 * min_v - x : x, raw)
end

# (dtype, category, group) panels; a group larger than MAX_PER_FIG is split
# into consecutive chunks of its sweep dims.
function panels(rows)
    out = Tuple{String, Vector{Row}}[]
    for dtype in PLOTTED_DTYPES, category in unique(r.category for r in rows),
            group in unique(r.group for r in rows if r.category == category)
        sub = filter(r -> r.dtype == dtype && r.category == category && r.group == group, rows)
        isempty(sub) && continue
        ncase = length(unique(r.case_id for r in sub))
        if ncase <= MAX_PER_FIG
            push!(out, ("$(dtype)_$(category)_$(group)", sub))
            continue
        end
        chunk, dims = Int[], sort(unique(r.dim for r in sub))
        flush_chunk() = (
            push!(
                out, (
                    "$(dtype)_$(category)_$(group)_dim$(first(chunk))-$(last(chunk))",
                    filter(r -> r.dim in chunk, sub),
                )
            ); empty!(chunk)
        )
        for d in dims
            nd = length(unique(r.case_id for r in sub if r.dim == d))
            n = length(unique(r.case_id for r in sub if r.dim in chunk))
            !isempty(chunk) && n + nd > MAX_PER_FIG && flush_chunk()
            push!(chunk, d)
        end
        isempty(chunk) || flush_chunk()
    end
    return out
end

for (name, subset) in panels(ROWS)
    ids = unique(r.case_id for r in subset)
    length(ids) < 2 && continue
    dtype = first(subset).dtype

    ratios = [ratio_for(subset, id, "QuasiStrided", "StridedBLAS") for id in ids]
    order = sortperm(ratios)
    ids, ratios = ids[order], ratios[order]
    labels = [case_label(first(r for r in subset if r.case_id == id)) for id in ids]
    title = replace(name, "_" => " / ")

    fig = Figure(size = (1250, max(400, 26 * length(ids) + 174)))
    Label(fig[0, 1:2], "* = BLAS-equivalent (upstream isblasequivalent: one gemm after reshapes alone)"; fontsize = 12)

    ax1 = Axis(
        fig[1, 1]; xscale = log10, yticks = (1:length(ids), labels),
        xlabel = "GFLOP/s (log scale)", title = "$title -- throughput"
    )
    for backend in ("StridedBLAS", "QuasiStrided")
        ys = [only(r.gflops for r in subset if r.case_id == id && r.backend == backend) for id in ids]
        scatter!(ax1, ys, 1:length(ids); label = backend, markersize = 10)
    end
    axislegend(ax1; position = :rb)

    ax2 = Axis(
        fig[1, 2]; xscale = log10,
        xlabel = "QuasiStrided / StridedBLAS time (log scale)",
        title = "ratio (< 1 = QuasiStrided faster)"
    )
    hideydecorations!(ax2)
    colors = [r <= 1 ? :seagreen : :firebrick for r in ratios]
    barplot!(ax2, 1:length(ids), ratios; direction = :x, color = colors)
    vlines!(ax2, [1.0]; color = :black, linestyle = :dash)

    path = joinpath(OUTDIR, "bench_to_suite_$(name).png")
    save(path, fig)
    println("wrote ", path)

    vfig = Figure(size = (1250, max(400, 26 * length(ids) + 174)))
    Label(vfig[0, 1], "* = BLAS-equivalent (upstream isblasequivalent: one gemm after reshapes alone)"; fontsize = 12)
    vax = Axis(
        vfig[1, 1]; yticks = (1:length(ids), labels),
        xlabel = "GFLOP/s -- modeled from median/min/std (see file header)",
        title = "$title -- throughput spread"
    )
    offset = 0.18
    for (boff, backend, color) in ((-offset, "StridedBLAS", :dodgerblue), (offset, "QuasiStrided", :orange))
        ys = Float64[]
        positions = Float64[]
        for (i, id) in enumerate(ids)
            row = only(r for r in subset if r.case_id == id && r.backend == backend)
            samples = synth_samples(row.gflops, row.min_gflops, row.std_gflops)
            append!(ys, samples)
            append!(positions, fill(Float64(i) + boff, length(samples)))
        end
        violin!(
            vax, positions, ys; orientation = :horizontal, side = boff < 0 ? :left : :right,
            width = 2 * abs(offset) * 1.8, color = color, label = backend
        )
    end
    axislegend(vax; position = :rb)
    vpath = joinpath(OUTDIR, "bench_to_suite_$(name)_violin.png")
    save(vpath, vfig)
    println("wrote ", vpath)
end
