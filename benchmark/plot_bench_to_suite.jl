# Plots benchmark/bench_to_suite.jl's results: per (dtype, category, group)
# -- group = upstream's `source` (contract) or `topic` (network) tag, with
# groups of more than MAX_PER_FIG cases split by sweep dim -- a
# GFLOP/s comparison (StridedBLAS vs QuasiStrided only), a log-scaled
# QuasiStrided / StridedBLAS time-ratio chart (< 1 = QuasiStrided faster),
# and a separate per-case throughput violin plot ("..._violin.png").
#
#   julia --project=benchmark benchmark/plot_bench_to_suite.jl [csv_path]
#
# With no `csv_path`, uses the most recently modified
# benchmark/results/*/bench_to_suite.csv. Writes PNGs next to that CSV.
#
# Only Float64/ComplexF64 rows are plotted (Float32/ComplexF32, if present in
# the CSV, are skipped) and StridedNative is dropped from both panels -- this
# is a two-backend comparison (StridedBLAS vs QuasiStrided) by design.
#
# The violin plot is NOT built from raw per-rep samples -- bench_to_suite.jl
# only logs min/median/std of each case's REPS-sample throughput, to keep the
# CSV small. `synth_samples` below turns those three numbers into a plausible
# distribution (a normal(median, std) reflected below the observed min, so
# nothing falls under the recorded floor) purely for visual shape; it is a
# MODELED approximation, not the empirical distribution -- treat the violin's
# location/spread as real, its exact tail shape as illustrative only. Reading
# an old-format CSV without min_gflops/std_gflops columns skips the violin
# plot for that file (see `read_rows`).

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

# `params` is the free-form "k1=v1;k2=v2;..." field bench_to_suite.jl writes
# from `params_string`; values are kept as strings since categories other
# are not all integers (e.g. "layout=gemm_ready").
function parse_params(s::AbstractString)
    d = Dict{String, String}()
    isempty(s) && return d
    for kv in split(s, ';')
        k, v = split(kv, '=')
        d[k] = v
    end
    return d
end

# Only the column layout bench_to_suite.jl has written since the move to
# upstream's :contract/:network categories (17 columns, with group/layout/
# blas/intensity/expr) is supported; older CSVs predate that.
function read_rows(path)
    lines = readlines(path)
    occursin(",expr", lines[1]) ||
        error("$path predates the :contract/:network CSV layout; re-run bench_to_suite.jl")
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

# QuasiStrided / StridedBLAS time ratio: < 1 = QuasiStrided faster. Log-scaled
# on the plot (not linear) since case-to-case values span orders of magnitude
# (some ~150x) -- a linear axis makes everything but the most extreme case
# collapse to an indistinguishable sliver near zero.
function ratio_for(rows, case_id, num, den)
    t_num = only(r.t for r in rows if r.case_id == case_id && r.backend == num)
    t_den = only(r.t for r in rows if r.case_id == case_id && r.backend == den)
    return t_num / t_den
end

# Per-case axis label: the compact einsum expression bench_to_suite.jl wrote
# (single letters, "A B>C"), the sweep dim and the arithmetic intensity at the
# run's dtype. A leading "*" marks upstream's `isblasequivalent` cases (a
# single BLAS gemm after reshapes alone, no permutation anywhere).
case_label(r::Row) = @sprintf("%s%s  %s  dim=%d  %.3g FLOP/B", r.blas ? "*" : "", r.expr, r.case_id, r.dim, r.intensity)

# See the file header for why this is a modeled approximation, not real
# per-rep data: normal(median, std), reflected below `min_v` (mirrored back
# up rather than clipped, so the reflected mass still contributes density
# instead of piling up at the boundary). `seed` is deterministic in the
# inputs so replotting the same CSV always draws the same shape.
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
