# Shared helpers for the benchmark scripts (`include`d, not a module):
# single-threaded warm-up-then-median timing, fixtures, a drift canary, CLI
# parsing and output paths.

using QuasiStrided
using QuasiStrided: SIMDKernel, tile_size, lanewidth, plan_contract, execute!
using StridedViews: StridedView
using LinearAlgebra
using Statistics: median
using Random
using Dates
using Printf

LinearAlgebra.BLAS.set_num_threads(1)
Threads.nthreads() == 1 ||
    @warn "Threads.nthreads() = $(Threads.nthreads()); these benchmarks are meant to run single-threaded"

# One discarded warm-up call, then the median of `reps` timed calls.
function median_time_s(f!::Function; reps::Int = 5)
    f!()
    ts = Vector{Float64}(undef, reps)
    for r in 1:reps
        t0 = time_ns()
        f!()
        ts[r] = (time_ns() - t0) / 1.0e9
    end
    return median(ts)
end

struct ShapeSpec
    name::String
    Ma::Int
    Ka::Int
    Na::Int
end
ShapeSpec(M::Int, K::Int, N::Int) = ShapeSpec("$(M)x$(K)x$(N)", M, K, N)

const MAIN_SHAPES = [
    ShapeSpec("64^3", 64, 64, 64),
    ShapeSpec("128^3", 128, 128, 128),
    ShapeSpec("256^3", 256, 256, 256),
    ShapeSpec("512^3", 512, 512, 512),
    ShapeSpec("shallowK_256x24x256", 256, 24, 256),
]
const EXTRA_SHAPES = [ShapeSpec("1024x256x1024", 1024, 256, 1024)]
# Small free extents: a larger MR/NR pads more of every micro-tile away, and
# small bond dimensions are the tensor-network common case.
const SMALL_SHAPES = [
    ShapeSpec("smallN_256x256x12", 256, 256, 12),
    ShapeSpec("smallM_12x256x256", 12, 256, 256),
    ShapeSpec("smallMN_16x256x16", 16, 256, 16),
]

function build_plain(::Type{T}, spec::ShapeSpec, rng) where {T}
    Amat = randn(rng, T, spec.Ma, spec.Ka)
    Bmat = randn(rng, T, spec.Ka, spec.Na)
    Cmat = zeros(T, spec.Ma, spec.Na)
    return (
        Av = StridedView(Amat), indA = (1, 2), Bv = StridedView(Bmat), indB = (2, 3),
        Cv = StridedView(Cmat), indC = (1, 3), Amat = Amat, Bmat = Bmat, Cmat = Cmat,
    )
end

full_grid(m_blocks, k_blocks, n_blocks) = [(m_block, k_block, n_block) for k_block in k_blocks for m_block in m_blocks for n_block in n_blocks]

const DTYPES = (Float64, Float32)
const CDTYPES = (ComplexF64, ComplexF32)

# Complex is charged the textbook 8 flops per multiply-accumulate, never an
# induced method's lower count, so methods compare on equal terms.
gflops(::Type{T}, Ma::Int, Ka::Int, Na::Int, seconds::Float64) where {T} =
    (T <: Complex ? 8 : 2) * Ma * Ka * Na / seconds / 1.0e9

geomean(v) = exp(sum(log, v) / length(v))

results_dir() = joinpath(@__DIR__, "results", "$(gethostname())-$(Dates.format(now(), "yyyy-mm-dd"))")

function outdir()
    d = something(argval("outdir"), results_dir())
    mkpath(d)
    return d
end

function git_commit()
    root = joinpath(@__DIR__, "..")
    return try
        sha = strip(read(`git -C $root rev-parse HEAD`, String))
        isempty(strip(read(`git -C $root status --porcelain`, String))) ? sha : sha * "-dirty"
    catch
        "unknown"
    end
end

function print_env_header(io::IO, script::String)
    println(io, "# QuasiStrided.jl benchmark/", script)
    println(io, "host = ", gethostname(), "  cpu = ", Sys.CPU_NAME, "  julia = ", VERSION)
    println(io, "nthreads = ", Threads.nthreads(), "  blas_threads = ", LinearAlgebra.BLAS.get_num_threads())
    println(io, "commit = ", git_commit())
    return println(io, "date = ", now())
end

# A fixed case timed at the start/middle/end of a sweep to catch drift. 15
# reps: fewer shows timer-resolution noise on this ~25 us shape.
function run_canary(rng, label::String)
    fx = build_plain(Float64, ShapeSpec("canary_64^3", 64, 64, 64), rng)
    plan = plan_contract(
        fx.Cv, fx.Av, fx.indA, fx.Bv, fx.indB, fx.indC;
        kernel = SIMDKernel(Val(8), Val(6), Float64), m_block = 128, k_block = 256, n_block = 1536
    )
    t = median_time_s(() -> execute!(plan, 1.0, 0.0); reps = 15)
    println("canary[$label] median = $(t) s")
    return t
end

relative_spread(ts) = isempty(ts) ? 0.0 : (maximum(ts) - minimum(ts)) / minimum(ts)

# `--name value` or `--name=value`.
function argval(name::String)
    for (i, a) in enumerate(ARGS)
        a == "--$name" && i < length(ARGS) && return ARGS[i + 1]
        startswith(a, "--$name=") && return split(a, '='; limit = 2)[2]
    end
    return nothing
end
hasflag(name::String) = "--$name" in ARGS
argopt(name::String, default::AbstractString) = something(argval(name), default)
argopt(name::String, default::Integer) = something(tryparse(Int, something(argval(name), "")), default)

const DTYPE_BY_NAME = Dict(
    "Float64" => Float64, "Float32" => Float32, "ComplexF64" => ComplexF64, "ComplexF32" => ComplexF32,
)
parse_dtypes(s::AbstractString) = Tuple(DTYPE_BY_NAME[strip(t)] for t in split(s, ','))
parse_ints(s::AbstractString) = Tuple(parse(Int, strip(t)) for t in split(s, ','))
