# Throughput of every register tile in the kernel menu, against the engine's
# default choice and OpenBLAS `gemm!`, single core.
#
#   julia --project=benchmark benchmark/bench_kernels.jl [--dtypes Float64,ComplexF64]
#       [--shapes 64x64x64,2048x2048x2048] [--reps 11] [--outdir DIR]
#
# Arms:
#   kernel  (real dtypes) the ceiling of each menu tile: `unsafe_execute_tile!`
#           on one L1-resident packed A and B sliver at the tile's default `k_block`.
#   engine  per dtype and shape: OpenBLAS, the default plan, and every menu
#           kernel (of every complex method, for complex dtypes) at its own
#           default blocking; all configurations of a shape back to back.
#
# The summary gives each configuration's geomean fraction of OpenBLAS and,
# when both are run, the complex efficiency: the default complex GF/s over the
# default real GF/s at the same shape (complex at 8 flops/MAC). It should
# exceed 1 -- complex has twice the arithmetic intensity -- so below ~0.9
# points at a complex-specific overhead (packing or accumulator spills).
#
# Default shapes: harness.jl's MAIN/EXTRA/SMALL shapes plus 2048^3.

include(joinpath(@__DIR__, "harness.jl"))

using QuasiStrided: RealMethod, PlanarMethod, OneMMethod, FMAddSubMethod, kernel_shapes,
    _kernel_from_shape, default_blocking, packed_panel, unsafe_execute_tile!,
    DestinationTile, AffineAxis, target_profile

const RUN_DTYPES = parse_dtypes(argopt("dtypes", "Float64,Float32,ComplexF64,ComplexF32"))
const SHAPES = let s = argval("shapes")
    s === nothing ? vcat(MAIN_SHAPES, EXTRA_SHAPES, SMALL_SHAPES, [ShapeSpec(2048, 2048, 2048)]) :
        [ShapeSpec(parse_ints(replace(x, 'x' => ','))...) for x in split(s, ',')]
end
const REPS = argopt("reps", 11)
const OUTDIR = outdir()
const CSV_PATH = joinpath(OUTDIR, "bench_kernels.csv")
const SUMMARY_PATH = joinpath(OUTDIR, "summary_kernels.txt")

tag(k) = "$(tile_size(k)[1])x$(tile_size(k)[2])/W$(lanewidth(k))"
menu_methods(::Type{T}) where {T} = T <: Complex ? (PlanarMethod(), OneMMethod(), FMAddSubMethod()) : (RealMethod(),)

csv = open(CSV_PATH, "w")
println(csv, "arm,dtype,shape,M,K,N,config,kernel,m_block,k_block,n_block,seconds,gflops,frac_openblas")
rows = NamedTuple[]
function record!(arm, T, spec, config, kernel, blk, t, t_blas)
    gf = gflops(T, spec.Ma, spec.Ka, spec.Na, t)
    println(
        csv, "$arm,$T,$(spec.name),$(spec.Ma),$(spec.Ka),$(spec.Na),$config,$kernel,",
        blk === nothing ? ",," : "$(blk.m_block),$(blk.k_block),$(blk.n_block)", ",",
        @sprintf("%.9f,%.3f,%.4f", t, gf, t_blas / t)
    )
    flush(csv)
    return push!(rows, (; arm, dtype = T, shape = spec.name, config, kernel, t, gf, frac = t_blas / t))
end

print_env_header(stdout, "bench_kernels.jl")
println("target = ", target_profile())
canaries = [run_canary(MersenneTwister(0xCA), "start")]

function kernel_hot!(kernel, C, apack, bpack, k_block, reps)
    GC.@preserve apack bpack begin
        ap = packed_panel(apack, 1, length(apack))
        bp = packed_panel(bpack, 1, length(bpack))
        dest = DestinationTile(C, 0, AffineAxis(0, 1, tile_size(kernel)[1]), AffineAxis(0, tile_size(kernel)...))
        for _ in 1:reps
            unsafe_execute_tile!(kernel, dest, ap, bp, k_block, one(eltype(C)), one(eltype(C)))
        end
    end
    return nothing
end

for T in filter(t -> t <: Real, RUN_DTYPES), sh in kernel_shapes(T)
    kernel = _kernel_from_shape(sh, T)
    MR, NR = sh
    k_block = default_blocking(kernel).k_block
    apack = rand(T, MR * k_block); bpack = rand(T, NR * k_block); C = zeros(T, MR * NR)
    kernel_hot!(kernel, C, apack, bpack, k_block, 10)
    reps = max(100, round(Int, 0.05 / @elapsed(kernel_hot!(kernel, C, apack, bpack, k_block, 100)) * 100))
    t = median_time_s(() -> kernel_hot!(kernel, C, apack, bpack, k_block, reps); reps = REPS) / reps
    record!("kernel", T, ShapeSpec("l1_tile", MR, k_block, NR), "menu", tag(kernel), nothing, t, NaN)
end

function time_plan(fx, ::Type{T}; kw...) where {T}
    plan = plan_contract(fx.Cv, fx.Av, fx.indA, fx.Bv, fx.indB, fx.indC; kw...)
    return median_time_s(() -> execute!(plan, one(T), zero(T)); reps = REPS), plan
end

for spec in SHAPES, T in RUN_DTYPES
    fx = build_plain(T, spec, MersenneTwister(0x5EED))
    tb = median_time_s(() -> BLAS.gemm!('N', 'N', one(T), fx.Amat, fx.Bmat, zero(T), fx.Cmat); reps = REPS)
    record!("engine", T, spec, "openblas", "-", nothing, tb, tb)
    td, pd = time_plan(fx, T)
    @assert isapprox(fx.Cmat, fx.Amat * fx.Bmat; rtol = sqrt(eps(real(T))))
    record!("engine", T, spec, "default", tag(pd.kernel), pd.blocking, td, tb)
    for m in menu_methods(T), sh in kernel_shapes(T, m)
        k = _kernel_from_shape(sh, T, m)
        b = default_blocking(k)
        t, _ = time_plan(fx, T; kernel = k, m_block = b.m_block, k_block = b.k_block, n_block = b.n_block)
        record!("engine", T, spec, lowercase(replace(string(nameof(typeof(m))), "Method" => "")), tag(k), b, t, tb)
    end
    println("  ", rpad(string(T), 11), rpad(spec.name, 22), @sprintf("default %7.2f GF/s  %.2fx OpenBLAS", gflops(T, spec.Ma, spec.Ka, spec.Na, td), tb / td))
end
push!(canaries, run_canary(MersenneTwister(0xCA), "end"))
close(csv)

open(SUMMARY_PATH, "w") do io
    print_env_header(io, "bench_kernels.jl")
    println(io, "target = ", target_profile(), "\nreps = ", REPS)
    @printf(io, "canary spread = %.1f%%  %s\n", 100 * relative_spread(canaries), canaries)
    println(io, "\n== kernel arm: L1-resident tile GF/s ==")
    for r in rows
        r.arm == "kernel" && @printf(io, "  %-8s %-14s %8.2f\n", r.dtype, r.kernel, r.gf)
    end
    println(io, "\n== engine arm: geomean fraction of OpenBLAS throughput over shapes (1 = parity) ==")
    eng = filter(r -> r.arm == "engine" && r.config != "openblas", rows)
    for T in RUN_DTYPES, key in unique((r.config, r.kernel) for r in eng if r.dtype == T && r.config != "default")
        v = [r.frac for r in eng if r.dtype == T && (r.config, r.kernel) == key]
        @printf(io, "  %-11s %-16s %-14s %.3f\n", T, key..., geomean(v))
    end
    for T in RUN_DTYPES
        v = [r.frac for r in eng if r.dtype == T && r.config == "default"]
        isempty(v) || @printf(io, "  %-11s %-31s %.3f\n", T, "default", geomean(v))
    end
    for (Tr, Tc) in ((Float64, ComplexF64), (Float32, ComplexF32))
        (Tr in RUN_DTYPES && Tc in RUN_DTYPES) || continue
        println(io, "\n== complex efficiency $Tc / $Tr (default plans) ==")
        effs = Float64[]
        for spec in SHAPES
            gr = only(r.gf for r in eng if r.dtype == Tr && r.shape == spec.name && r.config == "default")
            gc = only(r.gf for r in eng if r.dtype == Tc && r.shape == spec.name && r.config == "default")
            push!(effs, gc / gr)
            @printf(io, "  %-22s %.3f\n", spec.name, gc / gr)
        end
        @printf(io, "  %-22s %.3f\n", "geomean", geomean(effs))
    end
end
print(read(SUMMARY_PATH, String))
println("wrote ", CSV_PATH)
