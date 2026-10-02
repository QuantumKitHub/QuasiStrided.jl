# Same-node A/B of complex cache blocking: the kernel's own analytical blocking
# (`default_blocking`) against the real row of `real(T)`'s default kernel
# divided by packed reals per element, interleaved per case, for every menu
# shape of the planar, 1m and fmaddsub kernels that fits the host's vectors.
#
#   julia --project=benchmark benchmark/bench_complex_blocking.jl
#       [--dtypes ComplexF64,ComplexF32] [--rounds 3] [--reps 3] [--outdir DIR]

include(joinpath(@__DIR__, "harness.jl"))

using QuasiStrided: PlanarKernel, OneMKernel, FMAddSubKernel, Blocking, default_blocking,
    kernel_shapes, kernel_from_shape, derived_shape, target_profile, pack_formats,
    reals_per_element, scalartype

const RUN_DTYPES = parse_dtypes(argopt("dtypes", "ComplexF64,ComplexF32"))
const ROUNDS = argopt("rounds", 3)
const REPS = argopt("reps", 3)
const OUTDIR = outdir()
const CSV_PATH = joinpath(OUTDIR, "bench_complex_blocking.csv")

const CASES = [
    ShapeSpec("512^3", 512, 512, 512),
    ShapeSpec("2048^3", 2048, 2048, 2048),
    ShapeSpec("smallM_12x2048x2048", 12, 2048, 2048),
    ShapeSpec("smallM_24x2048x2048", 24, 2048, 2048),
]

function scaled_real_row(kernel)
    T = scalartype(kernel)
    R = real(T)
    row = default_blocking(kernel_from_shape(derived_shape(target_profile(), R), R, SIMDKernel))
    a_reals, b_reals = map(reals_per_element, pack_formats(typeof(kernel)))
    return Blocking(max(1, row.m_block ÷ a_reals), row.k_block, max(1, row.n_block ÷ b_reals))
end

function time_blocking(fx, kernel, b::Blocking, ::Type{T}) where {T}
    plan = plan_contract(
        fx.Cv, fx.Av, fx.indA, fx.Bv, fx.indB, fx.indC;
        kernel, m_block = b.m_block, k_block = b.k_block, n_block = b.n_block, oracle = false
    )
    return median_time_s(() -> execute!(plan, one(T), zero(T)); reps = REPS)
end

function run(io)
    print_env_header(stdout, "bench_complex_blocking.jl")
    println(io, "dtype,kernel,MR,NR,W,case,old_m,old_k,old_n,new_m,new_k,new_n,round,t_old,t_new")
    rng = MersenneTwister(1)
    lanes(T) = target_profile().vector_bytes ÷ sizeof(real(T))
    for T in RUN_DTYPES, K in (PlanarKernel, OneMKernel, FMAddSubKernel)
        for (MR, NR, W) in kernel_shapes(T, K)
            W <= max(lanes(T), 2) || continue
            kernel = K(Val(MR), Val(NR), T, Val(W))
            new = default_blocking(kernel)
            old = scaled_real_row(kernel)
            if new == old
                println("skip $T $(nameof(K)) ($MR,$NR,$W): blocking unchanged $new")
                continue
            end
            for spec in CASES
                fx = build_plain(T, spec, rng)
                for r in 1:ROUNDS
                    t_old = time_blocking(fx, kernel, old, T)
                    t_new = time_blocking(fx, kernel, new, T)
                    println(
                        io, join(
                            (
                                T, nameof(K), MR, NR, W, spec.name, old.m_block, old.k_block, old.n_block,
                                new.m_block, new.k_block, new.n_block, r, t_old, t_new,
                            ), ","
                        )
                    )
                    flush(io)
                    @printf(
                        "%s %s (%d,%d,%d) %s round %d: old %.4g s, new %.4g s, new/old %.3f\n",
                        T, nameof(K), MR, NR, W, spec.name, r, t_old, t_new, t_new / t_old
                    )
                end
            end
        end
    end
    return nothing
end

open(io -> run(io), CSV_PATH, "w")
println("wrote ", CSV_PATH)
