# A/B of the three automatically selected small/degenerate-shape paths against
# the five-loop nest, via their mode overrides:
#
#   outer      K = 1 outer product      C[a,b] = A[a] B[b]           _OUTER_MODE
#   dot        M = 1 gemv               C[cde] = A[ab] B[abcde]      _DOT_MODE
#   unpackedb  small square matmul      C[a,c] = A[a,b] B[b,c]       _UNPACKED_B_MODE
#
# `execute!` only (planning excluded), plus OpenBLAS `mul!` on the equivalent
# dense matrices for scale. Modes are interleaved rep by rep; each rep times a
# batch of calls sized to ~200 us. Prints median ns per call and [q25, q75].
#
#   julia --project=benchmark benchmark/bench_degenerate.jl [--reps 31]

include(joinpath(@__DIR__, "harness.jl"))
using Statistics: quantile

const REPS = argopt("reps", 31)

function batchsize(f!)
    f!()
    t0 = time_ns(); f!(); t = max(time_ns() - t0, 1)
    return clamp(round(Int, 200_000 / t), 1, 100_000)
end

# Interleaved timing of several closures; returns per-closure (median, q25, q75) ns.
function interleaved(fs::Vector)
    ns = [batchsize(f) for f in fs]
    ts = [Float64[] for _ in fs]
    for _ in 1:REPS, (i, f) in enumerate(fs)
        n = ns[i]
        t0 = time_ns()
        for _ in 1:n
            f()
        end
        push!(ts[i], (time_ns() - t0) / n)
    end
    return [(median(t), quantile(t, 0.25), quantile(t, 0.75)) for t in ts]
end

fmt((m, lo, hi)) = @sprintf("%9.0f [%6.0f,%6.0f]", m, lo, hi)

function withmode(f, ref::Ref{Symbol}, mode::Symbol)
    return () -> begin
        old = ref[]; ref[] = mode
        f()
        ref[] = old
    end
end

function row(name, ref, plan, blasf)
    ex = () -> execute!(plan, 1.0, 0.0)
    r = interleaved(Any[withmode(ex, ref, :never), withmode(ex, ref, :auto), blasf])
    @printf(
        "%-28s nest %s  path %s  blas %s  path/nest %.2f  path/blas %.2f\n",
        name, fmt(r[1]), fmt(r[2]), fmt(r[3]), r[2][1] / r[1][1], r[2][1] / r[3][1]
    )
    return flush(stdout)
end

println("# ns per call: median [q25, q75], $REPS interleaved reps")
for T in (Float64, ComplexF64)
    println("## $T")
    for d in (16, 63, 128)
        a = randn(T, d); b = randn(T, d); C = zeros(T, d, d)
        plan = plan_contract(StridedView(C), StridedView(a), (1,), StridedView(b), (2,), (1, 2))
        am = reshape(a, d, 1); bm = reshape(b, 1, d)
        row("outer $(d)x$(d)", QuasiStrided._OUTER_MODE, plan, () -> mul!(C, am, bm))
    end
    for d in (6, 8, 12, 16)
        A = randn(T, d, d); B = randn(T, d, d, d, d, d); C = zeros(T, d, d, d)
        plan = plan_contract(StridedView(C), StridedView(A), (1, 2), StridedView(B), (1, 2, 3, 4, 5), (3, 4, 5))
        Bm = reshape(B, d^2, d^3); av = vec(A); cv = vec(C)
        row("dot 1x$(d^2)x$(d^3)", QuasiStrided._DOT_MODE, plan, () -> mul!(cv, transpose(Bm), av))
    end
    for d in (16, 32, 64, 128)
        A = randn(T, d, d); B = randn(T, d, d); C = zeros(T, d, d)
        plan = plan_contract(StridedView(C), StridedView(A), (1, 2), StridedView(B), (2, 3), (1, 3))
        row("unpackedB $(d)^3", QuasiStrided._UNPACKED_B_MODE, plan, () -> mul!(C, A, B))
    end
end
