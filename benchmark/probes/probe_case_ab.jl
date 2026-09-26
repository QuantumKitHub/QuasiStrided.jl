# Interleaved A/B timing of ONE upstream-suite-style contraction case through
# `TensorOperations.tensorcontract!`, StridedBLAS vs QuasiStridedBackend,
# built exactly as benchmark/bench_to_suite.jl builds it (plain dense arrays in
# the given label orders, `contract_indices` for pA/pB/pAB, alpha=1, beta=0).
# ABAB interleaving over `--rounds` rounds, `--reps` calls per round per
# backend, so a slow patch on the shared machine hits both backends alike.
#
#   julia -t 1 --project=benchmark benchmark/probes/probe_case_ab.jl \
#       --expr "abcd bdef>acef" --dim 63 --dtype ComplexF64 --reps 2 --rounds 2

using TensorOperations
using TensorOperations: StridedBLAS
using QuasiStrided
using QuasiStrided: QuasiStridedBackend
using Random
using Statistics: median
using Printf
using LinearAlgebra

include(joinpath(@__DIR__, "..", "harness.jl"))

const EXPR = argopt("expr", "abcd bdef>acef")
const DIM = argopt("dim", 63)
const T = parse_dtypes(argopt("dtype", "ComplexF64"))[1]
const REPS = argopt("reps", 2)
const ROUNDS = argopt("rounds", 2)

function parse_expr(expr::AbstractString)
    lhs, rhs = split(expr, '>')
    a, b = split(strip(lhs), ' ')
    return collect(a), collect(b), collect(strip(rhs))
end
LA, LB, LC = parse_expr(EXPR)
IA = Symbol.(LA); IB = Symbol.(LB); IC = Symbol.(LC)

rng = MersenneTwister(42)
A = randn(rng, T, ntuple(_ -> DIM, length(LA))...)
B = randn(rng, T, ntuple(_ -> DIM, length(LB))...)
C = zeros(T, ntuple(_ -> DIM, length(LC))...)
pA, pB, pAB = TensorOperations.contract_indices(Tuple(IA), Tuple(IB), Tuple(IC))
labels = unique(vcat(LA, LB, LC))
nk = length(intersect(LA, LB))
flops = 2.0 * DIM^(length(labels))

run!(backend) = TensorOperations.tensorcontract!(C, A, pA, false, B, pB, false, pAB, one(T), zero(T), backend)

println("expr = $EXPR dim = $DIM T = $T  flops = $flops  bytes = $(sizeof(A) + sizeof(B) + sizeof(C))")
run!(StridedBLAS()); Cref = copy(C)
run!(QuasiStridedBackend()); println("max abs diff QS vs BLAS = ", maximum(abs, C .- Cref))

tb = Float64[]; tq = Float64[]
for round in 1:ROUNDS
    for _ in 1:REPS
        t0 = time_ns(); run!(StridedBLAS()); push!(tb, (time_ns() - t0) / 1e9)
    end
    for _ in 1:REPS
        t0 = time_ns(); run!(QuasiStridedBackend()); push!(tq, (time_ns() - t0) / 1e9)
    end
    @printf("round %d: BLAS %s  QS %s\n", round, join((@sprintf("%.3f", x) for x in tb[end-REPS+1:end]), ","), join((@sprintf("%.3f", x) for x in tq[end-REPS+1:end]), ","))
end
@printf("StridedBLAS   median %.4f s (min %.4f, max %.4f)  %.2f GF/s\n", median(tb), minimum(tb), maximum(tb), flops / median(tb) / 1e9)
@printf("QuasiStrided  median %.4f s (min %.4f, max %.4f)  %.2f GF/s\n", median(tq), minimum(tq), maximum(tq), flops / median(tq) / 1e9)
@printf("QS/BLAS = %.3f\n", median(tq) / median(tb))
