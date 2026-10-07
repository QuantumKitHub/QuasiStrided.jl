# Time to first call, in one fresh process:
#
#   julia --project=benchmark benchmark/probes/probe_ttfx.jl [--label X]
#
# Prints one line per step, in order: `using QuasiStrided`; the first Float64
# `plan_contract` + `execute!` (64^3); the first ComplexF64 one; the first
# `@tensor` with `QuasiStridedBackend`; the first `contract!`. Each step's time
# is what it adds on top of the steps before it, so run the script once per
# process, never twice in one session; repeat processes for a spread (the A/B
# script does, once per round).

label = let i = findfirst(==("--label"), ARGS)
    i === nothing ? "" : ARGS[i + 1]
end
report(step, t) = println(label, isempty(label) ? "" : "\t", rpad(step, 26), round(t; digits = 2), " s")

report("using QuasiStrided", @elapsed using QuasiStrided)
using StridedViews, TensorOperations
using QuasiStrided: plan_contract, execute!, contract!

gemm(T) = (StridedView(zeros(T, 64, 64)), StridedView(randn(T, 64, 64)), (1, 2), StridedView(randn(T, 64, 64)), (2, 3), (1, 3))
plan_execute(args) = execute!(plan_contract(args...), 1, 0)
function tensor_call(A, B, C)
    @tensor backend = QuasiStridedBackend() C[a, b, n] = A[a, k, b] * B[k, n]
    return C
end
contract_call((C, A, iA, B, iB, iC)) = contract!(C, 1, A, iA, B, iB, 0, iC)

f64, c64 = gemm(Float64), gemm(ComplexF64)
report("Float64 plan+execute!", @elapsed plan_execute(f64))
report("ComplexF64 plan+execute!", @elapsed plan_execute(c64))
report("@tensor", @elapsed tensor_call(randn(20, 30, 10), randn(30, 15), zeros(20, 10, 15)))
report("contract!", @elapsed contract_call(gemm(Float64)))
