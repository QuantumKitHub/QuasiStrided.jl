# Number of compiled specialisations of the engine's hot generic functions
# after a fixed workload: one plan per path (real, complex and mixed nests with
# packed and unpacked B, line-by-line packing, a C panel, dot, outer) plus an
# `@tensor` call, each executed once. A growing count means a change splits
# methods on more type parameters, which costs compile time.
#
#   julia --project=benchmark benchmark/probes/probe_specialisations.jl [--label X]
#
# Prints one line per function: its specialisations summed over its methods.
# `planned` is counted through its keyword body. Names the checked-out
# revision lacks print 0, so two revisions compare line by line.

using QuasiStrided, StridedViews, TensorOperations
using QuasiStrided: plan_contract, execute!, PathModes

label = let i = findfirst(==("--label"), ARGS)
    i === nothing ? "" : ARGS[i + 1]
end

views(T, ext, (iA, iB, iC); TA = T, TB = T, TC = T) =
    (StridedView(zeros(TC, map(l -> ext[l], iC))), StridedView(randn(TA, map(l -> ext[l], iA))), iA, StridedView(randn(TB, map(l -> ext[l], iB))), iB, iC)
gemm(T, M, K, N; kw...) = views(T, (M, K, N), ((1, 2), (2, 3), (1, 3)); kw...)

# The line-by-line case follows the test suite's `SP_I7`; whether it splits
# depends on the host's L2, so it is a representative call rather than a forced split.
const CASES = (
    (gemm(Float64, 64, 64, 64), (;)),
    (gemm(ComplexF64, 64, 64, 64), (;)),
    (gemm(ComplexF64, 64, 64, 64; TB = Float64), (;)),
    (gemm(ComplexF64, 64, 64, 64; TA = Float64), (;)),
    (gemm(Float64, 16, 16, 16), (; path_modes = PathModes(unpacked_b = :always))),
    (gemm(Float64, 16, 16, 16), (; path_modes = PathModes(unpacked_b = :never))),
    (gemm(ComplexF64, 16, 16, 16), (; path_modes = PathModes(unpacked_b = :never))),
    (views(Float64, (24, 24, 24, 24, 24, 24), ((1, 2, 3, 4, 5), (4, 6), (5, 3, 2, 6, 1))), (;)),
    (gemm(Float64, 13, 20, 40; TC = Float32), (; k_block = 4)),
    (gemm(Float64, 1, 256, 64), (;)),
    (gemm(ComplexF64, 1, 256, 64), (;)),
    (gemm(Float64, 64, 1, 64), (;)),
    (gemm(ComplexF64, 64, 1, 64), (;)),
)

for ((Cv, Av, iA, Bv, iB, iC), kw) in CASES
    execute!(plan_contract(Cv, Av, iA, Bv, iB, iC; kw...), 1, 0)
end
let A = randn(20, 30, 10), B = randn(30, 15), C = zeros(20, 10, 15)
    @tensor backend = QuasiStridedBackend() C[a, b, n] = A[a, k, b] * B[k, n]
end

nspec(m::Method) = count(_ -> true, Base.specializations(m))
nspec(f) = sum(nspec, methods(f); init = 0)
for name in (:nest!, :micro_tiles!, :execute_tile!, :store_tile!, :pack!, :execute_path!, :build_plan, :planned)
    n = !isdefined(QuasiStrided, name) ? 0 :
        name === :planned ? sum(m -> nspec(Base.bodyfunction(m)), methods(QuasiStrided.planned); init = 0) :
        nspec(getfield(QuasiStrided, name))
    println(label, isempty(label) ? "" : "\t", rpad(name, 15), n)
end
