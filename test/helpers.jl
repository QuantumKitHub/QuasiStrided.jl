# Fixtures and bindings shared by all test files (included first by runtests.jl).

using StridedViews: StridedView, offset

# `const` bindings rather than `using QuasiStrided: ...`, so runtests.jl must
# not import these names too.
const plan_contract = QuasiStrided.plan_contract
const execute! = QuasiStrided.execute!
const ContractPlan = QuasiStrided.ContractPlan
const execute_tilewise! = QuasiStrided.execute_tilewise!

import TensorOperations as TO

# Offsets of coordinate `q` of an `AxisGroup` in every map, by direct decoding.
function offsets(g::AxisGroup{D, P}, q::Int) where {D, P}
    c = Tuple(CartesianIndices(g.lengths)[q + 1])
    return ntuple(p -> sum((c[d] - 1) * g.strides[p][d] for d in 1:D; init = 0), P)
end

# Plan for the matmul C[m,n] = sum_k A[m,k]*B[k,n].
function _mm_plan(Cmat, Amat, Bmat; kwargs...)
    return plan_contract(
        StridedView(Cmat), StridedView(Amat), (1, 2), StridedView(Bmat), (2, 3), (1, 3);
        kwargs...
    )
end

# Steady-state allocation of one `run!(plan, 1.0, 0.0)` call, after a warm-up.
function _steady_allocs!(run!, plan, Cmat)
    run!(plan, 1.0, 0.0)
    fill!(Cmat, 0.0)
    return @allocated run!(plan, 1.0, 0.0)
end

# C[a,n,b] = sum_k A[a,k,b]*B[k,n] with sizes a=3, k=5, b=2, n=4 and labels
# a=1, k=2, b=3, n=4.
function _worked_fixture()
    A = reshape(collect(1.0:30.0), 3, 5, 2)
    B = reshape(collect(1.0:20.0), 5, 4)
    Cref = zeros(Float64, 3, 4, 2)
    for b in 1:2, n in 1:4, a in 1:3
        Cref[a, n, b] = sum(A[a, k, b] * B[k, n] for k in 1:5)
    end
    return A, B, Cref
end

const _INDA = (1, 2, 3)
const _INDB = (2, 4)
const _INDC = (1, 4, 3)

using QuasiStrided: TargetProfile, CacheLevel

const VALID_ISAS = (:avx512, :avx2, :neon, :unknown)
synthetic(isakey) = TargetProfile(isakey, "synthetic", CacheLevel(), CacheLevel(), CacheLevel())

# Permuted A (with a zero-stride axis), negative-stride B, offset sliced C.
function scattered_fixture(::Type{T}, a_n = 32, k_n = 32, b_n = 8, n_n = 32) where {T}
    A2 = randn(MersenneTwister(11), T, a_n, k_n)
    Aperm = permutedims(StridedView(vec(A2), (a_n, k_n, b_n), (1, a_n, 0), 0), (2, 3, 1))
    Bneg = StridedView(randn(MersenneTwister(12), T, k_n * n_n), (k_n, n_n), (-1, k_n), k_n - 1)
    Cbig = zeros(T, a_n + 2, n_n + 3, b_n + 1)
    Cv = StridedView(view(Cbig, 2:(a_n + 1), 2:(n_n + 1), 1:b_n))
    return (Cv, Aperm, (2, 3, 1), Bneg, (2, 4), (1, 4, 3))
end

# Steady-state allocation of one `pack!` call. `::F where {F}` forces
# specialization on the pass-through `transform`; without it the dynamic call
# allocates on Julia 1.10.
function steady_pack_allocs(pack!::P, packed, source, kernel, transform::F) where {P, F}
    pack!(packed, source, kernel, transform)
    return @allocated pack!(packed, source, kernel, transform)
end
