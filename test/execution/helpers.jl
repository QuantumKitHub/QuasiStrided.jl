# Fixtures, references and path predicates shared by the execution test files.

include("../helpers.jl")

# Brute-force reference over every label assignment; knows nothing of plans.
function _brute_ref(A, indA, B, indB, Cstart, indC, alpha, beta; conjA = false, conjB = false)
    dims = Dict{Int, Int}()
    for (l, L) in zip(indA, size(A))
        dims[l] = L
    end
    for (l, L) in zip(indB, size(B))
        dims[l] = L
    end
    klabels = Tuple(l for l in indA if l in indB && !(l in indC))
    ksizes = Tuple(dims[l] for l in klabels)
    T = eltype(Cstart)
    out = similar(Cstart, T)
    for Ic in CartesianIndices(Cstart)
        at = Dict{Int, Int}(zip(indC, Tuple(Ic)))
        acc = zero(T)
        for Ik in CartesianIndices(ksizes)
            for (l, i) in zip(klabels, Tuple(Ik))
                at[l] = i
            end
            a = A[(at[l] for l in indA)...]
            b = B[(at[l] for l in indB)...]
            acc += (conjA ? conj(a) : a) * (conjB ? conj(b) : b)
        end
        out[Ic] = iszero(beta) ? alpha * acc : alpha * acc + beta * Cstart[Ic]
    end
    return out
end

# The reference result for the fixture `mk()`.
function _ref_of(mk, alpha, beta; kw...)
    Cv, Av, iA, Bv, iB, iC = mk()
    return _brute_ref(Array(Av), iA, Array(Bv), iB, Array(Cv), iC, alpha, beta; kw...)
end

# Plan a fresh fixture `mk()` and run `run!` on it. Returns C and the plan.
function _run_fresh(run!, mk, alpha, beta; plankw...)
    Cv, Av, iA, Bv, iB, iC = mk()
    plan = plan_contract(Cv, Av, iA, Bv, iB, iC; plankw...)
    @test run!(plan, alpha, beta) === plan.Cstorage
    return Array(Cv), plan
end

# `_run_fresh` planned onto the nest with B packed, through the path modes the
# benchmarks use: the baseline for the dedicated paths.
const NEST_ONLY = QuasiStrided.PathModes(dot = :never, outer = :never, unpacked_b = :never)
_run_nest(mk, alpha, beta; plankw...) = _run_fresh(execute!, mk, alpha, beta; path_modes = NEST_ONLY, plankw...)

_lanes(T) = QuasiStrided.vector_lanes(QuasiStrided.target_profile(), real(T))

# A DenseMatrix that is not a DenseVector: a plan keeps it as storage, so paths
# that need raw-pointer loads must decline it.
struct _WrappedMat{T} <: DenseMatrix{T}
    data::Matrix{T}
end
Base.size(a::_WrappedMat) = size(a.data)
Base.IndexStyle(::Type{<:_WrappedMat}) = IndexLinear()
Base.getindex(a::_WrappedMat, i::Int) = a.data[i]
Base.setindex!(a::_WrappedMat, v, i::Int) = (a.data[i] = v)

# `M` as a StridedView with the same values and a chosen layout of its first
# axis. Gaps in padded parents are NaN, so reading one poisons C.
function _view_as(M::Matrix{T}, layout::Symbol) where {T}
    m, n = size(M)
    layout === :dense && return StridedView(M)
    layout === :transposed && return permutedims(StridedView(permutedims(M, (2, 1))), (2, 1))
    layout === :wrapped && return StridedView(_WrappedMat(copy(M)), (m, n), (1, m), 0)
    big = fill(convert(T, NaN), 2m + 4, n)
    rows = layout === :offset ? (3:(m + 2)) : layout === :gap ? (1:2:(2m)) :
        layout === :reversed ? ((m + 2):-1:3) : error("unknown layout $layout")
    big[rows, :] .= M
    return StridedView(view(big, rows, :))
end

# `C[m,n] = A[m,k] B[k,n]` with chosen layouts for A and B, seeded so every call
# reproduces the same values. `ABfill` poisons both operands.
function _mm_maker(::Type{T}, M, K, N, seed; A = :dense, B = :dense, Cfill = nothing, ABfill = nothing) where {T}
    return function ()
        rng = MersenneTwister(seed)
        Amat = ABfill === nothing ? randn(rng, T, M, K) : fill(convert(T, ABfill), M, K)
        Bmat = ABfill === nothing ? randn(rng, T, K, N) : fill(convert(T, ABfill), K, N)
        Cmat = Cfill === nothing ? randn(rng, T, M, N) : fill(convert(T, Cfill), M, N)
        return (StridedView(Cmat), _view_as(Amat, A), (1, 2), _view_as(Bmat, B), (2, 3), (1, 3))
    end
end

_dot_takes(plan) = _path_of(plan) isa QuasiStrided.DotPath
_ub_takes(plan) = _path_of(plan) isa QuasiStrided.NestPath{true}
_outer_takes(plan) = _path_of(plan) isa QuasiStrided.OuterPath

# `C[a, b] = A[a] * B[b]` at extents `(M, N)`:
#   :bstrided  B every other element of a longer vector
#   :cpermC    C stored [b, a] (M not unit-stride in C)
#   :agap      A every other element (M not unit-stride in A)
#   :coffset   C a view with an offset, M unit-stride
#   :multiM    M a composite of two labels that fold to one ramp
function _outer_maker(::Type{T}, M, N, seed; variant = :plain, Cfill = nothing) where {T}
    return function ()
        rng = MersenneTwister(seed)
        a, b = randn(rng, T, M), randn(rng, T, N)
        C = Cfill === nothing ? randn(rng, T, M, N) : fill(convert(T, Cfill), M, N)
        gapped(v, r) = (big = fill(convert(T, NaN), 2 * length(v)); big[r] .= v; StridedView(view(big, r)))
        Av = variant === :agap ? gapped(a, 1:2:(2M)) : StridedView(a)
        Bv = variant === :bstrided ? gapped(b, 2:2:(2N)) : StridedView(b)
        variant === :multiM &&
            return (StridedView(reshape(C, 4, M ÷ 4, N)), StridedView(reshape(a, 4, M ÷ 4)), (1, 2), Bv, (3,), (1, 2, 3))
        Cv = if variant === :cpermC
            permutedims(StridedView(permutedims(C, (2, 1))), (2, 1))
        elseif variant === :coffset
            big = fill(convert(T, NaN), M + 3, N + 2)
            big[2:(M + 1), 2:(N + 1)] .= C
            StridedView(view(big, 2:(M + 1), 2:(N + 1)))
        else
            StridedView(C)
        end
        return (Cv, Av, (1,), Bv, (2,), (1, 2))
    end
end

const CR, RC = QuasiStrided.ComplexRealKernel, QuasiStrided.RealComplexKernel
