# Helpers shared by the execution test files included after this one.

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

# `execute!` forced onto the nest with B packed, through the mode switches
# the benchmarks use: the baseline for the dedicated paths.
const _PATH_MODES = (QuasiStrided._DOT_MODE, QuasiStrided._OUTER_MODE, QuasiStrided._UNPACKED_B_MODE)
function _run_nest!(plan, alpha, beta)
    old = map(getindex, _PATH_MODES)
    foreach(m -> m[] = :never, _PATH_MODES)
    try
        return execute!(plan, alpha, beta)
    finally
        foreach(setindex!, _PATH_MODES, old)
    end
end

_path_of(plan) = QuasiStrided._select_path(plan)

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

# `C[n,m]` stored N-major: for a real eltype the planner swaps the operand roles.
function _swapped_maker(::Type{T}, M, K, N, seed; B = :dense) where {T}
    return function ()
        rng = MersenneTwister(seed)
        Amat, Bmat, Cmat = randn(rng, T, M, K), randn(rng, T, K, N), randn(rng, T, N, M)
        return (StridedView(Cmat), StridedView(Amat), (1, 2), _view_as(Bmat, B), (2, 3), (3, 1))
    end
end

@testset "contract!: worked fixture, permuted and sliced views" begin
    A, B, Cref = _worked_fixture()
    C = zeros(3, 4, 2)
    contract!(StridedView(C), 1.0, StridedView(A), _INDA, StridedView(B), _INDB, 0.0, _INDC)
    @test C ≈ Cref

    # Ap[k,b,a] == A[a,k,b], Bp[n,k] == B[k,n].
    C = zeros(3, 4, 2)
    Ap, Bp = permutedims(A, (2, 3, 1)), permutedims(B, (2, 1))
    contract!(StridedView(C), 1.0, StridedView(Ap), (2, 3, 1), StridedView(Bp), (4, 2), 0.0, _INDC)
    @test C ≈ Cref

    # Nonzero view offsets on A and C.
    A4 = reshape(collect(1.0:40.0), 4, 5, 2)
    Cfull = zeros(4, 4, 2)
    Av, Cv = StridedView(view(A4, 2:4, :, :)), StridedView(view(Cfull, 2:4, :, :))
    @test offset(Av) != 0
    contract!(Cv, 1.0, Av, _INDA, StridedView(B), _INDB, 0.0, _INDC)
    @test Array(Cv) ≈ [sum(A4[a, k, b] * B[k, n] for k in 1:5) for a in 2:4, n in 1:4, b in 1:2]
    @test all(iszero, Cfull[1, :, :])
end

@testset "execute!: short-circuits and empty output" begin
    Ma, Ka, Na = 5, 6, 4
    for (Ka, alpha) in ((0, 1.0), (Ka, 0.0)), beta in (0.5, 0.0)
        # A and B are poisoned: neither may be read.
        mk = _mm_maker(Float64, Ma, Ka, Na, 1; ABfill = NaN)
        C, _ = _run_fresh(execute!, mk, alpha, beta)
        @test C == beta .* Array(mk()[1])
    end
    # beta = 0 never reads a NaN C, across several blocks.
    C, _ = _run_fresh(execute!, _mm_maker(Float64, 11, 10, 9, 2; Cfill = NaN), 2.5, 0.0; m_block = 4, k_block = 4, n_block = 4)
    @test C ≈ _ref_of(_mm_maker(Float64, 11, 10, 9, 2), 2.5, 0.0)
    # Empty M or N: a no-op.
    for (M, N) in ((3, 0), (0, 3))
        C, _ = _run_fresh(execute!, _mm_maker(Float64, M, 4, N, 3; Cfill = NaN), 1.0, 2.0)
        @test size(C) == (M, N)
    end
    # Singleton extents, and two K labels against a singleton free label.
    mk2K = () -> (
        StridedView(zeros(5, 1)), StridedView(randn(MersenneTwister(6), 5, 3, 4)), (1, 2, 3),
        StridedView(randn(MersenneTwister(7), 4, 3, 1)), (3, 2, 4), (1, 4),
    )
    for mk in (_mm_maker(Float64, 1, 1, 1, 5), mk2K)
        C, _ = _run_fresh(execute!, mk, 1.5, 0.5)
        @test C ≈ _ref_of(mk, 1.5, 0.5)
    end
    # Block sizes beyond the extents are clamped, not an error.
    C, _ = _run_fresh(execute!, _mm_maker(Float64, 9, 7, 6, 4), 1.0, 0.5; kernel = ScalarKernel(Val(4), Val(3), Float64), m_block = 10_000, k_block = 10_000, n_block = 10_000)
    @test C ≈ _ref_of(_mm_maker(Float64, 9, 7, 6, 4), 1.0, 0.5)
end

@testset "execute!: out-of-bounds operands are rejected before any write" begin
    # StridedView and plan_contract do not validate a view against its parent;
    # the hoisted span checks do. Each operand is one element short.
    M, K, N = 8, 6, 5
    for (sa, sb, sc) in ((M * K - 1, K * N, M * N), (M * K, K * N - 1, M * N), (M * K, K * N, M * N - 1))
        Cstore = zeros(sc)
        plan = plan_contract(
            StridedView(Cstore, (M, N), (1, M), 0), StridedView(randn(sa), (M, K), (1, M), 0), (1, 2),
            StridedView(randn(sb), (K, N), (1, K), 0), (2, 3), (1, 3); kernel = SIMDKernel(Val(8), Val(6), Float64)
        )
        @test_throws BoundsError execute!(plan, 1.0, 0.0)
        @test all(iszero, Cstore)
        @test_throws BoundsError execute_tilewise!(plan, 1.0, 0.0)
    end
end

@testset "execute!: allocation-free in steady state" begin
    Ma, Ka, Na = 19, 23, 17
    Amat, Bmat = randn(MersenneTwister(1), Ma, Ka), randn(MersenneTwister(2), Ka, Na)
    kernel = SIMDKernel(Val(8), Val(6), Float64, Val(4))
    # Default everything; a reused workspace without the oracle (the backend's
    # shape); several blocks with tail tiles in M and N (the vectorized store's
    # row tail); a conj transform, which the real path never builds but which
    # must still cross `_pack_sliver!` concretely typed.
    Cmat = zeros(Ma, Na)
    p = _mm_plan(Cmat, Amat, Bmat)
    plans = (
        p, _mm_plan(Cmat, Amat, Bmat; workspace = p.workspace, oracle = false),
        _mm_plan(Cmat, Amat, Bmat; kernel = kernel, m_block = 16, k_block = 5, n_block = 8),
        ContractPlan(
            p.kernel, p.mgroup, p.ngroup, p.kgroup, p.blocking,
            p.Astorage, p.Abase, p.Bstorage, p.Bbase, p.Cstorage, p.Cbase,
            conj, conj, p.workspace, p.mpack, p.npack,
        ),
    )
    @test parent(StridedView(Cmat)) isa DenseVector{Float64}
    for plan in plans
        allocs = _steady_allocs!(execute!, plan, Cmat)
        @test Cmat ≈ Amat * Bmat
        @test allocs == 0 skip = (VERSION < v"1.11")
    end
    allocs_tw = _steady_allocs!(execute_tilewise!, plans[4], Cmat)
    @test Cmat ≈ Amat * Bmat
    @test allocs_tw == 0 skip = (VERSION < v"1.11")
    ps = _mm_plan(Cmat, Amat, Bmat; kernel = ScalarKernel(Val(4), Val(3), Float64), m_block = 8, k_block = 6, n_block = 7)
    @test _steady_allocs!(execute!, ps, Cmat) == 0 skip = (VERSION < v"1.11")
end

_ws_union_members(t) = t isa Union ? (_ws_union_members(t.a)..., _ws_union_members(t.b)...) : (t,)

# Every non-concrete Tile/ContractWorkspace type in `f`'s unoptimized typed
# IR, directly or as a union member (`Union{}` is a throw, not an instability).
function _ws_nonconcrete_types(f, argtypes)
    bad = Any[]
    for (ci, rt) in Base.code_typed(f, argtypes; optimize = false)
        types = Any[rt]
        ci.slottypes isa Vector && append!(types, ci.slottypes)
        ci.ssavaluetypes isa Vector && append!(types, ci.ssavaluetypes)
        for t in types
            t isa Type || continue
            for m in _ws_union_members(t)
                (m isa Type && m !== Union{}) || continue
                if (m <: QuasiStrided.Tile || m <: QuasiStrided.ContractWorkspace) && !isconcretetype(m)
                    push!(bad, t)
                    break
                end
            end
        end
    end
    return unique(bad)
end

@testset "plan_contract/execute!: no union-typed or partially-applied tile/workspace types" begin
    Amat, Bmat, Cmat = randn(19, 23), randn(23, 17), zeros(19, 17)
    Av, Bv, Cv = StridedView(Amat), StridedView(Bmat), StridedView(Cmat)
    plan = _mm_plan(Cmat, Amat, Bmat; m_block = 8, k_block = 6, n_block = 7)
    @test isconcretetype(typeof(plan))
    @test all(isconcretetype, fieldtypes(typeof(plan.workspace)))
    @test typeof(plan.workspace) === QuasiStrided.ContractWorkspace{Float64, Vector{Float64}, Vector{Float64}}
    plan_argtypes = (typeof(Cv), typeof(Av), NTuple{2, Int}, typeof(Bv), NTuple{2, Int}, NTuple{2, Int})
    @test isempty(_ws_nonconcrete_types(plan_contract, plan_argtypes))
    for f in (execute!, execute_tilewise!)
        @test isempty(_ws_nonconcrete_types(f, (typeof(plan), Float64, Float64)))
    end
    @test isconcretetype(only(Base.return_types(execute!, (typeof(plan), Float64, Float64))))
end

# Line-by-line packing. intensli_7, C[e,c,b,f,a] = A[a,b,c,d,e] * B[d,f], splits A
# (K steps of a page or more, every eltype); C[a1,au,f1,f2] = A[au,k,a1] * B[f2,k,f1]
# splits both operands (K steps within a page, real only).
const _SP_I7 = ((1, 2, 3, 4, 5), (4, 6), (5, 3, 2, 6, 1))
const _SP_BOTH = ((2, 3, 1), (5, 3, 4), (1, 2, 4, 5))

function _sp_views(T, (iA, iB, iC), ext, TA = T)
    arr(S, I) = StridedView(randn(S, map(l -> ext[l], I)))
    return (arr(T, iC), arr(TA, iA), iA, arr(T, iB), iB, iC)
end

# The plan with every structurally eligible group split, whatever this host's L2:
# the planner's decision at a zero L2 threshold, replanned at its block extents.
function _sp_forced_plan(Cv, Av, iA, Bv, iB, iC; kw...)
    p = plan_contract(Cv, Av, iA, Bv, iB, iC; kw...)
    k, k_block, d, T = p.kernel, p.blocking.k_block, default_blocking(p.kernel), eltype(Cv)
    function split(g, i, eff, req)
        R = tile_size(k, i)
        return pack_split(g, p.kgroup, sliver_spec(k, i), sizeof(T), k_block, eff, cld(req, R) * R, d.k_block; l2bytes = 0)
    end
    (m_block, ms) = split(p.mgroup, 1, p.blocking.m_block, something(get(kw, :m_block, nothing), d.m_block))
    (n_block, ns) = split(p.ngroup, 2, p.blocking.n_block, d.n_block)
    q = plan_contract(Cv, Av, iA, Bv, iB, iC; kw..., kernel = k, m_block, n_block)
    @assert q.mgroup == p.mgroup && q.ngroup == p.ngroup
    return ContractPlan(
        k, q.mgroup, q.ngroup, q.kgroup, q.blocking, q.Astorage, q.Abase,
        q.Bstorage, q.Bbase, q.Cstorage, q.Cbase, q.atransform, q.btransform, q.workspace, ms, ns
    )
end

@testset "line-by-line packing: the planner splits A of intensli_7 past the cache" begin
    plan_of(d) = plan_contract(_sp_views(Float64, _SP_I7, ntuple(_ -> d, 6))...)
    p = plan_of(16)
    k, b = p.kernel, p.blocking
    splits(l2bytes) = is_split(pack_split(p.mgroup, p.kgroup, sliver_spec(k, 1), 8, b.k_block, b.m_block, b.m_block, default_blocking(k).k_block; l2bytes)[2])
    @test splits(2^20) && !splits(2^24)  # a 4 MB reuse window
    host = splits(QuasiStrided.split_capacity(target_profile(), true))
    @test is_split(p.mpack) == host
    host && @test _path_of(p) isa _NestPath{false, <:Any, (true, false)}
    @test _path_of(plan_of(4)) isa _NestPath{<:Any, <:Any, (false, false)}
end

@testset "line-by-line packing: contractions ($T)" for T in (Float64, Float32, ComplexF64, ComplexF32)
    kernels = T === ComplexF64 ? (nothing, OneMKernel(Val(4), Val(4), T), FMAddSubKernel(Val(8), Val(4), T)) : (nothing,)
    for (ind, ext, m_block) in (
                (_SP_I7, (8, 8, 8, 3, 6, 5), nothing), (_SP_I7, (6, 8, 11, 5, 10, 7), 48),
                (_SP_BOTH, (20, 16, 9, 20, 8), nothing), (_SP_BOTH, (13, 12, 5, 11, 12), 64),
            ), kernel in kernels, conjB in (T <: Complex ? (false, true) : (false,)), beta in (0, 0.7)
        T <: Complex && ind === _SP_BOTH && continue  # K steps within a page: complex never splits
        Cv, Av, iA, Bv, iB, iC = _sp_views(T, ind, ext)
        iszero(beta) && fill!(Cv, NaN)
        Cref = _lo_reference(iszero(beta) ? zero(Array(Cv)) : Array(Cv), Av, iA, Bv, iB, iC; conjB, alpha = 1.3, beta)
        plan = _sp_forced_plan(Cv, Av, iA, Bv, iB, iC; conjB, m_block, kernel)
        @test _path_of(plan) isa _NestPath{false, <:Any, (true, ind === _SP_BOTH)}
        execute!(plan, 1.3, beta)
        @test Array(Cv) ≈ Cref rtol = 100 * eps(real(T))
    end
    plan = _sp_forced_plan(_sp_views(T, T <: Complex ? _SP_I7 : _SP_BOTH, (20, 16, 9, 20, 8, 5))...)
    execute!(plan, 1, 0)
    @test (@allocated execute!(plan, 1, 0)) == 0 skip = (VERSION < v"1.11")
end
