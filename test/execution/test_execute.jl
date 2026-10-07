include("helpers.jl")

@testset "contract!: worked fixture, permuted and sliced views" begin
    A, B, Cref = _worked_fixture()
    C = zeros(3, 4, 2)
    contract!(StridedView(C), 1.0, StridedView(A), INDA, StridedView(B), INDB, 0.0, INDC)
    @test C ≈ Cref

    # Ap[k,b,a] == A[a,k,b], Bp[n,k] == B[k,n].
    C = zeros(3, 4, 2)
    Ap, Bp = permutedims(A, (2, 3, 1)), permutedims(B, (2, 1))
    contract!(StridedView(C), 1.0, StridedView(Ap), (2, 3, 1), StridedView(Bp), (4, 2), 0.0, INDC)
    @test C ≈ Cref

    # Nonzero view offsets on A and C.
    A4 = reshape(collect(1.0:40.0), 4, 5, 2)
    Cfull = zeros(4, 4, 2)
    Av, Cv = StridedView(view(A4, 2:4, :, :)), StridedView(view(Cfull, 2:4, :, :))
    @test offset(Av) != 0
    contract!(Cv, 1.0, Av, INDA, StridedView(B), INDB, 0.0, INDC)
    @test Array(Cv) ≈ [sum(A4[a, k, b] * B[k, n] for k in 1:5) for a in 2:4, n in 1:4, b in 1:2]
    @test all(iszero, Cfull[1, :, :])
end

@testset "execute!: short-circuits and empty output" begin
    Ma, Ka, Na = 5, 6, 4
    for (Ka, alpha) in ((0, 1.0), (Ka, 0.0)), beta in (0.5, 0.0)
        # A and B are poisoned: neither may be read.
        mk = _mm_maker(Float64, Ma, Ka, Na, 1; ABfill = NaN)
        C, plan = _run_fresh(execute!, mk, alpha, beta)
        @test C == beta .* Array(mk()[1])
        Ka == 0 && @test _path_of(plan) isa QuasiStrided.ScalePath
    end
    # beta = 0 never reads a NaN C, across several blocks.
    C, _ = _run_fresh(execute!, _mm_maker(Float64, 11, 10, 9, 2; Cfill = NaN), 2.5, 0.0; m_block = 4, k_block = 4, n_block = 4)
    @test C ≈ _ref_of(_mm_maker(Float64, 11, 10, 9, 2), 2.5, 0.0)
    # Empty M or N: a no-op.
    for (M, N) in ((3, 0), (0, 3))
        C, plan = _run_fresh(execute!, _mm_maker(Float64, M, 4, N, 3; Cfill = NaN), 1.0, 2.0)
        @test size(C) == (M, N)
        @test _path_of(plan) isa QuasiStrided.EmptyPath
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
    end
end

@testset "execute!: allocation-free in steady state" begin
    Ma, Ka, Na = 19, 23, 17
    Amat, Bmat = randn(MersenneTwister(1), Ma, Ka), randn(MersenneTwister(2), Ka, Na)
    kernel = SIMDKernel(Val(8), Val(6), Float64, Val(4))
    # Default everything; several blocks with tail tiles in M and N (the
    # vectorized store's row tail); a conj transform, which the real path never
    # builds but which must still cross `pack_sliver!` concretely typed.
    Cmat = zeros(Ma, Na)
    p = _mm_plan(Cmat, Amat, Bmat)
    plans = (
        p,
        _mm_plan(Cmat, Amat, Bmat; kernel = kernel, m_block = 16, k_block = 5, n_block = 8),
        ContractPlan(p; atransform = conj, btransform = conj),
    )
    @test parent(StridedView(Cmat)) isa DenseVector{Float64}
    for plan in plans
        allocs = _steady_allocs!(execute!, plan, Cmat)
        @test Cmat ≈ Amat * Bmat
        @test allocs == 0 skip = (VERSION < v"1.11")
    end
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
    @test isempty(_ws_nonconcrete_types(execute!, (typeof(plan), Float64, Float64)))
    @test isconcretetype(only(Base.return_types(execute!, (typeof(plan), Float64, Float64))))
end

@testset "execute! still allocates nothing in steady state" begin
    Ma, Ka, Na = 40, 21, 30
    A = randn(Ma, Ka); B = randn(Ka, Na); C = zeros(Ma, Na)
    p = plan_contract(
        StridedView(C), StridedView(A), (1, 2), StridedView(B), (2, 3), (1, 3)
    )
    execute!(p, 1.0, 0.0)
    execute!(p, 1.0, 0.0)
    allocs = @allocated execute!(p, 1.0, 0.0)
    @test C ≈ A * B
    @test allocs == 0 skip = (VERSION < v"1.11")

    # Permuted A with a zero stride, reversed B, sliced C: the non-ramp path,
    # at both real eltypes' shipped defaults.
    for T in (Float64, Float32)
        Cv, Av, iA, Bv, iB, iC = scattered_fixture(T)
        ps = plan_contract(Cv, Av, iA, Bv, iB, iC)
        execute!(ps, one(T), zero(T))
        execute!(ps, one(T), zero(T))
        @test (@allocated execute!(ps, one(T), zero(T))) == 0 skip = (VERSION < v"1.11")
        @test Cv ≈ _lo_reference(zeros(T, size(Cv)), Av, iA, Bv, iB, iC)
    end

    # M, N and K composites ordered differently per operand: scattered tile axes.
    A4 = randn(5, 6, 7, 9); B4 = randn(7, 6, 11, 3); C4 = zeros(3, 9, 11, 5)
    p4 = plan_contract(
        StridedView(C4), StridedView(A4), (1, 2, 3, 4), StridedView(B4), (3, 2, 5, 6), (6, 4, 5, 1)
    )
    execute!(p4, 1.0, 0.5)
    execute!(p4, 1.0, 0.5)
    @test (@allocated execute!(p4, 1.0, 0.5)) == 0 skip = (VERSION < v"1.11")
end
