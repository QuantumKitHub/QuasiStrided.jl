include("../execution/helpers.jl")

@testset "driver: label validation errors" begin
    Amat = randn(3, 4)
    Bmat = randn(4, 5)
    Cmat = zeros(3, 5)
    Av, Bv, Cv = StridedView(Amat), StridedView(Bmat), StridedView(Cmat)

    # Mismatched shared-label length: B's k axis is 6, A's is 4.
    Bv_bad = StridedView(randn(6, 5))
    @test_throws DimensionMismatch contract!(Cv, 1.0, Av, (1, 2), Bv_bad, (2, 3), 0.0, (1, 3))

    # Diagonal: repeated label within indA, then within indC.
    Avsq = StridedView(randn(4, 4))
    @test_throws ArgumentError contract!(Cv, 1.0, Avsq, (1, 1), Bv, (2, 3), 0.0, (1, 3))
    @test_throws ArgumentError contract!(Cv, 1.0, Av, (1, 2), Bv, (2, 3), 0.0, (1, 1, 3)[1:2])

    # Label in indC absent from both indA and indB.
    @test_throws ArgumentError contract!(Cv, 1.0, Av, (1, 2), Bv, (2, 3), 0.0, (1, 9))

    # Dangling label: only in indA, then only in indB. Distinct code paths
    # (the A loop vs. the B loop in classify_labels).
    @test_throws ArgumentError contract!(Cv, 1.0, Av, (1, 2), Bv, (5, 3), 0.0, (1, 3))
    @test_throws ArgumentError contract!(Cv, 1.0, Av, (1, 2), Bv, (2, 9), 0.0, (1, 3))

    # Label present in all three (batch-like), unsupported.
    @test_throws ArgumentError contract!(Cv, 1.0, Av, (1, 2), Bv, (2, 3), 0.0, (2, 3))
end

@testset "driver: eltype validation errors" begin
    mk(TA, TB, TC) = (StridedView(zeros(TC, 3, 5)), StridedView(ones(TA, 3, 4)), (1, 2), StridedView(ones(TB, 4, 5)), (2, 3), (1, 3))
    for eltypes in ((Float16, Float16, Float16), (Float64, Float64, BigFloat), (ComplexF64, Float64, Float64))
        @test_throws ArgumentError plan_contract(mk(eltypes...)...)
    end
end

@testset "plan_contract: m_block/k_block/n_block keywords are validated and rounded" begin
    kernel = ScalarKernel(Val(4), Val(3), Float64)
    Ma, Ka, Na = 9, 10, 8
    Amat, Bmat, Cmat = randn(Ma, Ka), randn(Ka, Na), zeros(Ma, Na)
    Av, Bv, Cv = StridedView(Amat), StridedView(Bmat), StridedView(Cmat)
    indA, indB, indC = (1, 2), (2, 3), (1, 3)

    @test_throws ArgumentError plan_contract(Cv, Av, indA, Bv, indB, indC; kernel = kernel, m_block = 0)
    @test_throws ArgumentError plan_contract(Cv, Av, indA, Bv, indB, indC; kernel = kernel, k_block = 0)
    @test_throws ArgumentError plan_contract(Cv, Av, indA, Bv, indB, indC; kernel = kernel, n_block = -3)

    # m_block=5 with MR=4 rounds up to 8, then clamps to roundup(Ma=9,4)=12 -> 8.
    plan = _mm_plan(Cmat, Amat, Bmat; kernel = kernel, m_block = 5, k_block = 100, n_block = 100)
    @test plan.blocking.m_block == 8
    @test plan.blocking.k_block == 10  # clamped to k_length
    @test plan.blocking.n_block == 9   # NR=3: roundup(8,3)=9, requested 100 clamped down to that
end

@testset "op_conjugates is a total table with a throwing fallback" begin
    @test QuasiStrided.op_conjugates(identity) === false
    @test QuasiStrided.op_conjugates(conj) === true
    @test QuasiStrided.op_conjugates(transpose) === false   # elementwise identity
    @test QuasiStrided.op_conjugates(adjoint) === true
    @test_throws ArgumentError QuasiStrided.op_conjugates(sin)

    # A real element type is conjugated by nothing; a complex one composes
    # the flag and the op with XOR.
    for op in (identity, conj, transpose, adjoint), flag in (false, true)
        v = StridedView(randn(16), (4, 4), (1, 4), 0, op)
        @test QuasiStrided.isconj(v, flag) === false
    end
    for (op, oc) in ((identity, false), (conj, true), (transpose, false), (adjoint, true)),
            flag in (false, true)
        v = StridedView(randn(ComplexF64, 16), (4, 4), (1, 4), 0, op)
        @test QuasiStrided.isconj(v, flag) === (flag ⊻ oc)
    end
end

@testset "conjA/conjB add no specialization on the real path" begin
    Random.seed!(97531)
    Ma, Ka, Na = 12, 9, 7
    Amat, Bmat = randn(Ma, Ka), randn(Ka, Na)
    Cmat = zeros(Ma, Na)
    base = _mm_plan(Cmat, Amat, Bmat)

    for ca in (false, true), cb in (false, true)
        plan = _mm_plan(Cmat, Amat, Bmat; conjA = ca, conjB = cb)
        @test typeof(plan.atransform) === typeof(identity)
        @test typeof(plan.btransform) === typeof(identity)
        @test typeof(plan) === typeof(base)
        fill!(Cmat, 0.0)
        execute!(plan, 1.0, 0.0)
        @test Cmat ≈ Amat * Bmat
    end

    # A real eltype collapses a view's `conj` op to identity.
    Av = conj(StridedView(Amat))
    @test Av.op === identity
    plan = plan_contract(
        StridedView(Cmat), Av, (1, 2), StridedView(Bmat), (2, 3), (1, 3); conjA = true
    )
    @test typeof(plan) === typeof(base)
    fill!(Cmat, 0.0)
    execute!(plan, 1.0, 0.0)
    @test Cmat ≈ Amat * Bmat
end

@testset "plan_contract rejects a conjugated output, but not a real adjoint" begin
    Random.seed!(2469)
    Amat, Bmat = randn(6, 5), randn(5, 4)

    # A real adjoint output has op === identity and is accepted.
    Cadj = adjoint(zeros(4, 6))
    Cv = StridedView(Cadj)
    @test Cv.op === identity
    plan = plan_contract(Cv, StridedView(Amat), (1, 2), StridedView(Bmat), (2, 3), (1, 3))
    execute!(plan, 1.0, 0.0)
    @test Cadj ≈ Amat * Bmat

    # A conjugated complex output is rejected (checked by message).
    Ac, Bc = randn(ComplexF64, 6, 5), randn(ComplexF64, 5, 4)
    Cc = zeros(ComplexF64, 24)
    for op in (conj, adjoint)
        Ccv = StridedView(Cc, (6, 4), (1, 6), 0, op)
        err = try
            plan_contract(Ccv, StridedView(Ac), (1, 2), StridedView(Bc), (2, 3), (1, 3))
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("conjugated", err.msg)
    end
    # A non-conjugating op on a complex output is accepted.
    Ctv = StridedView(Cc, (6, 4), (1, 6), 0, transpose)
    execute!(plan_contract(Ctv, StridedView(Ac), (1, 2), StridedView(Bc), (2, 3), (1, 3)), 1.0, 0.0)
    @test reshape(Cc, 6, 4) ≈ Ac * Bc
    # StridedViews bounds `op` to exactly the four functions `op_conjugates`
    # tabulates; fail here if that ever widens.
    @test_throws TypeError StridedView(Cc, (6, 4), (1, 6), 0, sin)
    Fbound = fieldtype(typeof(StridedView(Cc, (6, 4), (1, 6), 0, conj)), :op)
    @test Fbound === typeof(conj)
    optypes = Base.unwrap_unionall(StridedView).parameters[4].ub
    @test Set(Base.uniontypes(optypes)) ==
        Set((typeof(identity), typeof(conj), typeof(transpose), typeof(adjoint)))
end

@testset "plan_contract and contract! reject an output aliased with an input" begin
    M = randn(8, 4)
    Cv, Av, Bv = StridedView(M)[1:4, :], StridedView(M)[5:8, :], StridedView(randn(4, 4))
    @test_throws ArgumentError plan_contract(Cv, Av, (1, 2), Bv, (2, 3), (1, 3))
    @test_throws ArgumentError plan_contract(Cv, Bv, (1, 2), Av, (2, 3), (1, 3))
    @test_throws ArgumentError contract!(Cv, 1.0, Av, (1, 2), Bv, (2, 3), 0.0, (1, 3))
end

@testset "plan_contract allocation stays under a ceiling" begin
    # A ceiling that a `Set`/`Vector`-based planner would exceed several-fold.
    A = randn(64, 64); B = randn(64, 64); C = zeros(64, 64)
    Av, Bv, Cv = StridedView(A), StridedView(B), StridedView(C)
    p = plan_contract(Cv, Av, (1, 2), Bv, (2, 3), (1, 3))
    ws() = QuasiStrided.ContractWorkspace(Float64, p.kernel, p.blocking)
    ws()
    f() = plan_contract(Cv, Av, (1, 2), Bv, (2, 3), (1, 3))
    f()
    @test (@allocated f()) <= 3000 + (@allocated ws())
end

@testset "the path modes override path selection" begin
    pd(; kw...) = plan_contract(_mm_maker(Float64, 1, 64, 9, 1)()...; kw...)
    po(; kw...) = plan_contract(_outer_maker(Float64, 16, 9, 1)()...; kw...)
    pu(; kw...) = plan_contract(_mm_maker(Float64, 20, 12, 9, 1)()...; kw...)
    ps(; kw...) = plan_contract(_mm_maker(Float64, 20, 12, 9, 1; B = :transposed)()...; kw...)
    @test _dot_takes(pd()) && _outer_takes(po()) && _ub_takes(pu()) && !_ub_takes(ps())
    @test !_dot_takes(pd(; path_modes = NEST_ONLY)) && !_outer_takes(po(; path_modes = NEST_ONLY)) &&
        !_ub_takes(pu(; path_modes = NEST_ONLY))
    @test _ub_takes(ps(; path_modes = QuasiStrided.PathModes(unpacked_b = :always)))
end
