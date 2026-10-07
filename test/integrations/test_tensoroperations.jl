# `QuasiStridedBackend` against TensorOperations' own backends and explicit references.

include("../helpers.jl")

using TensorOperations
using TensorOperations: StridedNative
using StridedViews: StridedView
using LinearAlgebra: Diagonal
using Bumper: default_buffer, @no_escape

const qsbackend = QuasiStrided.QuasiStridedBackend()
const to_native = StridedNative()

# NaN in C catches a kernel that computes `0 * C` instead of ignoring C when β == 0.
poison!(C) = fill!(C, convert(eltype(C), NaN))

const all_eltypes = (Float32, Float64, ComplexF32, ComplexF64)

const MATMUL_PAB = ((1,), (2,)), ((1,), (2,)), ((1, 2), ())

@testset "tensorcontract! agrees with StridedNative (eltype = $T)" for T in all_eltypes
    Random.seed!(1234567)
    @test QuasiStridedBackend <: TensorOperations.AbstractBackend
    # (A size, B size, pA, pB, pAB, C size): two different open/contracted orderings.
    cases = (
        ((3, 20, 5, 3, 4), (4, 6, 20, 3), ((3, 1, 4), (2, 5)), ((3, 1), (4, 2)), ((3, 1, 4), (5, 2)), (3, 5, 3, 6, 3)),
        ((4, 5, 6), (3, 5, 6), ((1,), (2, 3)), ((2, 3), (1,)), ((2, 1), ()), (3, 4)),
    )
    for (szA, szB, pA, pB, pAB, szC) in cases
        A, B = randn(T, szA), randn(T, szB)
        for (conjA, conjB) in (T <: Complex ? ((false, false), (true, true)) : ((false, false),)),
                (α, β) in ((one(T), zero(T)), (rand(T), zero(T)), (rand(T), rand(T)))
            Cn = randn(T, szC)
            iszero(β) && poison!(Cn)
            Cq = copy(Cn)
            tensorcontract!(Cn, A, pA, conjA, B, pB, conjB, pAB, α, β, to_native)
            tensorcontract!(Cq, A, pA, conjA, B, pB, conjB, pAB, α, β, qsbackend)
            @test all(isfinite, Cq)
            @test Cq ≈ Cn
        end
    end
end

# A plan that packs A line by line runs past the planner's predicted path.
@testset "tensorcontract! through a split plan" begin
    A, B = randn(16, 16, 16, 16, 16), randn(16, 16)
    Cn, Cq = zeros(16, 16, 16, 16, 16), zeros(16, 16, 16, 16, 16)
    pA, pB, pAB = ((1, 2, 3, 5), (4,)), ((1,), (2,)), ((4, 3, 2, 5, 1), ())
    tensorcontract!(Cn, A, pA, false, B, pB, false, pAB, 1.0, 0.0, to_native)
    tensorcontract!(Cq, A, pA, false, B, pB, false, pAB, 1.0, 0.0, qsbackend)
    @test Cq ≈ Cn
end

# `conjA`/`conjB` and each operand's `StridedView.op` compose by xor.
@testset "conjugation: flags x StridedView.op" begin
    Random.seed!(20260914)
    T = ComplexF64
    pA, pB, pAB = MATMUL_PAB
    M, N = randn(T, (4, 4)), randn(T, (4, 4))
    # `conj(::Matrix)` materialises; `conj(::StridedView)` only sets `op`.
    wrappers = (identity, adjoint, transpose, conj)
    variants(X) = vcat([w(X) for w in wrappers], [w(StridedView(X)) for w in wrappers])

    for Aw in variants(M), Bw in variants(N), conjA in (false, true), conjB in (false, true)
        Am, Bm = collect(Aw), collect(Bw)
        C = fill(convert(T, NaN), (4, 4))
        tensorcontract!(C, Aw, pA, conjA, Bw, pB, conjB, pAB, one(T), zero(T), qsbackend)
        @test C ≈ (conjA ? conj(Am) : Am) * (conjB ? conj(Bm) : Bm)
    end

    # Directly constructed views may carry any op, including `adjoint`, which
    # StridedViews' own arithmetic never produces for a `Number` eltype.
    for (op, conjugates) in ((identity, false), (conj, true), (transpose, false), (adjoint, true)),
            conjA in (false, true)
        Av = StridedView(M, size(M), strides(M), 0, op)
        C = fill(convert(T, NaN), (4, 4))
        tensorcontract!(C, Av, pA, conjA, N, pB, false, pAB, one(T), zero(T), qsbackend)
        @test C ≈ ((conjugates ⊻ conjA) ? conj(M) : M) * N
    end

    # α and β are never conjugated.
    α, β = convert(T, 0.75 - 1.25im), convert(T, -0.5 + 2.0im)
    C0 = randn(T, (4, 4))
    C = copy(C0)
    tensorcontract!(C, conj(StridedView(M)), pA, false, N, pB, false, pAB, α, β, qsbackend)
    @test C ≈ β * C0 + α * (conj(M) * N)
end

@testset "conjugated output view (eltype = $T)" for T in all_eltypes
    Random.seed!(271828)
    pA, pB, pAB = MATMUL_PAB
    A, B = randn(T, (4, 4)), randn(T, (4, 4))
    conjviews = (
        conj(StridedView(zeros(T, (4, 4)))), adjoint(zeros(T, (4, 4))),
        StridedView(zeros(T, (4, 4)), (4, 4), (1, 4), 0, adjoint),
    )
    for Cc in conjviews
        if T <: Complex
            @test_throws ArgumentError tensorcontract!(Cc, A, pA, false, B, pB, false, pAB, one(T), zero(T), qsbackend)
        else  # `op` is a no-op on real data, so a "conjugated" real output is fine
            @test collect(tensorcontract!(Cc, A, pA, false, B, pB, false, pAB, one(T), zero(T), qsbackend)) ≈ A * B
        end
    end
    for Cw in (StridedView(zeros(T, (4, 4)), (4, 4), (1, 4), 0, transpose), transpose(zeros(T, (4, 4))))
        @test collect(tensorcontract!(Cw, A, pA, false, B, pB, false, pAB, one(T), zero(T), qsbackend)) ≈ A * B
    end
end

@testset "tensorcontract! edge cases (eltype = $T)" for T in all_eltypes
    Random.seed!(13579)
    # (A, B, pA, pB, pAB, C, conjA, conjB): outer product, full contraction, strided views.
    cases = (
        (randn(T, (3, 4)), randn(T, (5,)), ((1, 2), ()), ((), (1,)), ((3, 1), (2,)), (5, 3, 4), true, false),
        (randn(T, (3, 4)), randn(T, (4, 3)), ((), (1, 2)), ((2, 1), ()), ((), ()), (), false, false),
        (
            view(randn(T, (6, 8)), 1:2:6, 1:2:8), view(randn(T, (8, 10)), 1:2:8, 1:2:10),
            MATMUL_PAB..., (3, 5), false, true,
        ),
    )
    for (A, B, pA, pB, pAB, szC, conjA, conjB) in cases
        Cn = fill(convert(T, NaN), szC)
        Cq = copy(Cn)
        tensorcontract!(Cn, A, pA, conjA, B, pB, conjB, pAB, one(T), zero(T), to_native)
        R = tensorcontract!(Cq, A, pA, conjA, B, pB, conjB, pAB, one(T), zero(T), qsbackend)
        @test R === Cq
        @test size(Cq) == szC
        @test all(isfinite, Cq)
        @test Cq ≈ Cn
    end
end

@testset "@tensor / ncon integration" begin
    Random.seed!(112233)
    A, B, C = randn(ComplexF64, (5, 5, 5, 5)), randn(ComplexF64, (5, 5, 5)), randn(ComplexF64, (5, 5, 5))
    @tensor backend = qsbackend D[a, b, c, d] := A[a, e, c, f] * B[g, d, e] * C[g, f, b]
    @tensor Dref[a, b, c, d] := A[a, e, c, f] * B[g, d, e] * C[g, f, b]
    @test D ≈ Dref
    network = [[-1, 1, -3, 2], [3, -4, 1], [3, 2, -2]]
    @test ncon([A, B, C], network; backend = qsbackend) ≈ ncon([A, B, C], network)
end

@testset "hard-reject: ineligible eltypes and non-strided operands" begin
    pA, pB, pAB = MATMUL_PAB
    # (eltype A, eltype B, eltype C): types outside the four, and complex into real.
    for (TA, TB, TC) in (
            (Float16, Float16, Float16), (Complex{Float16}, Complex{Float16}, Complex{Float16}),
            (Complex{Int}, Complex{Int}, Complex{Int}), (Complex{BigFloat}, Complex{BigFloat}, Complex{BigFloat}),
            (Float64, Float32, Float16), (ComplexF64, Float64, Float64),
        )
        A, B, C = ones(TA, (3, 4)), ones(TB, (4, 5)), zeros(TC, (3, 5))
        @test_throws ArgumentError tensorcontract!(C, A, pA, false, B, pB, false, pAB, 1, 0, qsbackend)
    end
    D4, D5 = Diagonal(randn(4)), Diagonal(randn(5))
    @test_throws ArgumentError tensorcontract!(zeros(4, 5), D4, pA, false, randn(4, 5), pB, false, pAB, 1, 0, qsbackend)
    @test_throws ArgumentError tensorcontract!(zeros(4, 5), randn(4, 5), pA, false, D5, pB, false, pAB, 1, 0, qsbackend)
    @test_throws ArgumentError QuasiStrided.QuasiStridedBackend(accumulator = Float16)
    @test_throws ArgumentError tensorcontract!(
        zeros(4, 5), randn(4, 4), pA, false, randn(4, 5), pB, false, pAB, 1, 0,
        QuasiStrided.QuasiStridedBackend{Float16}()
    )
end

@testset "@tensor with mixed eltypes and an accumulator" begin
    Random.seed!(4321)
    A32, B64, Bc = randn(Float32, 6, 7, 5), randn(7, 8), randn(ComplexF64, 7, 8)
    A64 = Float64.(A32)
    @tensor backend = qsbackend D[a, b, c] := A32[a, k, c] * B64[k, b]
    @tensor Dref[a, b, c] := A64[a, k, c] * B64[k, b]
    @test eltype(D) === Float64 && D ≈ Dref
    @tensor backend = qsbackend E[a, b, c] := conj(A32[a, k, c]) * Bc[k, b]
    @tensor Eref[a, b, c] := A64[a, k, c] * Bc[k, b]
    @test eltype(E) === ComplexF64 && E ≈ Eref
    B32 = randn(Float32, 7, 8)
    @tensor backend = QuasiStrided.QuasiStridedBackend(accumulator = Float64) F[a, b, c] := A32[a, k, c] * B32[k, b]
    B32w = Float64.(B32)
    @tensor Fref[a, b, c] := A64[a, k, c] * B32w[k, b]
    @test eltype(F) === Float32 && all(abs.(F .- Float32.(Fref)) .<= eps.(Float32.(Fref)))
end

@testset "hard-reject: C aliasing an input (eltype = $T)" for T in (Float64, ComplexF64)
    pA, pB, pAB = MATMUL_PAB
    A, B = randn(T, (4, 4)), randn(T, (4, 4))
    M = randn(T, (8, 4))
    # Regression: Base's `mightalias` misses a `PermutedDimsArray` of the input.
    P = randn(T, (4, 4))
    Cpd = PermutedDimsArray(P, (2, 1))
    for (C, A_, B_) in ((A, A, B), (B, A, B), (view(M, 3:6, :), view(M, 1:4, :), B), (Cpd, P, B), (Cpd, A, P))
        @test_throws ArgumentError tensorcontract!(C, A_, pA, false, B_, pB, false, pAB, 1, 0, qsbackend)
    end
    # Aliased and (for complex) conjugated: either rejection is fine.
    @test_throws ArgumentError tensorcontract!(P', P, pA, false, B, pB, false, pAB, 1, 0, qsbackend)
end

@testset "tensoradd! / tensortrace! forward to StridedNative" begin
    A = randn(3, 4)
    C1, C2, C3 = zeros(4, 3), zeros(4, 3), zeros(4, 3)
    @tensor backend = qsbackend C1[j, i] = A[i, j]
    @tensor backend = StridedNative() C2[j, i] = A[i, j]
    @tensor backend = qsbackend allocator = TensorOperations.ManualAllocator() C3[j, i] = A[i, j]
    @test C1 == C2 == C3

    A3 = randn(4, 3, 3)
    c1, c2 = zeros(4), zeros(4)
    @tensor backend = qsbackend c1[i] = A3[i, j, j]
    @tensor backend = StridedNative() c2[i] = A3[i, j, j]
    @test c1 == c2
end

@testset "tensorcontract!: allocator-routed workspace, released per call" begin
    Random.seed!(13571113)
    A, B = randn(20, 30), randn(30, 25)
    buffer = TensorOperations.BufferAllocator()
    for _ in 1:2
        C1, C2, C3 = zeros(20, 25), zeros(20, 25), zeros(20, 25)
        @tensor backend = qsbackend allocator = TensorOperations.ManualAllocator() C1[i, j] = A[i, k] * B[k, j]
        @tensor backend = qsbackend allocator = buffer C2[i, j] = A[i, k] * B[k, j]
        @no_escape begin
            @tensor backend = qsbackend allocator = default_buffer() C3[i, j] = A[i, k] * B[k, j]
        end
        @test C1 ≈ C2 ≈ C3 ≈ A * B
    end
    @test isempty(buffer)
end

@testset "tensorcontract!: the default path allocates no more than planning" begin
    A, B, C = randn(20, 30), randn(30, 25), zeros(20, 25)
    run_once() = TO.tensorcontract!(C, A, ((1,), (2,)), false, B, ((1,), (2,)), false, ((1, 2), ()), 1.0, 0.0, qsbackend)
    Cv, Av, Bv = StridedView(C), StridedView(A), StridedView(B)
    plan_once() = plan_contract(Cv, Av, (1, -1), Bv, (-1, 2), (1, 2))
    run_once(); plan_once()
    @test C ≈ A * B
    @test (@allocated run_once()) <= (@allocated plan_once()) skip = (VERSION < v"1.11")
end
