# Randomized agreement of the five-loop `execute!` with a dense matmul and with
# `execute_tilewise!`, at block sizes small enough to split every dimension,
# over real and complex kernels, alpha/beta and the conjugation flags and ops.
# helpers.jl binds `plan_contract`/`execute!` unqualified in this scope, so
# names here are written `QuasiStrided.<name>`.

# Agreement is up to summation-order rounding.
_macro_rtol(::Type{T}, Ka::Integer) where {T} = 50 * max(Ka, 1) * eps(real(T))

# Plan for C[m,n] = A[m,k] * B[k,n], with operands as `op`-carrying views. The
# 5-argument constructor keeps `op` on the view instead of materializing.
_macro_op_view(M::AbstractMatrix, op) = StridedView(M, size(M), strides(M), 0, op)
function _macro_plan(Cmat, Amat, Bmat, kernel, m_block, k_block, n_block; conjA = false, conjB = false, opA = identity, opB = identity)
    return QuasiStrided.plan_contract(
        StridedView(Cmat), _macro_op_view(Amat, opA), (1, 2), _macro_op_view(Bmat, opB), (2, 3), (1, 3);
        kernel = kernel, conjA = conjA, conjB = conjB, m_block = m_block, k_block = k_block, n_block = n_block
    )
end

# The engine's conjugation rule re-derived independently: the flag and the
# view's `op` compose by XOR; real eltypes never conjugate.
_macro_op_conjugates(op) = op === conj || op === adjoint
_macro_conjugated(::Type{T}, flag::Bool, op) where {T} = (T <: Complex) && (flag ⊻ _macro_op_conjugates(op))

const _MACRO_OPS = (identity, conj, adjoint, transpose)
const _MACRO_SHAPES = ((Val(4), Val(3)), (Val(8), Val(6)))

# Every constructible (kernel type, shape) for `T`; constructors reject shapes
# their lane width cannot tile.
function _macro_kernels(::Type{T}) where {T}
    ctors = T <: Complex ? (QuasiStrided.PlanarKernel, QuasiStrided.OneMKernel) : (ScalarKernel, SIMDKernel)
    ks = Any[]
    for ctor in ctors, (m, n) in _MACRO_SHAPES
        k = try
            ctor(m, n, T)
        catch err
            err isa ArgumentError || rethrow()
            nothing
        end
        k === nothing || push!(ks, k)
    end
    return ks
end

# Scalars that include the 0 and 1 shortcuts and are genuinely complex
# otherwise, so a dropped imaginary part cannot pass.
function _macro_scalar(rng, ::Type{T}) where {T}
    r = rand(rng)
    r < 0.1 && return zero(T)
    r < 0.2 && return one(T)
    return T <: Complex ? T(rand(rng, -3.0:0.5:3.0), rand(rng, -3.0:0.5:3.0)) : T(rand(rng, -3.0:0.5:3.0))
end

@testset "macro driver: randomized agreement vs dense matmul and the oracle ($T)" for T in (Float64, Float32, ComplexF64, ComplexF32)
    rng = MersenneTwister(0x5A17_D817)
    kernels = _macro_kernels(T)
    @test !isempty(kernels)
    for kernel in kernels, i in 1:(T <: Complex ? 12 : 5)
        Ma, Ka, Na = rand(rng, 1:37, 3)
        m_block, k_block, n_block = rand(rng, 1:13, 3)
        conjA, conjB = rand(rng, Bool, 2)
        opA, opB = rand(rng, _MACRO_OPS, 2)
        alpha, beta = _macro_scalar(rng, T), _macro_scalar(rng, T)
        Amat, Bmat = randn(rng, T, Ma, Ka), randn(rng, T, Ka, Na)
        # A NaN-poisoned C whenever beta == 0: it must never be read.
        Cstart = iszero(beta) ? fill(T(NaN), Ma, Na) : randn(rng, T, Ma, Na)
        kw = (; conjA, conjB, opA, opB)

        Aeff = _macro_conjugated(T, conjA, opA) ? conj.(Amat) : Amat
        Beff = _macro_conjugated(T, conjB, opB) ? conj.(Bmat) : Bmat
        expected = iszero(beta) ? alpha .* (Aeff * Beff) : alpha .* (Aeff * Beff) .+ beta .* Cstart

        C = copy(Cstart)
        QuasiStrided.execute!(_macro_plan(C, Amat, Bmat, kernel, m_block, k_block, n_block; kw...), alpha, beta)
        Ctw = copy(Cstart)
        QuasiStrided.execute_tilewise!(_macro_plan(Ctw, Amat, Bmat, kernel, m_block, k_block, n_block; kw...), alpha, beta)
        ok = isapprox(C, expected; rtol = _macro_rtol(T, Ka)) && isapprox(C, Ctw; rtol = _macro_rtol(T, Ka))
        ok || @error "macro driver case $i failed" kernel Ma Ka Na m_block k_block n_block kw alpha beta
        @test ok
    end
end

@testset "macro driver: permuted, zero-stride, negative-stride and sliced views" begin
    # A[a,k,b] independent of b (zero stride), viewed permuted as (k,b,a);
    # B[k,n] reversed along k; C a slice with a nonzero offset.
    a_n, k_n, b_n, n_n = 7, 11, 5, 9
    Aperm = permutedims(StridedView(vec(randn(a_n, k_n)), (a_n, k_n, b_n), (1, a_n, 0), 0), (2, 3, 1))
    Bneg = StridedView(randn(k_n * n_n), (k_n, n_n), (-1, k_n), k_n - 1)
    Cbig = randn(a_n + 2, n_n + 3, b_n + 1)
    Csub = view(Cbig, 2:(a_n + 1), 2:(n_n + 1), 1:b_n)
    Cstart = copy(Csub)
    Cref = _brute_ref(Array(Aperm), (2, 3, 1), Array(Bneg), (2, 4), Cstart, (1, 4, 3), 1.3, 0.6)
    plan = QuasiStrided.plan_contract(
        StridedView(Csub), Aperm, (2, 3, 1), Bneg, (2, 4), (1, 4, 3);
        kernel = ScalarKernel(Val(4), Val(3), Float64), m_block = 3, k_block = 4, n_block = 3
    )
    QuasiStrided.execute!(plan, 1.3, 0.6)
    @test isapprox(Array(Csub), Cref; rtol = _macro_rtol(Float64, k_n))
end

@testset "macro driver: irregular sliver at a nonzero offset (multi-label M)" begin
    # M = (a, q): A's map folds across the a/q boundary, C's (padded) does not,
    # so the second M sliver of the first block is irregular.
    a_n, q_n, k_n, n_n = 7, 2, 5, 3
    Amat, Bmat = randn(a_n, q_n, k_n), randn(k_n, n_n)
    Cstart = randn(a_n, q_n, n_n)
    Cref = _brute_ref(Amat, (1, 2, 3), Bmat, (3, 4), Cstart, (1, 2, 4), 1.7, -0.4)
    for run! in (QuasiStrided.execute!, QuasiStrided.execute_tilewise!)
        Cbig = randn(a_n + 3, q_n, n_n)
        Csub = view(Cbig, 1:a_n, :, :)
        Csub .= Cstart
        plan = QuasiStrided.plan_contract(
            StridedView(Csub), StridedView(Amat), (1, 2, 3), StridedView(Bmat), (3, 4), (1, 2, 4);
            kernel = ScalarKernel(Val(4), Val(3), Float64), m_block = 8, k_block = 5, n_block = 3
        )
        run!(plan, 1.7, -0.4)
        @test isapprox(Array(Csub), Cref; rtol = _macro_rtol(Float64, k_n))
    end
end

@testset "macro driver: several blocks at the default blocking ($(nameof(typeof(k))){$T})" for T in (ComplexF64, ComplexF32), k in _macro_kernels(T)
    QuasiStrided.tile_size(k)[1] == 8 || continue
    b = QuasiStrided.default_blocking(k)
    Ma, Ka, Na = b.m_block + QuasiStrided.tile_size(k)[1], b.k_block + 1, b.n_block + QuasiStrided.tile_size(k)[2]
    Amat, Bmat = randn(MersenneTwister(1), T, Ma, Ka), randn(MersenneTwister(2), T, Ka, Na)
    Cstart = randn(MersenneTwister(3), T, Ma, Na)
    C = copy(Cstart)
    alpha, beta = T(1.5, -0.25), T(-0.5, 0.75)
    # Conjugation from both sources: the flag on A, a conjugating op on B.
    plan = _macro_plan(C, Amat, Bmat, k, nothing, nothing, nothing; conjA = true, opB = adjoint)
    @test plan.blocking.m_block < Ma && plan.blocking.k_block < Ka && plan.blocking.n_block < Na
    QuasiStrided.execute!(plan, alpha, beta)
    @test isapprox(C, alpha .* (conj.(Amat) * conj.(Bmat)) .+ beta .* Cstart; rtol = _macro_rtol(T, Ka))
end

@testset "macro driver: conjugation sources compose by XOR ($T)" for T in (ComplexF64, ComplexF32)
    kernel = first(_macro_kernels(T))
    Amat, Bmat = randn(MersenneTwister(1), T, 11, 9), randn(MersenneTwister(2), T, 9, 7)
    rtol = _macro_rtol(T, 9)
    # (conjA, opA, opB) => (A conjugated, B conjugated). The first two pin the
    # cases a `||` rule or a `op === conj` test would get wrong.
    for ((conjA, opA, opB), (ca, cb)) in (
            (true, conj, identity) => (false, false), (false, adjoint, identity) => (true, false),
            (false, identity, transpose) => (false, false),
        )
        C = zeros(T, 11, 7)
        QuasiStrided.execute!(_macro_plan(C, Amat, Bmat, kernel, 4, 3, 4; conjA, opA, opB), one(T), zero(T))
        Aeff, Beff = ca ? conj.(Amat) : Amat, cb ? conj.(Bmat) : Bmat
        @test isapprox(C, Aeff * Beff; rtol)
        @test !isapprox(C, (ca ? Amat : conj.(Amat)) * Beff; rtol)
    end
end

@testset "macro driver: a conjugating op on C is rejected ($T)" for T in (ComplexF64, Float64)
    kernel = T <: Complex ? first(_macro_kernels(T)) : ScalarKernel(Val(4), Val(3), T)
    Amat, Bmat = randn(MersenneTwister(1), T, 9, 7), randn(MersenneTwister(2), T, 7, 6)
    for op in _MACRO_OPS
        C = zeros(T, 9, 6)
        mkplan() = QuasiStrided.plan_contract(
            _macro_op_view(C, op), _macro_op_view(Amat, op), (1, 2), _macro_op_view(Bmat, op), (2, 3), (1, 3);
            kernel = kernel, conjA = true, conjB = true, m_block = 4, k_block = 3, n_block = 4
        )
        if T <: Complex && _macro_op_conjugates(op)
            @test_throws ArgumentError mkplan()
        else
            plan = mkplan()
            QuasiStrided.execute!(plan, one(T), zero(T))
            # For a real eltype nothing conjugates, whatever the flags and ops say.
            T <: Real && @test plan.atransform === identity === plan.btransform
            Aeff = _macro_conjugated(T, true, op) ? conj.(Amat) : Amat
            Beff = _macro_conjugated(T, true, op) ? conj.(Bmat) : Bmat
            @test isapprox(C, Aeff * Beff; rtol = _macro_rtol(T, 7))
        end
    end
end
