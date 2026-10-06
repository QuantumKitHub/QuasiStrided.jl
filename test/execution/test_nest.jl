# Randomized agreement of the five-loop nest with a dense matmul, at block
# sizes small enough to split every dimension, over real and complex kernels,
# alpha/beta and the conjugation flags and ops; the once-per-block storage
# bounds checks; and the closed-form (`affine_ramp`) block description against
# the buffer path.

include("helpers.jl")

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
_macro_conj_op(op) = op === conj || op === adjoint
_macro_conjugated(::Type{T}, flag::Bool, op) where {T} = (T <: Complex) && (flag ⊻ _macro_conj_op(op))

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

@testset "macro driver: randomized agreement vs dense matmul ($T)" for T in (Float64, Float32, ComplexF64, ComplexF32)
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
        ok = isapprox(C, expected; rtol = _macro_rtol(T, Ka))
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
    Cbig = randn(a_n + 3, q_n, n_n)
    Csub = view(Cbig, 1:a_n, :, :)
    Csub .= Cstart
    plan = QuasiStrided.plan_contract(
        StridedView(Csub), StridedView(Amat), (1, 2, 3), StridedView(Bmat), (3, 4), (1, 2, 4);
        kernel = ScalarKernel(Val(4), Val(3), Float64), m_block = 8, k_block = 5, n_block = 3
    )
    QuasiStrided.execute!(plan, 1.7, -0.4)
    @test isapprox(Array(Csub), Cref; rtol = _macro_rtol(Float64, k_n))
end

@testset "macro driver: several blocks at the default blocking ($(nameof(typeof(k))){$T})" for T in (ComplexF64, ComplexF32), k in _macro_kernels(T)
    QuasiStrided.tile_size(k, 1) == 8 || continue
    b = QuasiStrided.default_blocking(k)
    Ma, Ka, Na = b.m_block + QuasiStrided.tile_size(k, 1), b.k_block + 1, b.n_block + QuasiStrided.tile_size(k, 2)
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
        if T <: Complex && _macro_conj_op(op)
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

@testset "classify_slivers! accumulates the TRUE block range" begin
    # The range handed to `checked_span_bounds` is the true min/max over the
    # whole block; a too-small range would silently under-validate.
    Random.seed!(20260922)
    for trial in 1:150
        blocklen = rand(1:24)
        reg = rand(1:5)
        nsliv = cld(blocklen, reg)
        buf1 = [rand(-40:40) for _ in 1:blocklen]
        buf2 = [rand(-40:40) for _ in 1:blocklen]
        d1 = Vector{BlockDescriptor}(undef, nsliv)
        d2 = Vector{BlockDescriptor}(undef, nsliv)
        (r1, r2) = QuasiStrided.classify_slivers!(QuasiStrided.GroupBuffers((buf1, buf2), (d1, d2)), blocklen, reg, nsliv)
        @test r1 == (minimum(buf1), maximum(buf1))
        @test r2 == (minimum(buf2), maximum(buf2))
        @test all(s -> d1[s].count == min(reg, blocklen - (s - 1) * reg), 1:nsliv)
    end

    # Both extremes in an interior sliver.
    blocklen, reg = 18, 6                      # slivers 1:6, 7:12, 13:18
    buf1 = fill(1, blocklen); buf1[8] = 500; buf1[9] = -500   # sliver 2, irregular
    buf2 = fill(2, blocklen)
    buf2[7:12] .= [0, -7, -14, -21, -28, -35]                 # sliver 2, REGULAR, stride -7
    d1 = Vector{BlockDescriptor}(undef, 3)
    d2 = Vector{BlockDescriptor}(undef, 3)
    (r1, r2) = QuasiStrided.classify_slivers!(QuasiStrided.GroupBuffers((buf1, buf2), (d1, d2)), blocklen, reg, 3)
    @test !d1[2].regular && d1[2].count == 6        # scan branch
    @test d2[2].regular && d2[2].stride == -7       # affine branch, negative stride
    @test r1 == (-500, 500)
    @test r2 == (-35, 2)
    @test QuasiStrided.descriptor_offset_range(d1[1], buf1, 0) == (1, 1)
    @test QuasiStrided.descriptor_offset_range(d1[3], buf1, 12) == (1, 1)
end

# Counts `length` calls: the hoisted bounds check asks once per operand per
# block, a per-tile check once per tile. Not a `DenseVector`, so the fast paths
# stay off.
mutable struct CountingStorage{T} <: AbstractVector{T}
    data::Vector{T}
    n::Int
end
CountingStorage(v::Vector{T}) where {T} = CountingStorage{T}(v, 0)
Base.size(c::CountingStorage) = size(c.data)
Base.length(c::CountingStorage) = (c.n += 1; length(c.data))
Base.@propagate_inbounds Base.getindex(c::CountingStorage, i::Int) = c.data[i]
Base.@propagate_inbounds Base.setindex!(c::CountingStorage, v, i::Int) = (c.data[i] = v)
Base.IndexStyle(::Type{<:CountingStorage}) = IndexLinear()

@testset "the destination bounds check runs once per macro block" begin
    Ma, Ka, Na = 40, 7, 30
    kernel = SIMDKernel(Val(8), Val(6), Float64)
    Amat, Bmat = randn(Ma, Ka), randn(Ka, Na)
    Cmat = zeros(Ma, Na)
    base = plan_contract(
        StridedView(Cmat), StridedView(Amat), (1, 2), StridedView(Bmat), (2, 3),
        (1, 3); kernel = kernel
    )

    # One (N, K, M) block, so the hoisted checks run once.
    @test base.blocking.m_block >= Ma && base.blocking.n_block >= Na && base.blocking.k_block >= Ka
    m_tiles = cld(Ma, tile_size(kernel, 1))
    n_tiles = cld(Na, tile_size(kernel, 2))
    ntiles = m_tiles * n_tiles
    @test ntiles >= 20   # a per-tile check would be at least this many calls

    cstore = CountingStorage(zeros(Ma * Na))
    astore = CountingStorage(copy(vec(Amat)))
    bstore = CountingStorage(copy(vec(Bmat)))
    p = ContractPlan(base; Astorage = astore, Abase = 0, Bstorage = bstore, Bbase = 0, Cstorage = cstore, Cbase = 0)
    astore.n = 0; bstore.n = 0; cstore.n = 0
    execute!(p, 1.0, 0.0)
    @test reshape(cstore.data, Ma, Na) ≈ Amat * Bmat

    # `--check-bounds=yes` (as under `Pkg.test`) also runs the per-tile and
    # per-sliver checks that `@inbounds` skips.
    forced = Base.JLOptions().check_bounds == 1
    @test cstore.n == 1 + forced * ntiles
    @test astore.n == 1 + forced * m_tiles
    @test bstore.n == 1
end

@testset "a block that must be rejected is still rejected" begin
    Ma, Ka, Na = 40, 7, 30
    kernel = SIMDKernel(Val(8), Val(6), Float64)
    Amat, Bmat = randn(Ma, Ka), randn(Ka, Na)
    Cmat = zeros(Ma, Na)
    base = plan_contract(
        StridedView(Cmat), StridedView(Amat), (1, 2), StridedView(Bmat), (2, 3),
        (1, 3); kernel = kernel
    )

    function replan(;
            Cstorage = vec(Cmat), Astorage = vec(Amat), Bstorage = vec(Bmat),
            Cbase = 0, Abase = 0, Bbase = 0
        )
        return ContractPlan(base; Astorage, Abase, Bstorage, Bbase, Cstorage, Cbase)
    end

    # One element short: rejected before anything is written.
    short_C = zeros(Ma * Na - 1)
    @test_throws BoundsError execute!(replan(Cstorage = short_C), 1.0, 0.0)
    @test all(iszero, short_C)

    @test_throws BoundsError execute!(replan(Cbase = -1), 1.0, 0.0)

    @test_throws BoundsError execute!(replan(Astorage = zeros(Ma * Ka - 1)), 1.0, 0.0)
    @test_throws BoundsError execute!(replan(Bstorage = zeros(Ka * Na - 1)), 1.0, 0.0)
    @test_throws BoundsError execute!(replan(Abase = -1), 1.0, 0.0)
    @test_throws BoundsError execute!(replan(Bbase = -1), 1.0, 0.0)

    exact = zeros(Ma * Na)
    execute!(replan(Cstorage = exact), 1.0, 0.0)
    @test reshape(exact, Ma, Na) ≈ Amat * Bmat
end

@testset "rejection when the binding address is in an INTERIOR sliver" begin
    # C[m,n1,n2] = A[m,k] * B[k,n1,n2] with C reversed along n2: the N
    # offsets climb within each n1 run and fall at its boundary, so the block
    # maximum sits in an interior, irregular sliver (2 of 7).
    M, K, N1, N2 = 20, 4, 13, 3
    kernel = SIMDKernel(Val(8), Val(6), Float64)
    Amat = randn(M, K)
    Barr = randn(K, N1, N2)
    Cfull = zeros(M, N1, N2)
    Cr = view(Cfull, :, :, N2:-1:1)
    Av, Bv, Cv = StridedView(Amat), StridedView(Barr), StridedView(Cr)

    base = plan_contract(Cv, Av, (1, 2), Bv, (2, 3, 4), (1, 3, 4); kernel = kernel)
    @test base.Astorage === parent(Av)                   # no M/N swap: M's run is 20 >= 8
    @test !first(QuasiStrided.affine_ramp(base.ngroup))            # the buffer path, not the ramp path
    n_length = axis_length(base.ngroup)
    @test n_length == N1 * N2
    @test base.blocking.n_block >= n_length                         # one N block, so 7 slivers
    nsliv = cld(n_length, tile_size(kernel, 2))
    @test nsliv == 7

    noffs = [offsets(base.ngroup, q)[2] for q in 0:(n_length - 1)]
    binding = argmax(noffs) - 1                          # zero-based logical coordinate
    @test binding == N1 - 1
    @test 0 < binding ÷ tile_size(kernel, 2) < nsliv - 1           # a strictly interior sliver
    @test maximum(noffs[1:tile_size(kernel, 2)]) < noffs[binding + 1]                 # not the first
    @test maximum(noffs[(1 + (nsliv - 1) * tile_size(kernel, 2)):end]) < noffs[binding + 1]  # not the last

    execute!(base, 1.0, 0.0)
    ref = zeros(M, N1, N2)
    for m in 1:M, n1 in 1:N1, n2 in 1:N2
        ref[m, n1, n2] = sum(Amat[m, k] * Barr[k, n1, n2] for k in 1:K)
    end
    @test Cr ≈ ref

    # One element short: only the address at `binding` overflows.
    short_C = zeros(M * N1 * N2 - 1)
    pshort = ContractPlan(base; Cstorage = short_C)
    @test_throws BoundsError execute!(pshort, 1.0, 0.0)
    @test all(iszero, short_C)                            # nothing written before the throw

    # A first- or last-sliver-only range would accept it.
    moffs = [offsets(base.mgroup, q)[2] for q in 0:(axis_length(base.mgroup) - 1)]
    mrange = (minimum(moffs), maximum(moffs))
    NR = tile_size(kernel, 2)
    truerange = (minimum(noffs), maximum(noffs))
    firstonly = (minimum(noffs[1:NR]), maximum(noffs[1:NR]))
    lastonly = let tail = noffs[(1 + (nsliv - 1) * NR):end]
        (minimum(tail), maximum(tail))
    end
    shortlen = length(short_C)
    @test_throws BoundsError QuasiStrided.checked_span_bounds(base.Cbase, mrange, truerange, shortlen)
    @test QuasiStrided.checked_span_bounds(base.Cbase, mrange, firstonly, shortlen) === nothing
    @test QuasiStrided.checked_span_bounds(base.Cbase, mrange, lastonly, shortlen) === nothing

    # An interior-sliver minimum: the base shifted down by one.
    plow = ContractPlan(base; Cstorage = zeros(M * N1 * N2), Cbase = base.Cbase - 1)
    @test_throws BoundsError execute!(plow, 1.0, 0.0)
end

@testset "ramp_slivers! reproduces classify_slivers! exactly" begin
    Random.seed!(31337)
    for trial in 1:150
        D = rand(1:3)
        lens = ntuple(_ -> rand(1:5), D)
        # A ramp by construction half the time.
        strd = if rand() < 0.5
            s1 = rand(-5:5); s2 = rand(-5:5)
            acc = 1
            t1 = Int[]; t2 = Int[]
            for d in 1:D
                push!(t1, acc * s1); push!(t2, acc * s2)
                acc *= lens[d]
            end
            (Tuple(t1), Tuple(t2))
        else
            ntuple(_ -> ntuple(_ -> rand(-5:5), D), 2)
        end
        g = AxisGroup(lens, strd)
        Q = axis_length(g)
        Q == 0 && continue
        (isramp, steps) = QuasiStrided.affine_ramp(g)
        isramp || continue

        reg = rand(1:4)
        first = rand(0:(Q - 1))
        blocklen = rand(1:(Q - first))
        nsliv = cld(blocklen, reg)

        buf1 = zeros(Int, blocklen); buf2 = zeros(Int, blocklen)
        d1 = Vector{BlockDescriptor}(undef, nsliv)
        d2 = Vector{BlockDescriptor}(undef, nsliv)
        fill_offsets!((buf1, buf2), g, first, blocklen)
        want = QuasiStrided.classify_slivers!(QuasiStrided.GroupBuffers((buf1, buf2), (d1, d2)), blocklen, reg, nsliv)

        r1 = Vector{BlockDescriptor}(undef, nsliv)
        r2 = Vector{BlockDescriptor}(undef, nsliv)
        got = QuasiStrided.ramp_slivers!(QuasiStrided.GroupBuffers((buf1, buf2), (r1, r2)), steps[1], steps[2], first, blocklen, reg, nsliv)

        for s in 1:nsliv
            @test r1[s].base == d1[s].base && r1[s].stride == d1[s].stride
            @test r1[s].count == d1[s].count && r1[s].regular == d1[s].regular
            @test r2[s].base == d2[s].base && r2[s].stride == d2[s].stride
            @test r2[s].count == d2[s].count && r2[s].regular == d2[s].regular
        end
        @test got == want
    end
end

@testset "ramp and buffer paths give the same contraction" begin
    Random.seed!(424242)
    kernel = SIMDKernel(Val(8), Val(6), Float64)

    function check(Csz, Asz, Bsz, indA, indB, indC; kw...)
        A = randn(Asz); B = randn(Bsz); C = randn(Csz)
        C0 = copy(C)
        p = plan_contract(
            StridedView(C), StridedView(A), indA, StridedView(B), indB, indC; kw...
        )
        execute!(p, 2.0, -0.5)
        @test C ≈ _lo_reference(C0, A, indA, B, indB, indC; alpha = 2.0, beta = -0.5)
        return p
    end

    p = check((40, 30), (40, 7), (7, 30), (1, 2), (2, 3), (1, 3); kernel = kernel)
    @test first(QuasiStrided.affine_ramp(p.mgroup)) && first(QuasiStrided.affine_ramp(p.ngroup)) &&
        first(QuasiStrided.affine_ramp(p.kgroup))

    # ao2mo_2: a ramp on M and K but not on N.
    d = 6
    p = check(
        (d, d, d, d), (d, d), (d, d, d, d),
        (1, 2), (3, 1, 4, 5), (3, 2, 4, 5); kernel = kernel
    )
    @test first(QuasiStrided.affine_ramp(p.mgroup))
    @test first(QuasiStrided.affine_ramp(p.kgroup))
    @test !first(QuasiStrided.affine_ramp(p.ngroup))   # the mixed path, both branches live

    # Several macro blocks, so ramp descriptors start at a nonzero `first`.
    check(
        (40, 30), (40, 21), (21, 30), (1, 2), (2, 3), (1, 3);
        kernel = kernel, m_block = 8, k_block = 5, n_block = 6
    )
    # Non-ramp composites on both sides (rank-4 operands, non-folding order).
    d = 5
    check(
        (d, d, d, d, d, d), (d, d, d, d), (d, d, d, d),
        (1, 2, 3, 4), (3, 5, 6, 7), (4, 6, 7, 1, 2, 5); kernel = kernel
    )
end
