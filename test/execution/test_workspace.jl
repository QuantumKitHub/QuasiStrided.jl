# ContractWorkspace, the `allocator` keyword, and the SIMDKernel default.

@testset "plan_contract: SIMDKernel is the engine-wide default kernel" begin
    for T in (Float64, Float32)
        Random.seed!(5150)
        Amat, Bmat = randn(T, 9, 10), randn(T, 10, 8)
        Cmat = zeros(T, 9, 8)
        plan = _mm_plan(Cmat, Amat, Bmat)

        # The shape is hardware- and extent-dependent: pin the resolution.
        @test plan.kernel isa QuasiStrided.SIMDKernel
        @test QuasiStrided.scalartype(plan.kernel) === T
        @test plan.kernel === auto_kernel(T, size(Amat, 1))
        execute!(plan, one(T), zero(T))
        @test Cmat ≈ Amat * Bmat

        Cmat2 = zeros(T, 9, 8)
        contract!(
            StridedView(Cmat2), one(T), StridedView(Amat), (1, 2),
            StridedView(Bmat), (2, 3), zero(T), (1, 3)
        )
        @test Cmat2 == Cmat
    end
end

@testset "ContractWorkspace: plan reuse refills every buffer before reading it" begin
    kernel = ScalarKernel(Val(4), Val(3), Float64)
    alpha, beta = 2.5, -0.75

    Random.seed!(606)
    Amat, Bmat = randn(23, 19), randn(19, 17)
    Cstart = randn(23, 17)
    Cref = copy(Cstart)
    execute!(_mm_plan(Cref, Amat, Bmat; kernel = kernel, m_block = 8, k_block = 6, n_block = 7), alpha, beta)

    C = copy(Cstart)
    plan = _mm_plan(C, Amat, Bmat; kernel = kernel, m_block = 8, k_block = 6, n_block = 7)
    execute!(plan, alpha, beta)
    copyto!(C, Cstart)

    # A stale packed slot yields NaN, a stale offset or descriptor addresses far
    # outside the operand.
    ws = plan.workspace
    fill!(ws.packed_a, NaN)
    fill!(ws.packed_b, NaN)
    for buf in (ws.m.offsets..., ws.n.offsets..., ws.k...)
        fill!(buf, typemin(Int) ÷ 4)
    end
    poison = BlockDescriptor(typemin(Int) ÷ 4, 0, 1, true)
    foreach(d -> fill!(d, poison), (ws.m.descriptors..., ws.n.descriptors...))

    execute!(plan, alpha, beta)
    @test C == Cref

    # The beta-only pass borrows the M/N offset buffers.
    foreach(buf -> fill!(buf, typemin(Int) ÷ 4), (ws.m.offsets..., ws.n.offsets...))
    execute!(plan, 0.0, 0.5)
    @test C == 0.5 .* Cref
end

@testset "plan_contract: explicit allocators serve the packed panels" begin
    Random.seed!(1717)
    kernel = SIMDKernel(Val(4), Val(3), Float64)
    Ma, Ka, Na = 19, 23, 17
    m_block, k_block, n_block = 8, 6, 7
    Amat, Bmat = randn(Ma, Ka), randn(Ka, Na)

    Cdefault = zeros(Ma, Na)
    default_plan = _mm_plan(Cdefault, Amat, Bmat; kernel = kernel, m_block = m_block, k_block = k_block, n_block = n_block)
    execute!(default_plan, 1.5, 0.0)
    @test default_plan.workspace.packed_a isa Vector{Float64}

    for allocator in (TO.ManualAllocator(), TO.BufferAllocator())
        checkpoint = TO.allocator_checkpoint!(allocator)

        Cmat = zeros(Ma, Na)
        plan = _mm_plan(
            Cmat, Amat, Bmat;
            kernel = kernel, m_block = m_block, k_block = k_block, n_block = n_block,
            allocator = allocator
        )
        ws = plan.workspace

        @test isconcretetype(typeof(ws))
        @test all(isconcretetype, fieldtypes(typeof(ws)))
        @test length(ws.packed_a) == cld(plan.blocking.m_block, 4) * 4 * plan.blocking.k_block
        @test length(ws.packed_b) == cld(plan.blocking.n_block, 3) * 3 * plan.blocking.k_block
        @test ws.m.offsets[1] isa Vector{Int}
        @test ws.k[2] isa Vector{Int}

        execute!(plan, 1.5, 0.0)
        @test Cmat == Cdefault

        QuasiStrided.release!(ws, allocator)
        TO.allocator_reset!(allocator, checkpoint)
    end
end

@testset "ContractWorkspace: storage type T, packed panels of real(T)" begin
    k64 = SIMDKernel(Val(8), Val(6), Float64)
    k32 = SIMDKernel(Val(8), Val(6), Float32)
    b = Blocking(16, 8, 12)

    ws = QuasiStrided.ContractWorkspace(Float64, k64, b)
    @test ws isa QuasiStrided.ContractWorkspace{Float64, Vector{Float64}}
    @test eltype(ws.packed_a) === Float64

    wsc = QuasiStrided.ContractWorkspace(ComplexF64, k64, b)
    @test wsc isa QuasiStrided.ContractWorkspace{ComplexF64, Vector{Float64}}
    @test eltype(wsc.packed_a) === Float64 === eltype(wsc.packed_b)
    @test all(isconcretetype, fieldtypes(typeof(wsc)))
    @test !any(t -> t isa Union, fieldtypes(typeof(wsc)))

    @test_throws ArgumentError QuasiStrided.ContractWorkspace(ComplexF64, k32, b)
    @test_throws ArgumentError QuasiStrided.ContractWorkspace(Float64, k32, b)
    @test_throws ArgumentError QuasiStrided.ContractWorkspace(ComplexF32, k64, b)
end
