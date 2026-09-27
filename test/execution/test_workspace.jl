# =====================================================================
# ContractWorkspace, the `workspace`/`allocator`/`oracle` keywords, and the
# SIMDKernel default.
# =====================================================================

# Buffer fields only: the workspace also holds its dispatch-barrier slot cache,
# which is not a buffer and has no length.
_ws_lengths(ws) = [
    length(getfield(ws, f)) for f in fieldnames(typeof(ws)) if getfield(ws, f) isa AbstractVector
]

@testset "plan_contract: SIMDKernel is the engine-wide default kernel" begin
    for T in (Float64, Float32)
        Random.seed!(5150)
        Amat, Bmat = randn(T, 9, 10), randn(T, 10, 8)
        Cmat = zeros(T, 9, 8)
        plan = _mm_plan(Cmat, Amat, Bmat)

        # A SIMDKernel, not a ScalarKernel. The shape itself is
        # hardware-derived and demoted when M cannot fill a register tile, so
        # pin the *resolution* rather than a literal shape: a literal would
        # hold only on hosts where Qm = 9 happens to trigger the same
        # demotion.
        @test plan.kernel isa QuasiStrided.SIMDKernel
        @test QuasiStrided.scalartype(plan.kernel) === T
        @test plan.kernel === QuasiStrided._default_kernel(T, size(Amat, 1), size(Bmat, 2))
        execute!(plan, one(T), zero(T))
        @test Cmat ≈ Amat * Bmat

        # ... and `contract!`, which must never disagree with plan_contract
        # about what "default" means.
        Cmat2 = zeros(T, 9, 8)
        contract!(
            StridedView(Cmat2), one(T), StridedView(Amat), (1, 2),
            StridedView(Bmat), (2, 3), zero(T), (1, 3)
        )
        @test Cmat2 == Cmat
    end
end

@testset "ContractWorkspace: reuse across shapes is bitwise identical to fresh plans" begin
    kernel = ScalarKernel(Val(4), Val(3), Float64)
    mc, kc, nc = 8, 6, 7
    alpha, beta = 1.75, -0.5

    # Deliberately not monotone in size: the workspace is sized by the first
    # (mid) shape, grown by the second (large) one, then reused oversized by
    # every smaller one after it.
    shapes = ((13, 11, 10), (23, 19, 17), (4, 3, 2), (9, 10, 8), (1, 1, 1), (16, 5, 6))

    ws = nothing
    lengths_before = nothing
    for (idx, (Ma, Ka, Na)) in enumerate(shapes)
        Random.seed!(31_000 + idx)
        Amat, Bmat = randn(Ma, Ka), randn(Ka, Na)
        Cstart = randn(Ma, Na)

        Cfresh = copy(Cstart)
        execute!(_mm_plan(Cfresh, Amat, Bmat; kernel = kernel, mc = mc, kc = kc, nc = nc), alpha, beta)

        Creuse = copy(Cstart)
        plan = _mm_plan(
            Creuse, Amat, Bmat;
            kernel = kernel, mc = mc, kc = kc, nc = nc, workspace = ws
        )
        execute!(plan, alpha, beta)

        # Bitwise, not approximate: reuse must not perturb the arithmetic.
        @test Creuse == Cfresh

        if ws !== nothing
            @test plan.workspace === ws               # reserve!d in place, not rebuilt
            @test all(_ws_lengths(ws) .>= lengths_before)  # grow-only, never shrunk
        end
        ws = plan.workspace
        lengths_before = _ws_lengths(ws)
    end

    # Wrong-eltype workspaces are rejected rather than silently rebuilt.
    Amat32, Bmat32, Cmat32 = randn(Float32, 4, 4), randn(Float32, 4, 4), zeros(Float32, 4, 4)
    @test_throws ArgumentError _mm_plan(
        Cmat32, Amat32, Bmat32;
        kernel = ScalarKernel(Val(4), Val(3), Float32), workspace = ws
    )
end

@testset "ContractWorkspace: an oversized reused buffer is not read beyond its live region" begin
    kernel = ScalarKernel(Val(4), Val(3), Float64)
    mc, kc, nc = 8, 6, 7
    alpha, beta = 2.5, -0.75

    # Size the workspace on a large contraction ...
    Random.seed!(606)
    Abig, Bbig = randn(23, 19), randn(19, 17)
    Cbig = zeros(23, 17)
    big = _mm_plan(Cbig, Abig, Bbig; kernel = kernel, mc = mc, kc = kc, nc = nc)
    execute!(big, 1.0, 0.0)
    ws = big.workspace

    # ... then run a much smaller one on it.
    Ma, Ka, Na = 5, 4, 3
    Amat, Bmat = randn(Ma, Ka), randn(Ka, Na)
    Cstart = randn(Ma, Na)

    Cref = copy(Cstart)
    execute!(_mm_plan(Cref, Amat, Bmat; kernel = kernel, mc = mc, kc = kc, nc = nc), alpha, beta)

    # Poison every slot of every reused buffer. A packed slot read without
    # having been written this call turns the output into NaN; an offset or
    # descriptor read outside the live region addresses far outside the
    # operand and is rejected by checked_tile_storage_bounds.
    fill!(ws.packed_a, NaN)
    fill!(ws.packed_b, NaN)
    for buf in (ws.m_buf_A, ws.m_buf_C, ws.n_buf_B, ws.n_buf_C, ws.k_buf_A, ws.k_buf_B)
        fill!(buf, typemin(Int) ÷ 4)
    end
    poison = BlockDescriptor(typemin(Int) ÷ 4, 0, 1, true)
    for desc in (ws.m_desc_A, ws.m_desc_C, ws.n_desc_B, ws.n_desc_C)
        fill!(desc, poison)
    end

    lengths_before = _ws_lengths(ws)
    Cpoisoned = copy(Cstart)
    plan = _mm_plan(
        Cpoisoned, Amat, Bmat;
        kernel = kernel, mc = mc, kc = kc, nc = nc, workspace = ws
    )
    # Nothing was regrown, so this really did run on the oversized buffers.
    @test _ws_lengths(ws) == lengths_before
    @test length(ws.packed_a) > cld(Ma, 4) * 4 * min(kc, Ka)
    @test length(ws.packed_b) > cld(Na, 3) * 3 * min(kc, Ka)

    execute!(plan, alpha, beta)
    @test all(isfinite, Cpoisoned)
    @test Cpoisoned == Cref
end

@testset "plan_contract: oracle = false skips execute_tilewise!'s buffers" begin
    Random.seed!(909)
    kernel = ScalarKernel(Val(4), Val(3), Float64)
    Ma, Ka, Na = 9, 10, 8
    Amat, Bmat = randn(Ma, Ka), randn(Ka, Na)

    Cmat = zeros(Ma, Na)
    plan = _mm_plan(Cmat, Amat, Bmat; kernel = kernel, kc = 4, oracle = false)
    ws = plan.workspace

    @test isempty(ws.tw_packed_a)
    @test isempty(ws.tw_packed_b)
    @test isempty(ws.tw_k_buf_A)
    @test isempty(ws.tw_k_buf_B)
    # The MR/NR-sized ones are NOT oracle-only: _scale_all_of_C!, the beta-only
    # pass of both drivers, uses them.
    @test length(ws.tile_m_buf_A) == 4
    @test length(ws.tile_n_buf_C) == 3

    execute!(plan, 1.0, 0.0)
    @test Cmat ≈ Amat * Bmat

    # The beta-only short-circuit still works without the oracle buffers.
    Cstart = randn(Ma, Na)
    Cbeta = copy(Cstart)
    execute!(_mm_plan(Cbeta, Amat, Bmat; kernel = kernel, oracle = false), 0.0, 0.5)
    @test Cbeta ≈ 0.5 .* Cstart

    # ... but the oracle itself refuses to run rather than reading empty buffers.
    @test_throws ArgumentError execute_tilewise!(plan, 1.0, 0.0)

    # Reusing the same workspace with oracle = true grows them back.
    Ctw = zeros(Ma, Na)
    plan_tw = _mm_plan(Ctw, Amat, Bmat; kernel = kernel, kc = 4, workspace = ws, oracle = true)
    @test plan_tw.workspace === ws
    @test !isempty(ws.tw_packed_a)
    execute_tilewise!(plan_tw, 1.0, 0.0)
    @test Ctw ≈ Amat * Bmat
end

@testset "plan_contract: explicit allocators size the packed panels exactly once" begin
    Random.seed!(1717)
    kernel = SIMDKernel(Val(4), Val(3), Float64)
    Ma, Ka, Na = 19, 23, 17
    mc, kc, nc = 8, 6, 7
    Amat, Bmat = randn(Ma, Ka), randn(Ka, Na)

    Cdefault = zeros(Ma, Na)
    default_plan = _mm_plan(Cdefault, Amat, Bmat; kernel = kernel, mc = mc, kc = kc, nc = nc)
    execute!(default_plan, 1.5, 0.0)
    @test default_plan.workspace.packed_a isa Vector{Float64}

    for allocator in (TO.ManualAllocator(), TO.BufferAllocator())
        checkpoint = TO.allocator_checkpoint!(allocator)

        Cmat = zeros(Ma, Na)
        plan = _mm_plan(
            Cmat, Amat, Bmat;
            kernel = kernel, mc = mc, kc = kc, nc = nc,
            allocator = allocator, oracle = false
        )
        ws = plan.workspace

        # Concretely typed instance, exact sizing, no oracle buffers.
        @test isconcretetype(typeof(ws))
        @test all(isconcretetype, fieldtypes(typeof(ws)))
        @test length(ws.packed_a) == cld(plan.blocking.mc, 4) * 4 * plan.blocking.kc
        @test length(ws.packed_b) == cld(plan.blocking.nc, 3) * 3 * plan.blocking.kc
        @test isempty(ws.tw_packed_a)
        # Offset buffers are Val(false) requests: a plain Vector{Int} from
        # every allocator, because fill_offsets!/describe_block are fixed on
        # that concrete type.
        @test ws.m_buf_A isa Vector{Int}
        @test ws.k_buf_B isa Vector{Int}

        execute!(plan, 1.5, 0.0)
        @test Cmat == Cdefault  # same blocking, so bitwise identical

        QuasiStrided.release!(ws, allocator)
        TO.allocator_reset!(allocator, checkpoint)
    end

    # A PtrArray-backed workspace is not reserve!-able at all: the grow-upward
    # discipline and allocator-owned temporaries are mutually exclusive.
    manual = TO.ManualAllocator()
    Cmanual = zeros(Ma, Na)
    manual_plan = _mm_plan(
        Cmanual, Amat, Bmat;
        kernel = kernel, mc = mc, kc = kc, nc = nc, allocator = manual, oracle = false
    )
    @test_throws MethodError QuasiStrided.reserve!(
        manual_plan.workspace, kernel, manual_plan.blocking, false
    )
    QuasiStrided.release!(manual_plan.workspace, manual)

    # An explicit allocator and a reusable workspace are contradictory.
    @test_throws ArgumentError _mm_plan(
        zeros(Ma, Na), Amat, Bmat;
        kernel = kernel, allocator = TO.ManualAllocator(),
        workspace = default_plan.workspace
    )
end

@testset "ContractWorkspace: the relaxed VT bound keeps every old spelling" begin
    k64 = SIMDKernel(Val(8), Val(6), Float64)
    k32 = SIMDKernel(Val(8), Val(6), Float32)
    b = Blocking(16, 8, 12)

    # Real storage: the plain `ContractWorkspace{T, Vector{T}}` form.
    ws = QuasiStrided.ContractWorkspace(Float64, k64, b; oracle = true)
    @test ws isa QuasiStrided.ContractWorkspace{Float64, Vector{Float64}}
    @test eltype(ws.packed_a) === Float64

    # `T` is the STORAGE element type and the packed panels hold `real(T)`: a
    # complex storage type over Float64 packing is admitted at the SAME
    # arity.
    wsc = QuasiStrided.ContractWorkspace(ComplexF64, k64, b; oracle = true)
    @test wsc isa QuasiStrided.ContractWorkspace{ComplexF64, Vector{Float64}}
    @test eltype(wsc.packed_a) === Float64 === eltype(wsc.tw_packed_b)
    @test all(isconcretetype, fieldtypes(typeof(wsc)))
    @test !any(t -> t isa Union, fieldtypes(typeof(wsc)))

    # ... and a mismatched (T, VT) pair cannot be constructed at all: the inner
    # constructor enforces eltype(VT) === real(T).
    @test_throws ArgumentError QuasiStrided.ContractWorkspace(ComplexF64, k32, b)
    @test_throws ArgumentError QuasiStrided.ContractWorkspace(Float64, k32, b)
    @test_throws ArgumentError QuasiStrided.ContractWorkspace(ComplexF32, k64, b)

    # Reuse still refuses a workspace of the wrong storage type, by dispatch.
    Amat, Bmat, Cmat = randn(6, 5), randn(5, 4), zeros(6, 4)
    @test_throws ArgumentError plan_contract(
        StridedView(Cmat), StridedView(Amat), (1, 2), StridedView(Bmat), (2, 3), (1, 3);
        workspace = wsc
    )
end
