# Register shape, kernel and blocking selection from a `TargetProfile`.

include("helpers.jl")

using StridedViews: StridedView

@testset "real shape selection" begin
    @testset "unknown target: fallback shape and blocking" begin
        u = unknown_target()
        for T in (Float64, Float32)
            @test derived_shape(u, T) === fallback_shape(T)
            k = kernel_from_shape(derived_shape(u, T), T)
            @test k isa SIMDKernel
            @test (tile_size(k)..., lanewidth(k)) === fallback_shape(T)
            @test default_blocking(k, u) === fallback_blocking(T)
        end
        @test fallback_blocking(Float64) === Blocking(128, 256, 768)
        @test fallback_blocking(Float32) === Blocking(96, 768, 1152)
    end

    @testset "rule: MR = MV*W, NR = 6; MV = 4 on Intel :avx512, 2 on :avx2 and AMD" begin
        for (isakey, vb, MV) in ((:avx512, 64, 4), (:avx2, 32, 2)), T in (Float64, Float32)
            W = vb ÷ sizeof(T)
            @test derived_shape(synthetic(isakey), T) === (MV * W, NR_DEFAULT, W)
            @test QuasiStrided.rule_mv(isakey, SIMDKernel) == MV
            @test QuasiStrided.rule_shape(vb, T, MV) === (MV * W, NR_DEFAULT, W)
        end
        for K in (PlanarKernel, OneMKernel, FMAddSubKernel), key in (:avx512, :avx2)
            @test QuasiStrided.rule_mv(key, K) == 2
        end
        for (cpu, MV) in (("znver4", 2), ("znver5", 2), ("icelake-server", 4), ("cascadelake", 4)),
                T in (Float64, Float32)
            p = TargetProfile(:avx512, cpu, CacheLevel(), CacheLevel(), CacheLevel())
            W = 64 ÷ sizeof(T)
            @test derived_shape(p, T) === (MV * W, NR_DEFAULT, W)
            @test derived_shape(p, ComplexF64) === derived_shape(synthetic(:avx512), ComplexF64)
        end
        @test derived_shape(synthetic(:avx2), Float64) === fallback_shape(Float64)
        # ISAs without a rule get the fallback shape, whatever their width.
        for T in (Float64, Float32)
            @test derived_shape(synthetic(:neon), T) === fallback_shape(T)
            @test derived_shape(synthetic(:unknown), T) === fallback_shape(T)
        end
        # No real override row on any ISA: the rule is the optimum.
        for T in (Float64, Float32), key in VALID_ISAS
            @test shape_override(key, T) === nothing
        end
        @test rule_applies(:avx2, SIMDKernel) && rule_applies(:avx512, SIMDKernel)
    end

    @testset "every real menu shape is constructible and fits the register file" begin
        for T in (Float64, Float32), (MR, NR, W) in kernel_shapes(T)
            @test MR % W == 0
            k = SIMDKernel(Val(MR), Val(NR), T, Val(W))
            @test (tile_size(k)..., lanewidth(k)) == (MR, NR, W)
            @test (MR ÷ W) * NR + (MR ÷ W) <= 32
            @test sliver_width(k) === (MR, NR)
            @test realtype(k) === T === scalartype(k)
            @test packed_a_length(k, 7) === MR * 7 && packed_b_length(k, 7) === NR * 7
        end
        for T in (Float64, Float32)
            W = 64 ÷ sizeof(T)
            # The MV = 2 sibling stays in the menu as the step-down target.
            @test (2 * W, NR_DEFAULT, W) in kernel_shapes(T)
            @test kernel_shapes(T, SIMDKernel) === kernel_shapes(T)
            k = host_kernel(T)
            @test k isa SIMDKernel && scalartype(k) === T
            @test (tile_size(k)..., lanewidth(k)) in kernel_shapes(T)
        end
    end

    @testset "extent demotion when M cannot fill a register tile" begin
        for T in (Float64, Float32)
            MR = tile_size(host_kernel(T), 1)
            @test tile_size(auto_kernel(T, 4 * MR), 1) == MR
            small = auto_kernel(T, 1)
            @test (tile_size(small)..., lanewidth(small)) === fallback_shape(T)
            @test tile_size(auto_kernel(T, 0), 1) == MR        # empty must not demote
        end
    end

    @testset "extent_shape: MV = 4 steps down to MV = 2 where it pads less" begin
        ext(shape, T, m_length) = QuasiStrided.extent_shape(shape, T, SIMDKernel, m_length)
        for T in (Float64, Float32)
            tall = derived_shape(synthetic(:avx512), T)
            MR, NR, W = tall
            half = (2 * W, NR, W)
            @test ext(tall, T, 0) === tall
            for m_length in (1, 2 * W - 1, 2 * W, MR - 1, MR + 1, MR + W, MR + 2 * W)
                @test ext(tall, T, m_length) === half
            end
            for m_length in (MR, MR + 2 * W + 1, 2 * MR - 1, 2 * MR, 2 * MR + 1, 3 * MR + W, 100 * MR)
                @test ext(tall, T, m_length) === tall
            end
            # Never below MV = 2, and complex kernels are untouched.
            avx2 = derived_shape(synthetic(:avx2), T)
            @test ext(avx2, T, 3) === avx2
            cs = derived_shape(synthetic(:avx512), ComplexF64)
            @test QuasiStrided.extent_shape(cs, ComplexF64, PlanarKernel, 3) === cs
            # Through `auto_kernel`, where the host's shape is the tall one.
            if derived_shape(target_profile(), T) === tall
                @test tile_size(auto_kernel(T, MR + W)) == (2 * W, NR)
                @test tile_size(auto_kernel(T, 2 * W), 1) == 2 * W
                @test tile_size(auto_kernel(T, 2 * W - 1), 1) == fallback_shape(T)[1]
                @test tile_size(auto_kernel(T, 2 * MR), 1) == MR
            end
        end
    end

    @testset "fit_to_run: C's run along M decides the real shape" begin
        fit(shape, T, m_length, k_length, run) =
            QuasiStrided.fit_to_run(shape, T, SIMDKernel, m_length, k_length, run)
        largest_divisor(T, run) = argmax(s -> (run % s[1] == 0, s[1]), kernel_shapes(T))
        deep = 10^6
        for T in (Float64, Float32)
            tall = derived_shape(synthetic(:avx512), T)
            MR, NR, W = tall
            half = (2 * W, NR, W)
            m_length = 64 * MR
            # MV = 4 -> MV = 2 at any K.
            for run in (m_length, MR, 2 * MR, 5 * MR)                  # every tall sliver whole
                @test fit(tall, T, m_length, deep, run) === tall
            end
            @test fit(tall, T, 3 * W, deep, 3 * W) === tall
            for run in (2 * W, 3 * W, 6 * W, MR + 2 * W, m_length - 1)
                @test fit(tall, T, m_length, deep, run) === half
            end
            for run in (1, W, 2 * W - 1)                          # below one half tile
                @test fit(tall, T, m_length, deep, run) === tall
            end
            @test fit(half, T, m_length, deep, 3 * W) === half
            # Then, at shallow K, the largest menu shape whose `MR` divides the run.
            @test fit(tall, T, m_length, 1, 3 * W) === largest_divisor(T, 3 * W)
            kmax = T === Float64 ? QuasiStrided.RUN_DEMOTE_KMAX_F64 : QuasiStrided.RUN_DEMOTE_KMAX_F32
            for shape in kernel_shapes(T)
                S = shape[1]
                @test fit(shape, T, 1000, 1, 3 * S) === shape
                @test fit(shape, T, 7, 1, 7) === shape
                @test fit(shape, T, 1000, kmax + 1, 4) === shape
                for run in (1, 4, 8, 12, 16, 20, 24, 48)
                    (run % S == 0 || (S == 4 * shape[3] && run >= 2 * shape[3])) && continue
                    want = any(s -> run % s[1] == 0, kernel_shapes(T)) ? largest_divisor(T, run) : shape
                    @test fit(shape, T, 1000, kmax, run) === want
                end
            end
            cs = derived_shape(synthetic(:avx512), ComplexF64)
            @test QuasiStrided.fit_to_run(cs, ComplexF64, PlanarKernel, m_length, 1, 1) === cs
            if derived_shape(target_profile(), T) === tall
                select(run) = QuasiStrided.select_shape(T, SIMDKernel, m_length, deep, run)[1]
                @test select(2 * W) === half
                @test select(MR) === tall
                @test select(m_length) === tall
            end
        end
    end

    @testset "fit_to_run feeds the swap: ccsd_t_2 / ao2mo_2 at dim 16" begin
        # Regression: C's only unit run is 16 long, on the N side. Judged
        # against m_tile = 32 the swap was declined and every tile stored scattered.
        T = Float64
        if derived_shape(target_profile(), T)[1] == 32
            d = 16
            a, b, c, i, j, k, m = 1, 2, 3, 4, 5, 6, -1
            Bv = StridedView(randn(T, d, d, d, d))
            plan = QuasiStrided.plan_contract(
                StridedView(zeros(T, d, d, d, d, d, d)), StridedView(randn(T, d, d, d, d)),
                (i, j, m, b), Bv, (m, k, a, c), (a, b, c, i, j, k)
            )
            @test tile_size(plan.kernel, 1) == 16
            @test plan.Astorage === parent(Bv)
            q, r, s = -1, 3, 4
            B2 = StridedView(randn(T, d, d, d, d))
            plan2 = QuasiStrided.plan_contract(
                StridedView(zeros(T, d, d, d, d)), StridedView(randn(T, d, d)), (q, b),
                B2, (a, q, r, s), (a, b, r, s)
            )
            @test tile_size(plan2.kernel, 1) == 16
            @test plan2.Astorage === parent(B2)
            @test axis_length(plan2.mgroup) == d^3
        end
    end
end

@testset "complex shape selection" begin
    for (T, W, swept) in ((ComplexF64, 8, (24, 3, 8)), (ComplexF32, 16, (48, 3, 16)))
        # W counts REAL lanes; the AVX-512 override replaces the spilling rule shape.
        @test QuasiStrided.rule_shape(64, T, 2) === (2 * W, NR_DEFAULT, W)
        @test shape_override(:avx512, T) === swept
        @test derived_shape(synthetic(:avx512), T) === swept === first(kernel_shapes(T, PlanarKernel))
        @test Set(kernel_shapes(T, PlanarKernel)) == Set(
            (
                (2 * W, NR_DEFAULT, W), swept, (W, 8, W), shape_override(:avx2, T),
                (W ÷ 2, NR_DEFAULT, W ÷ 4), (W ÷ 4, NR_DEFAULT, W ÷ 4),
            )
        )
        # Every override row fits its ISA's register file with room to spare.
        for (key, nreg) in ((:avx512, 32), (:avx2, 16))
            ovr = shape_override(key, T)
            @test planar_pressure(ovr...) < nreg
            @test ovr in kernel_shapes(T, PlanarKernel)
        end
        @test fallback_shape(T) === (8, NR_DEFAULT, fallback_shape(real(T))[3])
        # Off :avx512 an override row wins whatever the width; without a row
        # the shape is fitted to the register budget.
        @test derived_shape(synthetic(:avx2), T) === shape_override(:avx2, T)
        for key in (:avx2, :neon, :unknown)
            @test !rule_applies(key, PlanarKernel)
        end
        for key in (:neon, :unknown)
            @test shape_override(key, T) === nothing
        end
        @test planar_pressure(derived_shape(synthetic(:unknown), T)...) <= 16
    end
    @test rule_applies(:avx512, PlanarKernel)
    @test derived_shape(synthetic(:neon), ComplexF64) === (4, 6, 2)
    @test derived_shape(synthetic(:neon), ComplexF32) === (8, 6, 4)
    @test planar_pressure(fallback_shape(ComplexF64)...) == 30
    @test planar_pressure(fallback_shape(ComplexF32)...) == 16

    # Complex menus fit AVX-512 and stay bounded.
    nreg = isa_nregisters(:avx512)
    target_profile().isa === :avx512 && @test target_profile().nregisters == nreg
    for T in (ComplexF64, ComplexF32), K in (PlanarKernel, OneMKernel)
        planes = accumulator_planes(K)
        for (MR, NR, W) in kernel_shapes(T, K)
            rows = (2 * MR) ÷ planes
            @test rows % W == 0
            mv = rows ÷ W
            @test planes * mv * NR + planes * mv + planes <= nreg
        end
    end
    for T in (ComplexF64, ComplexF32)
        @test length(kernel_shapes(T, PlanarKernel)) <= 6
        @test length(kernel_shapes(T, OneMKernel)) <= 4
    end
end

@testset "AVX2 complex: fmaddsub where C's rows take its vector store, else planar" begin
    saved = QuasiStrided.TARGET[]
    try
        QuasiStrided.TARGET[] = synthetic(:avx2)
        for T in (ComplexF64, ComplexF32)
            W = 32 ÷ sizeof(real(T))
            fms, planar = ((W, NR_DEFAULT, W), Val(FMAddSubKernel)), ((W, 5, W), Val(PlanarKernel))
            select(m_length, run) = QuasiStrided.select_shape(T, PlanarKernel, m_length, 64, run)
            @test select(64, 64) === select(64, 2W) === select(1, 1) === fms
            @test select(64, W + 2) === select(64, 1) === planar
        end
    finally
        QuasiStrided.TARGET[] = saved
    end
end

@testset "mixed-domain selection: the real default of real(T), mapped" begin
    CR, RC = QuasiStrided.ComplexRealKernel, QuasiStrided.RealComplexKernel
    mapped(m, (MR, NR, W)) = m === CR ? (MR ÷ 2, NR, W) : (MR, NR ÷ 2, W)
    dkernel = QuasiStrided.default_kernel_type
    for T in (ComplexF64, ComplexF32)
        R = real(T)
        @test all(((MR, NR, W),) -> iseven(NR) && iseven(W), kernel_shapes(R))
        for m in (CR, RC)
            @test kernel_shapes(T, m) === map(s -> mapped(m, s), kernel_shapes(R))
            for shape in kernel_shapes(T, m)
                k = QuasiStrided.kernel_from_shape(shape, T, m)
                @test k isa m && (tile_size(k)..., lanewidth(k)) === shape
            end
        end
        @test dkernel(T, T, R) === dkernel(T, ComplexF32, Float64) === CR
        @test dkernel(T, R, T) === dkernel(T, Float32, ComplexF64) === RC
        @test dkernel(T, R, R) === dkernel(T, T, T) === dkernel(T, ComplexF32, T) === PlanarKernel
        @test dkernel(R, Float32, Float64) === SIMDKernel
    end
    # Every real step carries over.
    saved = QuasiStrided.TARGET[]
    try
        for key in (:avx512, :avx2, :neon)
            QuasiStrided.TARGET[] = synthetic(key)
            for T in (ComplexF64, ComplexF32), m in (CR, RC), m_length in (1, 3, 17, 40, 4096),
                    run in (0, 1, 8, m_length), k_length in (1, 1000)
                real_m, real_run = m === CR ? (2 * m_length, 2 * run) : (m_length, run)
                real_shape = QuasiStrided.select_shape(real(T), SIMDKernel, real_m, k_length, real_run)[1]
                @test QuasiStrided.select_shape(T, m, m_length, k_length, run) === (mapped(m, real_shape), Val(m))
            end
        end
    finally
        QuasiStrided.TARGET[] = saved
    end
end

@testset "derived_shape is always a constructible member of the kernel's menu" begin
    pairs = (
        (Float64, SIMDKernel), (Float32, SIMDKernel), (ComplexF64, PlanarKernel),
        (ComplexF32, PlanarKernel), (ComplexF64, OneMKernel), (ComplexF32, OneMKernel),
    )
    for (T, m) in pairs, isakey in VALID_ISAS
        shape = derived_shape(synthetic(isakey), T, m)
        @test shape in kernel_shapes(T, m)
        k = QuasiStrided.kernel_from_shape(shape, T, m)
        @test (tile_size(k)..., lanewidth(k)) === shape
    end
end

@testset "pack_formats agrees with every kernel's descriptor" begin
    QS = QuasiStrided
    kernels = (
        ScalarKernel(Val(4), Val(2), Float64), SIMDKernel(Val(8), Val(4), Float32),
        QS.PlanarKernel(Val(4), Val(2), ComplexF64), QS.OneMKernel(Val(4), Val(2), ComplexF64),
        QS.FMAddSubKernel(Val(4), Val(2), ComplexF64), QS.ComplexRealKernel(Val(4), Val(2), ComplexF64),
        QS.RealComplexKernel(Val(4), Val(2), ComplexF64),
    )
    for k in kernels
        @test pack_formats(typeof(k)) === (QS.a_format(k), QS.b_format(k))
    end
end

@testset "kernel construction: the complex default, 1m by name, and throws" begin
    for T in (ComplexF64, ComplexF32)
        kernel = host_kernel(T)
        @test kernel isa QuasiStrided.PlanarKernel
        @test scalartype(kernel) === T && realtype(kernel) === real(T)
        @test QuasiStrided.default_kernel_type(T) === PlanarKernel
        shape = (tile_size(kernel)..., lanewidth(kernel))
        profile = target_profile()
        @test planar_pressure(shape...) <= (profile.nregisters > 0 ? profile.nregisters : 16)
        profile.isa === :avx512 && @test shape === first(kernel_shapes(T, PlanarKernel))
        fms = profile.isa in (:avx512, :avx2)
        @test auto_kernel(T, 1024) isa (profile.isa === :avx2 ? QuasiStrided.FMAddSubKernel : QuasiStrided.PlanarKernel)
        # The small-M demotion: FMAddSub on AVX-512 (and AVX2's default), planar elsewhere.
        @test auto_kernel(T, 1) isa (fms ? QuasiStrided.FMAddSubKernel : QuasiStrided.PlanarKernel)

        shape1m = first(kernel_shapes(T, OneMKernel))
        k1m = QuasiStrided.kernel_from_shape(shape1m, T, OneMKernel)
        @test k1m isa OneMKernel
        @test (tile_size(k1m)..., lanewidth(k1m)) === shape1m
        # A kernel type with no menu for `T` throws, naming itself.
        err = try
            QuasiStrided.kernel_from_shape(shape1m, T, SIMDKernel)
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("SIMDKernel", err.msg) && occursin(string(T), err.msg)
    end
    for T in (ComplexF64, ComplexF32), K in (PlanarKernel, OneMKernel, FMAddSubKernel)
        for shape in kernel_shapes(T, K)
            k = kernel_from_shape(shape, T, K)
            @test k isa K && scalartype(k) === T && (tile_size(k)..., lanewidth(k)) === shape
        end
        @test_throws ArgumentError kernel_from_shape((7, 7, 7), T, K)
    end
    for T in (Float16, Int, ComplexF16)
        @test_throws ArgumentError QuasiStrided.kernel_from_shape((8, 6, 4), T)
    end
    @test_throws ArgumentError QuasiStrided.kernel_from_shape((8, 6, 4), Float64, OneMKernel)
    @test_throws ArgumentError QuasiStrided.kernel_from_shape((8, 6, 5), Float64)
    @test_throws ArgumentError QuasiStrided.menu_val((8, 6, 5), Float64, SIMDKernel)
    @test QuasiStrided.menu_val((8, 6, 4), Float64, SIMDKernel) === Val(SIMDKernel{8, 6, Float64, 4})
end

@testset "small_m_shape: AVX-512 complex small-M demotion to FMAddSub" begin
    sms(T, m_length, p = synthetic(:avx512)) = QuasiStrided.small_m_shape(p, T, m_length)
    # Least padded rows, then the larger tile.
    @test sms(ComplexF64, 12) === (12, 8, 8)
    @test sms(ComplexF64, 16) === (8, 8, 8)
    @test sms(ComplexF64, 20) === (12, 8, 8)
    @test sms(ComplexF32, 12) === (16, 8, 16)
    @test sms(ComplexF32, 16) === (16, 8, 16)
    for T in (ComplexF64, ComplexF32), m_length in 1:47
        shape = sms(T, m_length)
        @test shape in kernel_shapes(T, FMAddSubKernel)
        @test shape[3] == 64 ÷ sizeof(real(T))
    end
    @test sms(Float64, 4) === nothing
    for isakey in (:avx2, :neon, :unknown), T in (ComplexF64, ComplexF32)
        @test sms(T, 2, synthetic(isakey)) === nothing
    end
end

@testset "run-length-aware kernel demotion" begin
    # ccsd_t_1 with C's leading run exactly `d` and m_length != run, so only
    # `run % m_tile == 0` avoids demotion. Expectations are derived from the
    # menu, so this holds on every ISA.
    d = 16
    extra = 4
    IA = (:i, :j, :m, :a)
    IB = (:m, :k, :b, :c)
    IC = (:a, :b, :c, :i, :j, :k)
    (indA, indB, indC), _ = _lo_labels(IA, IB)

    for T in (Float64, Float32)
        A = randn(T, extra, extra, extra, d)  # (i, j, m, a)
        B = randn(T, extra, extra, extra, extra)  # (m, k, b, c)
        C = zeros(T, d, extra, extra, extra, extra, extra)  # (a, b, c, i, j, k)
        Av, Bv, Cv = StridedView(A), StridedView(B), StridedView(C)

        mlab, nlab, klab = QuasiStrided.classify_labels(indA, indB, indC)
        msorted = _lo_order(mlab, indC, Cv)
        run = _lo_run(msorted, indC, Cv)
        cpos(l) = findfirst(==(l), indC)::Int
        m_length = prod(size(Cv, cpos(l)) for l in mlab)
        n_length = prod(size(Cv, cpos(l)) for l in nlab)

        plan = plan_contract(Cv, Av, indA, Bv, indB, indC)
        @test plan.Astorage === parent(Av)

        default_kernel = auto_kernel(T, m_length)
        default_m_tile = tile_size(default_kernel, 1)
        if m_length == run || run % default_m_tile == 0
            @test plan.kernel === default_kernel
        else
            candidates = [sh[1] for sh in QuasiStrided.kernel_shapes(T) if run % sh[1] == 0]
            if isempty(candidates)
                @test plan.kernel === default_kernel
            else
                @test run % tile_size(plan.kernel, 1) == 0
                @test tile_size(plan.kernel, 1) == maximum(candidates)
            end
        end

        Cref = _lo_reference(C, Av, indA, Bv, indB, indC; alpha = 1.3, beta = -0.7)
        Cex = copy(C)
        plan_ex = plan_contract(StridedView(Cex), Av, indA, Bv, indB, indC)
        execute!(plan_ex, 1.3, -0.7)
        @test Cex ≈ Cref
    end

    # Plain GEMM: `m_length == run`, so no demotion; SIMDKernel is the
    # engine-wide default, and `contract!` runs the same plan.
    for T in (Float64, Float32)
        Ma, Ka, Na = 37, 11, 23
        Amat = randn(T, Ma, Ka)
        Bmat = randn(T, Ka, Na)
        Cmat = zeros(T, Ma, Na)
        Av, Bv, Cv = StridedView(Amat), StridedView(Bmat), StridedView(Cmat)
        plan = plan_contract(Cv, Av, (1, 2), Bv, (2, 3), (1, 3))
        @test plan.kernel isa SIMDKernel && plan.kernel === auto_kernel(T, Ma)

        execute!(plan, one(T), zero(T))
        @test Cmat ≈ Amat * Bmat
        C2 = zeros(T, Ma, Na)
        contract!(StridedView(C2), one(T), Av, (1, 2), Bv, (2, 3), zero(T), (1, 3))
        @test C2 == Cmat
    end
end

@testset "run-length demotion K-depth guard: deep-K does not demote, shallow-K does" begin
    # C[a,b,c,i,j,k] = A[i,j,m,a] * B[m,k,b,c], i=j=k=b=c=6: k_length = m is swept.
    IA = (:i, :j, :m, :a)
    IB = (:m, :k, :b, :c)
    IC = (:a, :b, :c, :i, :j, :k)
    (indA, indB, indC), _ = _lo_labels(IA, IB)

    function _run_demote_fixture(::Type{T}, a::Int, m::Int) where {T}
        i = j = k = b = c = 6
        A = randn(T, i, j, m, a)
        B = randn(T, m, k, b, c)
        C = zeros(T, a, b, c, i, j, k)
        return StridedView(C), StridedView(A), StridedView(B)
    end

    kmax_of(::Type{Float64}) = QuasiStrided.RUN_DEMOTE_KMAX_F64
    kmax_of(::Type{Float32}) = QuasiStrided.RUN_DEMOTE_KMAX_F32

    for (T, a) in ((Float64, 8), (Float32, 16))
        m_length = a * 36
        n_length = 216
        default_kernel = auto_kernel(T, m_length)

        # Deep K never demotes; the fixture never swaps.
        m_deep = 2 * kmax_of(T)
        Cv, Av, Bv = _run_demote_fixture(T, a, m_deep)
        plan_deep = plan_contract(Cv, Av, indA, Bv, indB, indC)
        @test plan_deep.Astorage === parent(Av)
        @test plan_deep.kernel === default_kernel

        # Shallow K demotes where the default's `m_tile` breaks the run (ISA-dependent).
        m_shallow = 8
        Cv2, Av2, Bv2 = _run_demote_fixture(T, a, m_shallow)
        plan_shallow = plan_contract(Cv2, Av2, indA, Bv2, indB, indC)
        @test plan_shallow.Astorage === parent(Av2)
        mlab, = QuasiStrided.classify_labels(indA, indB, indC)
        msorted = _lo_order(mlab, indC, Cv2)
        run = _lo_run(msorted, indC, Cv2)
        if m_length == run || run % tile_size(default_kernel, 1) == 0
            @test plan_shallow.kernel === default_kernel
        else
            @test plan_shallow.kernel !== default_kernel
            @test typeof(plan_shallow.kernel) !== typeof(default_kernel)
            @test run % tile_size(plan_shallow.kernel, 1) == 0
        end

        # The guard is `k_length > kmax`: `k_length == kmax` may still demote.
        Cv_b, Av_b, Bv_b = _run_demote_fixture(T, a, kmax_of(T))
        plan_b = plan_contract(Cv_b, Av_b, indA, Bv_b, indB, indC)
        run_b = _lo_run(msorted, indC, Cv_b)  # same M order at every m in this fixture
        if m_length == run_b || run_b % tile_size(default_kernel, 1) == 0
            @test plan_b.kernel === default_kernel
        else
            @test plan_b.kernel !== default_kernel
        end

        Cv_b1, Av_b1, Bv_b1 = _run_demote_fixture(T, a, kmax_of(T) + 1)
        plan_b1 = plan_contract(Cv_b1, Av_b1, indA, Bv_b1, indB, indC)
        @test plan_b1.kernel === default_kernel
    end
end
