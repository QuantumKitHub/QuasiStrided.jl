# Register shape, kernel and blocking selection from a `TargetProfile`.

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

    @testset "shipped defaults are allocation-free on a scattered destination" begin
        for T in (Float64, Float32)
            plan = QuasiStrided.plan_contract(scattered_fixture(T)...)
            QuasiStrided.execute!(plan, one(T), zero(T))
            QuasiStrided.execute!(plan, one(T), zero(T))
            @test (@allocated QuasiStrided.execute!(plan, one(T), zero(T))) == 0 skip = (VERSION < v"1.11")
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
            kmax = T === Float64 ? QuasiStrided._RUN_DEMOTE_KMAX_F64 : QuasiStrided._RUN_DEMOTE_KMAX_F32
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

@testset "analytical blocking model" begin
    KiB, MiB = 1024, 1024^2
    # No SMT with an L3 shared by 4 cores; SMT 2 with an L3 shared by 8 cores.
    nosmt = TargetProfile(
        :avx2, "znver2", CacheLevel(32KiB, 64, 1),
        CacheLevel(512KiB, 64, 1), CacheLevel(16MiB, 64, 4)
    )
    smt2 = TargetProfile(
        :avx512, "cascadelake", CacheLevel(32KiB, 64, 2),
        CacheLevel(1MiB, 64, 2), CacheLevel(25952256, 64, 16)
    )
    @test modelled_blocking(nosmt, 8, 6, 8, 8) === Blocking(96, 341, 1728)
    @test modelled_blocking(nosmt, 16, 6, 4, 4) === Blocking(96, 682, 1728)
    @test modelled_blocking(smt2, 16, 6, 8, 8) === Blocking(192, 341, 1572)
    # A complex kernel keeps the real default kernel's `k_block` and blocks M
    # and N by its packed bytes: 1m packs 4 reals per A element, 2 per B element.
    @test default_blocking(kernel_from_shape((12, 8, 8), ComplexF64, OneMKernel), nosmt) === Blocking(24, 341, 864)
    for p in (nosmt, smt2), T in (Float64, Float32, ComplexF64, ComplexF32), K in (SIMDKernel, PlanarKernel, OneMKernel, FMAddSubKernel)
        for shape in kernel_shapes(T, K)
            k = kernel_from_shape(shape, T, K)
            b = default_blocking(k, p)
            MR, NR = tile_size(k)
            @test b.m_block % MR == 0 && b.n_block % NR == 0
            real_default = kernel_from_shape(derived_shape(p, real(T)), real(T), SIMDKernel)
            @test b.k_block == default_blocking(real_default, p).k_block
        end
    end
    # No L3: the B panel is budgeted from the L2 alone.
    nol3 = TargetProfile(
        :neon, "", CacheLevel(64KiB, 64, 1),
        CacheLevel(4MiB, 64, 4), CacheLevel()
    )
    @test modelled_blocking(nol3, 4, 6, 8, 8).n_block == (1MiB ÷ (682 * 8)) ÷ 6 * 6
    tiny = TargetProfile(
        :avx2, "", CacheLevel(64, 64, 1), CacheLevel(64, 64, 1), CacheLevel()
    )
    @test modelled_blocking(tiny, 8, 6, 8, 8) === Blocking(8, 1, 6)
    @test modelled_blocking(unknown_target(), 8, 6, 8, 8) === nothing
    @test modelled_blocking(
        TargetProfile(:avx2, "", CacheLevel(32KiB, 64, 1), CacheLevel(), CacheLevel()),
        8, 6, 8, 8
    ) === nothing
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

@testset "undetected caches: the fallback row scaled by packed reals" begin
    u = unknown_target()
    for T in (ComplexF64, ComplexF32), K in (PlanarKernel, OneMKernel, FMAddSubKernel)
        b = default_blocking(kernel_from_shape(first(kernel_shapes(T, K)), T, K), u)
        base = fallback_blocking(real(T))
        @test b.k_block === base.k_block
        @test b.m_block === base.m_block ÷ reals_per_element(pack_formats(K)[1])
        @test b.n_block === base.n_block ÷ reals_per_element(pack_formats(K)[2])
    end
    @test default_blocking(kernel_from_shape((24, 3, 8), ComplexF64, PlanarKernel), u) === Blocking(64, 256, 384)
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
        @test auto_kernel(T, 1024) isa QuasiStrided.PlanarKernel
        # The small-M demotion: FMAddSub on AVX-512, planar elsewhere.
        small = auto_kernel(T, 1)
        @test small isa (profile.isa === :avx512 ? QuasiStrided.FMAddSubKernel : QuasiStrided.PlanarKernel)

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
