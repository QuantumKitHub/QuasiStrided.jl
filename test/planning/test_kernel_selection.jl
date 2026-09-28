# Register shape, kernel and blocking selection from a `TargetProfile`.

using StridedViews: StridedView
using QuasiStrided: TargetProfile, CacheLevel, target_profile, unknown_target,
    _derived_shape, _fallback_shape, _shape_override, _kernel_for, _default_kernel,
    _fallback_blocking, kernel_shapes, NR_DEFAULT, _rule_applies, _isa_nregisters,
    packed_a_per_k, packed_b_per_k, realtype, complex_method, RealMethod,
    PlanarMethod, OneMMethod, FMAddSubMethod, accumulator_planes, a_reals, b_reals,
    _modelled_blocking, _scale_blocking, _real_blocking_row, _planar_pressure

@testset "real shape selection" begin
    @testset "unknown target: fallback shape and blocking" begin
        u = unknown_target()
        for T in (Float64, Float32)
            @test _derived_shape(u, T) === _fallback_shape(T)
            k = _kernel_for(u, T)
            @test k isa SIMDKernel
            @test (mr(k), nr(k), lanewidth(k)) === _fallback_shape(T)
            @test _real_blocking_row(u, T) === _fallback_blocking(T)
        end
        @test _fallback_blocking(Float64) === Blocking(128, 256, 768)
        @test _fallback_blocking(Float32) === Blocking(96, 768, 1152)
        # The type-argument form ignores detection.
        @test default_blocking(Float64) === _fallback_blocking(Float64)
        @test default_blocking(Float32) === _fallback_blocking(Float32)
    end

    @testset "rule: MR = MV*W, NR = 6; MV = 4 on Intel :avx512, 2 on :avx2 and AMD" begin
        for (isakey, vb, MV) in ((:avx512, 64, 4), (:avx2, 32, 2)), T in (Float64, Float32)
            W = vb ÷ sizeof(T)
            @test _derived_shape(synthetic(isakey, vb), T) === (MV * W, NR_DEFAULT, W)
            @test QuasiStrided._rule_mv(Val(isakey), RealMethod()) == MV
            @test QuasiStrided._rule_shape(vb, T, MV) === (MV * W, NR_DEFAULT, W)
        end
        for m in (PlanarMethod(), OneMMethod(), FMAddSubMethod()), key in (:avx512, :avx2)
            @test QuasiStrided._rule_mv(Val(key), m) == 2
        end
        for (cpu, MV) in (("znver4", 4), ("icelake-server", 4), ("cascadelake", 4)),
                T in (Float64, Float32)
            p = TargetProfile(:avx512, Sys.ARCH, cpu, 64, 32, CacheLevel(), CacheLevel(), CacheLevel())
            W = 64 ÷ sizeof(T)
            @test _derived_shape(p, T) === (MV * W, NR_DEFAULT, W)
            @test _derived_shape(p, ComplexF64) === _derived_shape(synthetic(:avx512, 64), ComplexF64)
        end
        @test _derived_shape(synthetic(:avx2, 32), Float64) === _fallback_shape(Float64)
        # ISAs without a rule get the fallback shape, whatever their width.
        for T in (Float64, Float32), vb in (0, 16, 32, 64)
            @test _derived_shape(synthetic(:neon, vb), T) === _fallback_shape(T)
            @test _derived_shape(synthetic(:unknown, vb), T) === _fallback_shape(T)
        end
        # No real override row on any ISA: the rule is the optimum.
        for T in (Float64, Float32), key in VALID_ISAS
            @test _shape_override(Val(key), T) === nothing
        end
        @test _rule_applies(Val(:avx2), RealMethod()) && _rule_applies(Val(:avx512), RealMethod())
    end

    @testset "every real menu shape is constructible and fits the register file" begin
        for T in (Float64, Float32), (MR, NR, W) in kernel_shapes(T)
            @test MR % W == 0
            k = SIMDKernel(Val(MR), Val(NR), T, Val(W))
            @test (mr(k), nr(k), lanewidth(k)) == (MR, NR, W)
            @test (MR ÷ W) * NR + (MR ÷ W) <= 32
            @test packed_a_per_k(k) === MR && packed_b_per_k(k) === NR
            @test realtype(k) === T === scalartype(k)
            @test complex_method(k) === RealMethod()
            @test packed_a_length(k, 7) === MR * 7 && packed_b_length(k, 7) === NR * 7
        end
        for T in (Float64, Float32)
            W = 64 ÷ sizeof(T)
            # The MV = 2 sibling stays in the menu as the step-down target.
            @test (2 * W, NR_DEFAULT, W) in kernel_shapes(T)
            @test kernel_shapes(T, RealMethod()) === kernel_shapes(T)
            k = _default_kernel(T)
            @test k isa SIMDKernel && scalartype(k) === T
            @test (mr(k), nr(k), lanewidth(k)) in kernel_shapes(T)
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
            MR = mr(_default_kernel(T))
            @test mr(_default_kernel(T, 4 * MR, 256)) == MR
            small = _default_kernel(T, 1, 256)
            @test (mr(small), nr(small), lanewidth(small)) === _fallback_shape(T)
            @test mr(_default_kernel(T, 0, 256)) == MR        # empty must not demote
        end
    end

    @testset "_extent_shape: MV = 4 steps down to MV = 2 where it pads less" begin
        ext(shape, T, Qm) = QuasiStrided._extent_shape(shape, T, RealMethod(), Qm)
        for T in (Float64, Float32)
            tall = _derived_shape(synthetic(:avx512, 64), T)
            MR, NR, W = tall
            half = (2 * W, NR, W)
            @test ext(tall, T, 0) === tall
            for Qm in (1, 2 * W - 1, 2 * W, MR - 1, MR + 1, MR + W, MR + 2 * W)
                @test ext(tall, T, Qm) === half
            end
            for Qm in (MR, MR + 2 * W + 1, 2 * MR - 1, 2 * MR, 2 * MR + 1, 3 * MR + W, 100 * MR)
                @test ext(tall, T, Qm) === tall
            end
            # Never below MV = 2, and complex methods are untouched.
            avx2 = _derived_shape(synthetic(:avx2, 32), T)
            @test ext(avx2, T, 3) === avx2
            cs = _derived_shape(synthetic(:avx512, 64), ComplexF64)
            @test QuasiStrided._extent_shape(cs, ComplexF64, PlanarMethod(), 3) === cs
            # Through `_default_kernel`, where the host's shape is the tall one.
            if _derived_shape(target_profile(), T) === tall
                @test (mr(_default_kernel(T, MR + W, 256)), nr(_default_kernel(T, MR + W, 256))) == (2 * W, NR)
                @test mr(_default_kernel(T, 2 * W, 256)) == 2 * W
                @test mr(_default_kernel(T, 2 * W - 1, 256)) == _fallback_shape(T)[1]
                @test mr(_default_kernel(T, 2 * MR, 256)) == MR
            end
        end
    end

    @testset "_store_shape: MV = 4 steps down where C's run breaks its slivers" begin
        st = QuasiStrided._store_shape
        for T in (Float64, Float32)
            tall = _derived_shape(synthetic(:avx512, 64), T)
            MR, NR, W = tall
            half = (2 * W, NR, W)
            Qm = 64 * MR
            for run in (Qm, MR, 2 * MR, 5 * MR)                  # every tall sliver whole
                @test st(tall, T, RealMethod(), Qm, run) === tall
            end
            @test st(tall, T, RealMethod(), 3 * W, 3 * W) === tall
            for run in (2 * W, 3 * W, 6 * W, MR + 2 * W, Qm - 1)
                @test st(tall, T, RealMethod(), Qm, run) === half
            end
            for run in (1, W, 2 * W - 1)                          # below one half tile
                @test st(tall, T, RealMethod(), Qm, run) === tall
            end
            @test st(half, T, RealMethod(), Qm, 3 * W) === half
            @test st(_fallback_shape(T), T, RealMethod(), Qm, 3 * W) === _fallback_shape(T)
            cs = _derived_shape(synthetic(:avx512, 64), ComplexF64)
            @test st(cs, ComplexF64, PlanarMethod(), Qm, 3 * W) === cs
            if _derived_shape(target_profile(), T) === tall
                @test QuasiStrided._default_shape(T, Qm, 256, 2 * W)[1] === half
                @test QuasiStrided._default_shape(T, Qm, 256, MR)[1] === tall
                @test QuasiStrided._default_shape(T, Qm, 256)[1] === tall   # no run given
            end
        end
    end

    @testset "_store_shape feeds the swap: ccsd_t_2 / ao2mo_2 at dim 16" begin
        # Regression: C's only unit run is 16 long, on the N side. Judged
        # against mr = 32 the swap was declined and every tile stored scattered.
        T = Float64
        if _derived_shape(target_profile(), T)[1] == 32
            d = 16
            a, b, c, i, j, k, m = 1, 2, 3, 4, 5, 6, -1
            Bv = StridedView(randn(T, d, d, d, d))
            plan = QuasiStrided.plan_contract(
                StridedView(zeros(T, d, d, d, d, d, d)), StridedView(randn(T, d, d, d, d)),
                (i, j, m, b), Bv, (m, k, a, c), (a, b, c, i, j, k); oracle = false
            )
            @test mr(plan.kernel) == 16
            @test plan.Astorage === parent(Bv)
            q, r, s = -1, 3, 4
            B2 = StridedView(randn(T, d, d, d, d))
            plan2 = QuasiStrided.plan_contract(
                StridedView(zeros(T, d, d, d, d)), StridedView(randn(T, d, d)), (q, b),
                B2, (a, q, r, s), (a, b, r, s); oracle = false
            )
            @test mr(plan2.kernel) == 16
            @test plan2.Astorage === parent(B2)
            @test axis_length(plan2.mgroup) == d^3
        end
    end
end

@testset "complex shape selection" begin
    for (T, W, swept) in ((ComplexF64, 8, (24, 3, 8)), (ComplexF32, 16, (48, 3, 16)))
        # W counts REAL lanes; the AVX-512 override replaces the spilling rule shape.
        @test QuasiStrided._rule_shape(64, T, 2) === (2 * W, NR_DEFAULT, W)
        @test _shape_override(Val(:avx512), T) === swept
        @test _derived_shape(synthetic(:avx512, 64), T) === swept === first(kernel_shapes(T, PlanarMethod()))
        @test Set(kernel_shapes(T, PlanarMethod())) == Set(
            (
                (2 * W, NR_DEFAULT, W), swept, (W, 8, W), _shape_override(Val(:avx2), T),
                _shape_override(Val(:neon), T), (W ÷ 4, NR_DEFAULT, W ÷ 4),
            )
        )
        # Every override row fits its ISA's register file with room to spare.
        for (key, nreg) in ((:avx512, 32), (:avx2, 16), (:neon, 32))
            ovr = _shape_override(Val(key), T)
            @test _planar_pressure(ovr...) < nreg
            @test ovr in kernel_shapes(T, PlanarMethod())
        end
        @test _fallback_shape(T) === (8, NR_DEFAULT, _fallback_shape(real(T))[3])
        # Off :avx512 an override row wins whatever the width; without a row
        # the shape is fitted to the register budget.
        for key in (:avx2, :neon), vb in (0, 16, 32, 64), nreg in (0, 16, 32)
            @test !_rule_applies(Val(key), PlanarMethod())
            @test _derived_shape(synthetic(key, vb; nregisters = nreg), T) === _shape_override(Val(key), T)
        end
        for key in (:unknown, :somethingelse), vb in (0, 16, 32, 64), nreg in (0, 16, 32)
            @test !_rule_applies(Val(key), PlanarMethod())
            @test _shape_override(Val(key), T) === nothing
            shape = _derived_shape(synthetic(key, vb; nregisters = nreg), T)
            @test _planar_pressure(shape...) <= (nreg > 0 ? nreg : 16)
        end
        @test _derived_shape(synthetic(:avx512, 0), T) in kernel_shapes(T, PlanarMethod())
    end
    @test _rule_applies(Val(:avx512), PlanarMethod())
    @test _planar_pressure(_fallback_shape(ComplexF64)...) == 30
    @test _planar_pressure(_fallback_shape(ComplexF32)...) == 16

    # Complex menus fit AVX-512 and stay bounded.
    nreg = _isa_nregisters(Val(:avx512))
    target_profile().isa === :avx512 && @test target_profile().nregisters == nreg
    for T in (ComplexF64, ComplexF32), m in (PlanarMethod(), OneMMethod())
        planes = accumulator_planes(m)
        for (MR, NR, W) in kernel_shapes(T, m)
            rows = (2 * MR) ÷ planes
            @test rows % W == 0
            mv = rows ÷ W
            @test planes * mv * NR + planes * mv + planes <= nreg
        end
    end
    for T in (ComplexF64, ComplexF32)
        @test length(kernel_shapes(T, PlanarMethod())) <= 6
        @test length(kernel_shapes(T, OneMMethod())) <= 4
    end
end

@testset "_derived_shape is always a constructible member of the method's menu" begin
    methods = (
        (Float64, RealMethod()), (Float32, RealMethod()), (ComplexF64, PlanarMethod()),
        (ComplexF32, PlanarMethod()), (ComplexF64, OneMMethod()), (ComplexF32, OneMMethod()),
    )
    for (T, m) in methods, isakey in (VALID_ISAS..., :somethingelse),
            vb in (0, 16, 32, 64, 128), nreg in (0, 16, 32)
        shape = _derived_shape(synthetic(isakey, vb; nregisters = nreg), T, m)
        @test shape in kernel_shapes(T, m)
        k = QuasiStrided._kernel_from_shape(shape, T, m)
        @test (mr(k), nr(k), lanewidth(k)) === shape
    end
end

@testset "analytical blocking model" begin
    KiB, MiB = 1024, 1024^2
    # No SMT with an L3 shared by 4 cores; SMT 2 with an L3 shared by 8 cores.
    nosmt = TargetProfile(
        :avx2, :x86_64, "znver2", 32, 16, CacheLevel(32KiB, 8, 64, 1),
        CacheLevel(512KiB, 8, 64, 1), CacheLevel(16MiB, 16, 64, 4)
    )
    smt2 = TargetProfile(
        :avx512, :x86_64, "cascadelake", 64, 32, CacheLevel(32KiB, 8, 64, 2),
        CacheLevel(1MiB, 16, 64, 2), CacheLevel(25952256, 11, 64, 16)
    )
    @test _modelled_blocking(nosmt, Float64, 8, 6) === Blocking(96, 341, 1728)
    @test _modelled_blocking(nosmt, Float32, 16, 6) === Blocking(96, 682, 1728)
    @test _modelled_blocking(smt2, Float64, 16, 6) === Blocking(192, 341, 1572)
    @test _modelled_blocking(nosmt, Float64) === _modelled_blocking(nosmt, Float64, 8, 6)
    for p in (nosmt, smt2), T in (Float64, Float32)
        b = _modelled_blocking(p, T)
        MR, NR, _ = _derived_shape(p, T)
        @test b.mc % MR == 0 && b.nc % NR == 0
        @test NR * b.kc * sizeof(T) <= p.l1d.bytes ÷ 2
        @test _real_blocking_row(p, T) === b
    end
    # No L3: the B panel is budgeted from the L2 alone.
    nol3 = TargetProfile(
        :neon, :aarch64, "", 16, 32, CacheLevel(64KiB, 4, 64, 1),
        CacheLevel(4MiB, 16, 64, 4), CacheLevel()
    )
    @test _modelled_blocking(nol3, Float64, 4, 6).nc == (1MiB ÷ (682 * 8)) ÷ 6 * 6
    tiny = TargetProfile(
        :avx2, :x86_64, "", 32, 16, CacheLevel(64, 1, 64, 1), CacheLevel(64, 1, 64, 1), CacheLevel()
    )
    @test _modelled_blocking(tiny, Float64, 8, 6) === Blocking(8, 1, 6)
    @test _modelled_blocking(unknown_target(), Float64, 8, 6) === nothing
    @test _modelled_blocking(
        TargetProfile(:avx2, :x86_64, "", 32, 16, CacheLevel(32KiB, 8, 64, 1), CacheLevel(), CacheLevel()),
        Float64, 8, 6
    ) === nothing
end

@testset "complex blocking is the real row scaled by packed reals" begin
    for base in (_fallback_blocking(Float64), _fallback_blocking(Float32), Blocking(1, 5, 1))
        for m in (PlanarMethod(), OneMMethod(), FMAddSubMethod())
            b = _scale_blocking(base, m)
            @test b.kc === base.kc
            @test b.mc === max(1, base.mc ÷ a_reals(m))
            @test b.nc === max(1, base.nc ÷ b_reals(m))
        end
    end
    @test _scale_blocking(_fallback_blocking(Float64), PlanarMethod()) === Blocking(64, 256, 384)
    @test _scale_blocking(_fallback_blocking(Float64), OneMMethod()) === Blocking(32, 256, 384)
    for T in (ComplexF64, ComplexF32)
        row = _real_blocking_row(target_profile(), real(T))
        @test default_blocking(_default_kernel(T)) === _scale_blocking(row, PlanarMethod())
    end
    for T in (Float64, Float32)
        @test default_blocking(_default_kernel(T)) === _real_blocking_row(target_profile(), T)
    end
end

@testset "kernel construction: the complex default, 1m by name, and throws" begin
    for T in (ComplexF64, ComplexF32)
        kernel = _default_kernel(T)
        @test kernel isa QuasiStrided.PlanarKernel
        @test scalartype(kernel) === T && realtype(kernel) === real(T)
        @test complex_method(kernel) === PlanarMethod() === QuasiStrided._default_method(T)
        shape = (mr(kernel), nr(kernel), lanewidth(kernel))
        profile = target_profile()
        @test _planar_pressure(shape...) <= (profile.nregisters > 0 ? profile.nregisters : 16)
        profile.isa === :avx512 && @test shape === first(kernel_shapes(T, PlanarMethod()))
        @test _default_kernel(T, 1024, 1024) isa QuasiStrided.PlanarKernel
        # The small-M demotion: FMAddSub on AVX-512, planar elsewhere.
        small = _default_kernel(T, 1, 1)
        @test small isa (profile.isa === :avx512 ? QuasiStrided.FMAddSubKernel : QuasiStrided.PlanarKernel)

        shape1m = first(kernel_shapes(T, OneMMethod()))
        k1m = QuasiStrided._kernel_from_shape(shape1m, T, OneMMethod())
        @test k1m isa QuasiStrided.OneMKernel && complex_method(k1m) === OneMMethod()
        @test (mr(k1m), nr(k1m), lanewidth(k1m)) === shape1m
        # A method with no kernel for `T` throws, naming itself.
        err = try
            QuasiStrided._kernel_from_shape(shape1m, T, RealMethod())
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("RealMethod", err.msg) && occursin(string(T), err.msg)
    end
    for T in (Float16, Int, ComplexF16)
        @test_throws ArgumentError QuasiStrided._kernel_from_shape((8, 6, 4), T)
    end
    @test_throws ArgumentError QuasiStrided._kernel_from_shape((8, 6, 4), Float64, OneMMethod())
    @test_throws ArgumentError QuasiStrided._kernel_from_shape((8, 6, 5), Float64)
    @test_throws ArgumentError QuasiStrided._menu_val((8, 6, 5), Float64, RealMethod())
    @test QuasiStrided._menu_val((8, 6, 4), Float64, RealMethod()) === Val((8, 6, 4))
end

@testset "_small_m_shape: AVX-512 complex small-M demotion to FMAddSub" begin
    sms(T, Qm, p = synthetic(:avx512, 64)) =
        QuasiStrided._small_m_shape(QuasiStrided._small_m_candidates(Val(p.isa), p, T), Qm)
    # Least padded rows, then the larger tile.
    @test sms(ComplexF64, 12) === (12, 8, 8)
    @test sms(ComplexF64, 16) === (8, 8, 8)
    @test sms(ComplexF64, 20) === (12, 8, 8)
    @test sms(ComplexF32, 12) === (16, 8, 16)
    @test sms(ComplexF32, 16) === (16, 8, 16)
    for T in (ComplexF64, ComplexF32), Qm in 1:47
        shape = sms(T, Qm)
        @test shape in kernel_shapes(T, FMAddSubMethod())
        @test shape[3] == 64 ÷ sizeof(real(T))
    end
    @test sms(Float64, 4) === nothing
    for isakey in (:avx2, :neon, :unknown), T in (ComplexF64, ComplexF32)
        @test sms(T, 2, synthetic(isakey, 32)) === nothing
    end
end

@testset "_demote_shape_for_run: largest menu shape whose mr divides the run" begin
    demote = QuasiStrided._demote_shape_for_run
    for T in (Float64, Float32), shape in kernel_shapes(T)
        MR = shape[1]
        @test demote(T, shape, RealMethod(), 3 * MR, 1000, 1) === shape
        @test demote(T, shape, RealMethod(), 7, 7, 1) === shape
        kmax = T === Float64 ? QuasiStrided._RUN_DEMOTE_KMAX_F64 : QuasiStrided._RUN_DEMOTE_KMAX_F32
        @test demote(T, shape, RealMethod(), 1, 1000, kmax + 1) === shape
        for run in (1, 4, 8, 12, 16, 20, 24, 48)
            run % MR == 0 && continue
            fits = [s for s in kernel_shapes(T) if run % s[1] == 0]
            want = isempty(fits) ? shape : fits[argmax(first.(fits))]
            @test demote(T, shape, RealMethod(), run, 1000, 1) === want
        end
    end
    # Complex kernels are never demoted.
    for T in (ComplexF64, ComplexF32)
        shape = first(kernel_shapes(T, PlanarMethod()))
        @test demote(T, shape, PlanarMethod(), 1, 1000, 1) === shape
    end
end
