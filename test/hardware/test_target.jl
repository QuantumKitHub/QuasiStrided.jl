# Hardware detection (src/hardware/target.jl) and the register shape / blocking it
# selects. Two properties matter most: detection never throws, and every
# failure path resolves to exactly the fallback constants.

using StridedViews: StridedView
using QuasiStrided: TargetProfile, CacheLevel, target_profile, cache_topology,
    unknown_target, _detect_isa, _detect_target, _derived_shape, _fallback_shape,
    _shape_override, _kernel_for, _default_kernel, _fallback_blocking, kernel_shapes,
    _parse_size, _count_cpu_list, NR_DEFAULT, _rule_applies,
    _isa_nregisters, packed_a_per_k, packed_b_per_k, realtype, complex_method,
    RealMethod, PlanarMethod, OneMMethod, accumulator_planes, a_reals, b_reals


@testset "target detection" begin
    @testset "runs on this host without throwing" begin
        p = target_profile()
        # Logged, so a host-dependent CI failure can be read off the log.
        @info "target profile" p Sys.CPU_NAME QuasiStrided._isa_from_cpuid() VERSION
        @test p isa TargetProfile
        @test p.isa in VALID_ISAS
        @test p.arch === Sys.ARCH
        @test _detect_target().isa === p.isa   # detection is per-process
        @test _detect_isa() in VALID_ISAS
        # vector_bytes/nregisters are 0 exactly when the ISA is unknown.
        if p.isa === :unknown
            @test (p.vector_bytes, p.nregisters) == (0, 0)
        else
            @test p.vector_bytes > 0
            @test p.nregisters > 0
            @test ispow2(p.vector_bytes)
        end
    end

    @testset "CPUID probe is an independent fallback, not decoration" begin
        # The uarch table is an optimization; the probe is what makes an
        # *unlisted* CPU still get the right shape -- the portability claim.
        probe = QuasiStrided._isa_from_cpuid()
        @test probe in (:avx512, :avx2, :unknown)
        if Sys.ARCH === :x86_64
            table = get(QuasiStrided._UARCH_ISA, Sys.CPU_NAME, :miss)
            # A hypervisor can mask CPUID features down from what the named
            # microarchitecture nominally supports without renaming the CPU
            # (observed on a GitHub Actions runner: an AVX-512-listed name,
            # AVX2-only live probe) -- `_detect_isa` itself now defers to the
            # narrower of the two (src/hardware/target.jl), so only a probe
            # reporting something WIDER than the table is a real table bug.
            #
            # Except where the NAME is stale: a Julia whose CPU detector
            # predates a microarchitecture reports it under an older name.
            # Julia 1.10 (LLVM 15) has no `znver4`, and names AMD Zen 4 and
            # Zen 5 (AVX-512) "znver3" (AVX2) -- a GitHub ubuntu runner on
            # Zen 4 then probes AVX-512 against an AVX2 entry. That entry is
            # right for real Zen 3, so it is not a table bug; the name just
            # cannot tell the two apart. There only the reconciliation itself
            # is checked (below, for every name).
            stale = VERSION < v"1.11" ? ("znver3",) : ()
            (table === :miss || probe === :unknown || Sys.CPU_NAME in stale) ||
                @test QuasiStrided._isa_rank(probe) <= QuasiStrided._isa_rank(table)
            # `_detect_isa` is the narrower of a listed name and a live probe.
            if table !== :miss && probe !== :unknown
                @test _detect_isa() ===
                    (QuasiStrided._isa_rank(probe) < QuasiStrided._isa_rank(table) ? probe : table)
            end
            # Base's CPUID submodule is undocumented: if these names move,
            # `_isa_from_cpuid` silently returns :unknown and every unlisted
            # x86 CPU quietly loses the derived shape. Fail loudly instead.
            C = Base.BinaryPlatforms.CPUID
            @test isdefined(C, :JL_X86_avx512f)
            @test isdefined(C, :JL_X86_avx2)
        end
    end

    @testset "cache topology is optional and non-negative" begin
        for _ in 1:2   # must not throw when called repeatedly (shells out on macOS)
            topo = cache_topology()
            @test topo === nothing || topo isa NamedTuple
            topo === nothing && continue
            for lvl in (topo.l1d, topo.l2, topo.l3)
                @test lvl isa CacheLevel
                # Any field may be 0 for "not detected", never negative.
                @test all(>=(0), (lvl.bytes, lvl.ways, lvl.line, lvl.sharing))
            end
        end
    end

    @testset "sysfs parsing helpers" begin
        for (str, want) in (
                "32K" => 32 * 1024, "1024K" => 1024 * 1024,
                "25344K" => 25344 * 1024, "2M" => 2 * 1024 * 1024,
                "512" => 512, "" => 0, "garbage" => 0,
            )
            @test _parse_size(str) == want
        end
        # "0,16" is a typical SMT-pair L2 list; "0-7,16-23" a shared L3.
        for (str, want) in ("0,16" => 2, "0-7,16-23" => 16, "3" => 1, "" => 0, "0-3" => 4)
            @test _count_cpu_list(str) == want
        end
    end

end
