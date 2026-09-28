# Hardware detection: it never throws, and every failure path resolves to the fallbacks.

using StridedViews: StridedView

@testset "target detection" begin
    @testset "runs on this host without throwing" begin
        p = target_profile()
        # Logged so that a host-dependent CI failure can be read off the log.
        @info "target profile" p Sys.CPU_NAME QuasiStrided._isa_from_cpuid() VERSION
        @test p.isa in VALID_ISAS
        @test p.arch === Sys.ARCH
        @test _detect_target().isa === p.isa
        if p.isa === :unknown
            @test (p.vector_bytes, p.nregisters) == (0, 0)
        else
            @test p.nregisters > 0
            @test p.vector_bytes > 0 && ispow2(p.vector_bytes)
        end
    end

    @testset "CPU name table and CPUID probe agree" begin
        probe = QuasiStrided._isa_from_cpuid()
        @test probe in (:avx512, :avx2, :unknown)
        if Sys.ARCH === :x86_64
            rank = QuasiStrided._isa_rank
            table = get(QuasiStrided._UARCH_ISA, Sys.CPU_NAME, :miss)
            # A probe narrower than the table is a masked hypervisor, not a table
            # bug. Julia 1.10 (LLVM 15) names AVX-512 Zen 4/5 "znver3", so there a
            # wider probe is not a table bug either.
            stale = VERSION < v"1.11" ? ("znver3",) : ()
            (table === :miss || probe === :unknown || Sys.CPU_NAME in stale) ||
                @test rank(probe) <= rank(table)
            if table !== :miss && probe !== :unknown
                @test _detect_isa() === (rank(probe) < rank(table) ? probe : table)
            end
            # If Base's undocumented CPUID names move, every unlisted CPU would
            # silently lose its derived shape.
            C = Base.BinaryPlatforms.CPUID
            @test isdefined(C, :JL_X86_avx512f)
            @test isdefined(C, :JL_X86_avx2)
        end
    end

    @testset "cache topology is optional and non-negative" begin
        for _ in 1:2   # repeated calls must not throw (macOS shells out)
            topo = cache_topology()
            @test topo === nothing || topo isa NamedTuple
            topo === nothing && continue
            for lvl in (topo.l1d, topo.l2, topo.l3)
                @test all(>=(0), (lvl.bytes, lvl.ways, lvl.line, lvl.sharing))
            end
        end
    end

    @testset "sysfs parsing helpers" begin
        for (str, want) in (
                "32K" => 32 * 1024, "25344K" => 25344 * 1024, "2M" => 2 * 1024 * 1024,
                "512" => 512, "" => 0, "garbage" => 0,
            )
            @test _parse_size(str) == want
        end
        for (str, want) in ("0,16" => 2, "0-7,16-23" => 16, "3" => 1, "" => 0)
            @test _count_cpu_list(str) == want
        end
    end
end
