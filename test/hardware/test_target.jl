# Hardware detection: it never throws, and every failure path resolves to the fallbacks.

using StridedViews: StridedView

@testset "target detection" begin
    @testset "runs on this host without throwing" begin
        p = target_profile()
        # Logged so that a host-dependent CI failure can be read off the log.
        @info "target profile" p Sys.CPU_NAME VERSION
        @test p.isa in VALID_ISAS
        @test detect_target().isa === p.isa
        @test_throws ArgumentError synthetic(:sse)
    end

    if Sys.ARCH === :x86_64
        @testset "the ISA is the CPUID probe's" begin
            @test detect_isa() === QuasiStrided.isa_from_cpuid()
            # If Base's undocumented CPUID names move, every x86 host would
            # silently fall back to `:unknown`.
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
                @test all(>=(0), (lvl.bytes, lvl.line, lvl.sharing))
            end
        end
    end

    @testset "sysfs parsing helpers" begin
        for (str, want) in (
                "32K" => 32 * 1024, "25344K" => 25344 * 1024, "2M" => 2 * 1024 * 1024,
                "512" => 512, "" => 0, "garbage" => 0,
            )
            @test parse_size(str) == want
        end
        for (str, want) in ("0,16" => 2, "0-7,16-23" => 16, "3" => 1, "" => 0)
            @test count_cpu_list(str) == want
        end
    end
end
