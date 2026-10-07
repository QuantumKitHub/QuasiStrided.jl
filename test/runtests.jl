# Every test/**/test_*.jl file runs in its own module, on a pool of worker
# processes. Arguments select files by name prefix, e.g.
#
#   Pkg.test("QuasiStrided"; test_args = ["execution/test_dot", "packing"])
#
# and `--list`, `--jobs=N`, `--verbose` are the runner's own. `Pkg.test` runs
# with `--check-bounds=yes`, kept deliberately: it also runs the per-tile bounds
# checks that `@inbounds` skips. `Pkg.test(; julia_args = ["--check-bounds=auto"])`
# is the fast local run.

using ParallelTestRunner
using QuasiStrided

testsuite = find_tests(@__DIR__)
filter!(p -> startswith(basename(first(p)), "test_"), testsuite)

# Set by forced_isa_runner.jl: every worker runs on that target.
fake_target = :()
if haskey(ENV, "QS_FAKE_ISA")
    fake_target = quote
        using QuasiStrided: TARGET, TargetProfile, CacheLevel
        TARGET[] = TargetProfile(
            Symbol(ENV["QS_FAKE_ISA"]), "forced-isa-runner",
            let l1d = parse(Int, get(ENV, "QS_FAKE_L1D", "0"))
                CacheLevel(l1d, 0, l1d > 0 ? 1 : 0)
            end,
            CacheLevel(parse(Int, get(ENV, "QS_FAKE_L2", "0")), 0, parse(Int, get(ENV, "QS_FAKE_L2_SHARING", "0"))),
            CacheLevel()
        )
    end
    eval(fake_target)
    let p = QuasiStrided.target_profile()
        println(
            "forced target: isa=", p.isa, " vector_bytes=", p.vector_bytes, " nregisters=", p.nregisters,
            " l1d=", p.l1d.bytes, " l2=", p.l2.bytes, "/", p.l2.sharing
        )
    end
end

runtests(QuasiStrided, ARGS; testsuite, init_worker_code = fake_target)
