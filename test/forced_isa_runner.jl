# Run the whole test suite as if the host had a different vector ISA, by
# overriding the process-wide profile that `_init_target!` populates.
#
# This catches a class of bug a local run cannot see: a test that asserts
# something only true on the host it was written on (a hardware-derived
# constant or capability silently makes every test asserting its value
# platform-dependent).
#
#   QS_FAKE_ISA=avx2 QS_FAKE_VB=32 QS_FAKE_NREG=16 \
#       julia --project=. test/forced_isa_runner.jl
#
# Useful triples: avx512/64/32, avx2/32/16, neon/16/32, unknown/0/0.
#
# The cache levels default to "not detected" (all zero); set the L1d/L2 size
# (bytes) and the number of CPUs sharing L2 to model a specific host, e.g.
# GitHub's macOS arm64 runner (a 3-vCPU virtual M1, per its sysctl):
#
#   QS_FAKE_ISA=neon QS_FAKE_VB=16 QS_FAKE_NREG=32 QS_FAKE_L1D=131072 \
#       QS_FAKE_L2=12582912 QS_FAKE_L2_SHARING=3 julia --project=. test/forced_isa_runner.jl
#
# Expected residue: `hardware/test_target.jl`'s "runs on this host without throwing"
# compares a fresh `_detect_target()` against the stored profile and therefore
# fails by construction here -- that one failure is this harness, not the
# package. Run from the test environment (`Pkg.test`-style) if you want the
# Bumper-dependent allocator testsets too.
using QuasiStrided
const _QS = QuasiStrided

_QS._TARGET[] = _QS.TargetProfile(
    Symbol(get(ENV, "QS_FAKE_ISA", "avx512")), Sys.ARCH, "forced-isa-runner",
    parse(Int, get(ENV, "QS_FAKE_VB", "64")),
    parse(Int, get(ENV, "QS_FAKE_NREG", "32")),
    # Unset: exactly `CacheLevel()` ("not detected"), as before these knobs.
    let l1d = parse(Int, get(ENV, "QS_FAKE_L1D", "0"))
        _QS.CacheLevel(l1d, 0, 0, l1d > 0 ? 1 : 0)
    end,
    _QS.CacheLevel(
        parse(Int, get(ENV, "QS_FAKE_L2", "0")), 0, 0,
        parse(Int, get(ENV, "QS_FAKE_L2_SHARING", "0"))
    ),
    _QS.CacheLevel()
)

let p = _QS.target_profile()
    println(
        "forced target: isa=", p.isa, " vector_bytes=", p.vector_bytes,
        " nregisters=", p.nregisters, " l1d=", p.l1d.bytes, " l2=", p.l2.bytes,
        "/", p.l2.sharing
    )
end

include(joinpath(@__DIR__, "runtests.jl"))
