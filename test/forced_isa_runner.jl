# Run the whole test suite as if the host had a different vector ISA, to catch
# tests that assert something true only on the host they were written on:
#
#   QS_FAKE_ISA=avx2 QS_FAKE_VB=32 QS_FAKE_NREG=16 \
#       julia --project=. test/forced_isa_runner.jl
#
# Useful triples: avx512/64/32, avx2/32/16, neon/16/32, unknown/0/0. The cache
# levels default to "not detected"; QS_FAKE_L1D, QS_FAKE_L2 (bytes) and
# QS_FAKE_L2_SHARING (CPUs) model a specific host, e.g. a 3-CPU M1:
#
#   QS_FAKE_ISA=neon QS_FAKE_VB=16 QS_FAKE_NREG=32 QS_FAKE_L1D=131072 \
#       QS_FAKE_L2=12582912 QS_FAKE_L2_SHARING=3 julia --project=. test/forced_isa_runner.jl
#
# "runs on this host without throwing" in hardware/test_target.jl fails by
# construction here. Run from the test environment for the Bumper testsets.

using QuasiStrided
const _QS = QuasiStrided

_QS._TARGET[] = _QS.TargetProfile(
    Symbol(get(ENV, "QS_FAKE_ISA", "avx512")), Sys.ARCH, "forced-isa-runner",
    parse(Int, get(ENV, "QS_FAKE_VB", "64")),
    parse(Int, get(ENV, "QS_FAKE_NREG", "32")),
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
