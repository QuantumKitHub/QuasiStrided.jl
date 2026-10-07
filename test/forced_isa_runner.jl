# Run the whole test suite as if the host had a different vector ISA, to catch
# tests that assert something true only on the host they were written on:
#
#   QS_FAKE_ISA=avx2 julia --project=. test/forced_isa_runner.jl
#
# QS_FAKE_ISA is one of avx512 (the default), avx2, neon, unknown. The cache
# levels default to "not detected"; QS_FAKE_L1D, QS_FAKE_L2 (bytes) and
# QS_FAKE_L2_SHARING (CPUs) model a specific host, e.g. a 3-CPU M1:
#
#   QS_FAKE_ISA=neon QS_FAKE_L1D=131072 QS_FAKE_L2=12582912 QS_FAKE_L2_SHARING=3 \
#       julia --project=. test/forced_isa_runner.jl
#
# Arguments are passed on to runtests.jl. "runs on this host without throwing"
# in hardware/test_target.jl fails by construction here.

using Pkg

get!(ENV, "QS_FAKE_ISA", "avx512")
Pkg.test("QuasiStrided"; test_args = ARGS)
