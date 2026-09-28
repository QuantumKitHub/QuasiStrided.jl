using Test
using Aqua
using QuasiStrided

@testset "Aqua" begin
    Aqua.test_all(
        QuasiStrided;
        # Every flagged ambiguity is between methods of upstream packages.
        ambiguities = false,
        # Flags `acc::NTuple{NV, Vec{W, R}}` signatures, which leave `W`/`R`
        # unbound only for an empty accumulator, which no kernel has.
        unbound_args = false,
    )
end
