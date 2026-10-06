using Test
using Aqua
using QuasiStrided

@testset "Aqua" begin
    Aqua.test_all(
        QuasiStrided;
        # Ambiguities between methods of upstream packages are not ours to fix.
        ambiguities = (recursive = false,),
        # Flags `acc::NTuple{NV, Vec{W, R}}` signatures, which leave `W`/`R`
        # unbound only for an empty accumulator, which no kernel has.
        unbound_args = false,
    )
end
