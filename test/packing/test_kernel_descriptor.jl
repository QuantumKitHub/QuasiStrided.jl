using QuasiStrided: ComplexKernelDescriptor, RealFormat, PlanarFormat, OneEFormat,
    InterleavedFormat, realtype, packed_a_per_k, packed_b_per_k

@testset "KernelDescriptor" begin
    k = KernelDescriptor(Val(8), Val(6), Float64)
    @test (mr(k), nr(k), scalartype(k), realtype(k)) == (8, 6, Float64, Float64)
    @test packed_a_offset(k, 3, 2) == 3 + 8 * 2
    @test packed_b_offset(k, 5, 2) == 5 + 6 * 2
    @test packed_a_length(k, 0) == 0
    @test packed_a_length(k, 4) == 32
    @test packed_b_length(k, 4) == 24

    @test_throws ArgumentError KernelDescriptor(Val(0), Val(6), Float64)
    @test_throws ArgumentError KernelDescriptor(Val(8), Val(-1), Float64)
    @test_throws ArgumentError KernelDescriptor(Val(8), Val(6), Int)
end

@testset "ComplexKernelDescriptor: lengths count reals at a logical kc ($T)" for
    T in (ComplexF64, ComplexF32)

    MR, NR = 8, 6
    planar = ComplexKernelDescriptor(Val(MR), Val(NR), T, PlanarFormat(), PlanarFormat())
    onem = ComplexKernelDescriptor(Val(MR), Val(NR), T, OneEFormat(), PlanarFormat())
    fmas = ComplexKernelDescriptor(Val(MR), Val(NR), T, InterleavedFormat(), PlanarFormat())
    @test realtype(planar) === real(T)
    @test (packed_a_per_k(planar), packed_a_per_k(onem), packed_a_per_k(fmas)) ==
        (2MR, 4MR, 2MR)
    @test packed_b_per_k(planar) == packed_b_per_k(onem) == packed_b_per_k(fmas) == 2NR
    @test packed_a_length(onem, 7) == 4MR * 7
    @test packed_b_length(planar, 7) == 2NR * 7

    @test_throws ArgumentError ComplexKernelDescriptor(Val(MR), Val(NR), T, RealFormat(), PlanarFormat())
    @test packed_b_per_k(ComplexKernelDescriptor(Val(MR), Val(NR), T, RealFormat(), InterleavedFormat())) == 2NR
    @test_throws ArgumentError ComplexKernelDescriptor(Val(MR), Val(NR), Float64, PlanarFormat(), PlanarFormat())
    @test_throws ArgumentError ComplexKernelDescriptor(Val(0), Val(NR), T, PlanarFormat(), PlanarFormat())
end
