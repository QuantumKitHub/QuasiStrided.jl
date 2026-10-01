using QuasiStrided: RealFormat, PlanarFormat, OneEFormat,
    InterleavedFormat, realtype, sliver_width

@testset "Descriptor" begin
    k = Descriptor(Val(8), Val(6), Float64)
    @test (tile_size(k)..., scalartype(k), realtype(k)) == (8, 6, Float64, Float64)
    @test packed_a_offset(k, 4, 3) == 3 + 8 * 2
    @test packed_b_offset(k, 6, 3) == 5 + 6 * 2
    @test packed_a_length(k, 0) == 0
    @test packed_a_length(k, 4) == 32
    @test packed_b_length(k, 4) == 24

    @test_throws ArgumentError Descriptor(Val(0), Val(6), Float64)
    @test_throws ArgumentError Descriptor(Val(8), Val(-1), Float64)
    @test_throws ArgumentError Descriptor(Val(8), Val(6), Int)
end

@testset "Descriptor: lengths count reals at a logical k_block_length ($T)" for
    T in (ComplexF64, ComplexF32)

    MR, NR = 8, 6
    planar = Descriptor(Val(MR), Val(NR), T, PlanarFormat(), PlanarFormat())
    onem = Descriptor(Val(MR), Val(NR), T, OneEFormat(), PlanarFormat())
    fmas = Descriptor(Val(MR), Val(NR), T, InterleavedFormat(), PlanarFormat())
    @test realtype(planar) === real(T)
    @test (sliver_width(planar, 1), sliver_width(onem, 1), sliver_width(fmas, 1)) ==
        (2MR, 4MR, 2MR)
    @test sliver_width(planar, 2) == sliver_width(onem, 2) == sliver_width(fmas, 2) == 2NR
    @test packed_a_length(onem, 7) == 4MR * 7
    @test packed_b_length(planar, 7) == 2NR * 7
    @test packed_a_offset(planar, 3, 2, 1) == 2MR + MR + 2

    @test_throws ArgumentError Descriptor(Val(MR), Val(NR), T, RealFormat(), PlanarFormat())
    @test sliver_width(Descriptor(Val(MR), Val(NR), T, RealFormat(), InterleavedFormat()))[2] == 2NR
    @test_throws ArgumentError Descriptor(Val(MR), Val(NR), Float64, PlanarFormat(), PlanarFormat())
    @test_throws ArgumentError Descriptor(Val(0), Val(NR), T, PlanarFormat(), PlanarFormat())
end
