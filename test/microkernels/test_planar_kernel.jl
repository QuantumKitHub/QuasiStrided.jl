include("helpers.jl")

using QuasiStrided: PlanarKernel, lanewidth,
    KERNEL_SHAPES_C64_PLANAR, KERNEL_SHAPES_C32_PLANAR

@testset "PlanarKernel" begin
    full = ((ComplexF64, (16, 6, 8)), (ComplexF32, (8, 5, 8)))
    for (T, (MR, NR, W)) in full
        mk_contract(PlanarKernel(Val(MR), Val(NR), T, Val(W)))
    end
    for (T, menu) in ((ComplexF64, KERNEL_SHAPES_C64_PLANAR), (ComplexF32, KERNEL_SHAPES_C32_PLANAR)), s in menu
        (T, s) in full || mk_contract(PlanarKernel(Val(s[1]), Val(s[2]), T, Val(s[3])); full = false)
    end

    k = PlanarKernel(Val(16), Val(6), ComplexF64, Val(8))
    @test lanewidth(PlanarKernel(Val(8), Val(4), ComplexF64)) == 4  # from the REAL type
    @test lanewidth(PlanarKernel(Val(8), Val(4), ComplexF32)) == 8
    @test_throws ArgumentError PlanarKernel(Val(6), Val(4), ComplexF64, Val(4))
    @test_throws ArgumentError PlanarKernel(Val(8), Val(4), ComplexF64, Val(0))
    @test_throws ArgumentError PlanarKernel(Val(8), Val(4), Float64)

    # The fenced AVX2 tiles keep every broadcast in registers.
    for (T, (MR, NR, W)) in ((ComplexF64, (4, 5, 4)), (ComplexF32, (8, 5, 8)))
        loop = mk_hot_loop(PlanarKernel(Val(MR), Val(NR), T, Val(W)))
        @test count(r"vfn?madd\d+p", loop) == 4 * NR skip = !MK_HAS_FMA
        @test count(r"\[r[sb]p", loop) == 0 skip = !MK_HAS_FMA
    end
end
