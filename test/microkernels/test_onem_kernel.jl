include("helpers.jl")

using QuasiStrided: OneMKernel, lanewidth, kernel_shapes

@testset "OneMKernel" begin
    full = ((ComplexF64, (12, 8, 8)), (ComplexF32, (8, 6, 8)))
    for (T, (MR, NR, W)) in full
        mk_contract(OneMKernel(Val(MR), Val(NR), T, Val(W)))
    end
    for T in (ComplexF64, ComplexF32), s in (kernel_shapes(T, OneMKernel)..., (8, 4, 4))
        (T, s) in full || mk_contract(OneMKernel(Val(s[1]), Val(s[2]), T, Val(s[3])); full = false)
    end

    @testset "construction" begin
        k = OneMKernel(Val(12), Val(8), ComplexF64, Val(8))
        @test QuasiStrided.inner(k) isa SIMDKernel{24, 8, Float64, 8}  # the real kernel, at 2MR rows
        @test zero_accumulator(k) === zero_accumulator(QuasiStrided.inner(k))
        @test lanewidth(OneMKernel(Val(8), Val(4), ComplexF32)) == 8
        # 2MR, not MR, must divide by W; W must be even even where 2MR divides.
        @test_throws ArgumentError OneMKernel(Val(12), Val(8), ComplexF64, Val(16))
        @test_throws ArgumentError OneMKernel(Val(3), Val(4), ComplexF64, Val(3))
        @test_throws ArgumentError OneMKernel(Val(8), Val(4), ComplexF64, Val(0))
        @test_throws ArgumentError OneMKernel(Val(8), Val(4), Float64)
        @test_throws ArgumentError OneMKernel(Val(8), Val(4), Float64, Val(4))
        # The error reports the logical k_block_length, not the doubled real one.
        err = try
            add_tile(k, zero_accumulator(k), Float64[], Float64[], -3)
        catch e
            e
        end
        @test err isa ArgumentError && occursin("k_block_length = -3", err.msg)
    end
end
