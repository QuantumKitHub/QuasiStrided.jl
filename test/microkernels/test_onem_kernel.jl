using QuasiStrided: OneMKernel, PlanarKernel, lanewidth, kernel_shapes,
    kernel_from_shape, default_kernel_type

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

    @testset "blocking and selection" begin
        for T in (ComplexF64, ComplexF32)
            bm = default_blocking(OneMKernel(Val(8), Val(8), T, Val(8)))
            bp = default_blocking(PlanarKernel(Val(8), Val(8), T, Val(8)))
            # Twice the packed A reals, rounded to MR = 8.
            @test (bm.k_block, bm.n_block) == (bp.k_block, bp.n_block) && bp.m_block - 2 * bm.m_block in (0, 8)
            for s in kernel_shapes(T, OneMKernel)
                @test kernel_from_shape(s, T, OneMKernel) isa OneMKernel{s[1], s[2], T, s[3]}
            end
            @test_throws ArgumentError kernel_from_shape((7, 7, 7), T, OneMKernel)
            # Never the default: method ranking does not transfer between machines.
            @test default_kernel_type(T) === PlanarKernel
            @test auto_kernel(T, 1024) isa PlanarKernel
        end
    end

    mk_e2e(ComplexF64, OneMKernel(Val(12), Val(8), ComplexF64, Val(8)))
    mk_e2e(ComplexF32, OneMKernel(Val(24), Val(8), ComplexF32, Val(16)))
end
