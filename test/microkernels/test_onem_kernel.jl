using QuasiStrided: OneMKernel, OneMMethod, PlanarKernel, PlanarMethod,
    KernelMethod, lanewidth, kernel_shapes,
    kernel_from_shape, default_method

@testset "OneMKernel" begin
    full = ((ComplexF64, (12, 8, 8)), (ComplexF32, (8, 6, 8)))
    for (T, (MR, NR, W)) in full
        mk_contract(OneMKernel(Val(MR), Val(NR), T, Val(W)))
    end
    for T in (ComplexF64, ComplexF32), s in (kernel_shapes(T, OneMMethod())..., (8, 4, 4))
        (T, s) in full || mk_contract(OneMKernel(Val(s[1]), Val(s[2]), T, Val(s[3])); full = false)
    end

    @testset "construction" begin
        k = OneMKernel(Val(12), Val(8), ComplexF64, Val(8))
        @test QuasiStrided.inner(k) isa SIMDKernel{24, 8, Float64, 8}  # the real kernel, at 2MR rows
        @test zero_accumulator(k) === zero_accumulator(QuasiStrided.inner(k))
        @test KernelMethod(k) === OneMMethod()
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
            @test (bm.m_block, bm.k_block, bm.n_block) == (bp.m_block ÷ 2, bp.k_block, bp.n_block)  # twice the packed A reals
            for s in kernel_shapes(T, OneMMethod())
                @test kernel_from_shape(s, T, OneMMethod()) isa OneMKernel{s[1], s[2], T, s[3]}
            end
            @test_throws ArgumentError kernel_from_shape((7, 7, 7), T, OneMMethod())
            # Never the default: method ranking does not transfer between machines.
            @test default_method(T) === PlanarMethod()
            @test QuasiStrided._default_kernel(T, 1024, 1024) isa PlanarKernel
        end
    end

    mk_e2e(ComplexF64, OneMKernel(Val(12), Val(8), ComplexF64, Val(8)))
    mk_e2e(ComplexF32, OneMKernel(Val(24), Val(8), ComplexF32, Val(16)))
end
