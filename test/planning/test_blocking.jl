include("../helpers.jl")

@testset "Blocking: field validation" begin
    b = Blocking(4, 8, 16)
    @test b.m_block == 4 && b.k_block == 8 && b.n_block == 16

    @test_throws ArgumentError Blocking(0, 8, 16)
    @test_throws ArgumentError Blocking(4, 0, 16)
    @test_throws ArgumentError Blocking(4, 8, 0)
    @test_throws ArgumentError Blocking(-1, 8, 16)
end

@testset "default_blocking: dispatches on kernel scalar type" begin
    bf64 = default_blocking(ScalarKernel(Val(8), Val(6), Float64))
    bf32 = default_blocking(ScalarKernel(Val(8), Val(6), Float32))
    @test bf64 isa Blocking
    @test bf32 isa Blocking
    @test bf64.m_block >= 1 && bf64.k_block >= 1 && bf64.n_block >= 1
    @test bf32.m_block >= 1 && bf32.k_block >= 1 && bf32.n_block >= 1
    # Same kernel shape, different scalar type, through SIMDKernel too.
    @test default_blocking(SIMDKernel(Val(8), Val(6), Float64)) == bf64
end

@testset "analytical blocking model" begin
    KiB, MiB = 1024, 1024^2
    # No SMT with an L3 shared by 4 cores; SMT 2 with an L3 shared by 8 cores.
    nosmt = TargetProfile(
        :avx2, "znver2", CacheLevel(32KiB, 64, 1),
        CacheLevel(512KiB, 64, 1), CacheLevel(16MiB, 64, 4)
    )
    smt2 = TargetProfile(
        :avx512, "cascadelake", CacheLevel(32KiB, 64, 2),
        CacheLevel(1MiB, 64, 2), CacheLevel(25952256, 64, 16)
    )
    @test modelled_blocking(nosmt, 8, 6, 8, 8) === Blocking(96, 341, 1728)
    @test modelled_blocking(nosmt, 16, 6, 4, 4) === Blocking(96, 682, 1728)
    @test modelled_blocking(smt2, 16, 6, 8, 8) === Blocking(192, 341, 1572)
    # A complex kernel keeps the real default kernel's `k_block` and blocks M
    # and N by its packed bytes: 1m packs 4 reals per A element, 2 per B element.
    @test default_blocking(kernel_from_shape((12, 8, 8), ComplexF64, OneMKernel), nosmt) === Blocking(24, 341, 864)
    for p in (nosmt, smt2), T in (Float64, Float32, ComplexF64, ComplexF32), K in (SIMDKernel, PlanarKernel, OneMKernel, FMAddSubKernel)
        for shape in kernel_shapes(T, K)
            k = kernel_from_shape(shape, T, K)
            b = default_blocking(k, p)
            MR, NR = tile_size(k)
            @test b.m_block % MR == 0 && b.n_block % NR == 0
            real_default = kernel_from_shape(derived_shape(p, real(T)), real(T), SIMDKernel)
            @test b.k_block == default_blocking(real_default, p).k_block
        end
    end
    # No L3: the B panel is budgeted from the L2 alone.
    nol3 = TargetProfile(
        :neon, "", CacheLevel(64KiB, 64, 1),
        CacheLevel(4MiB, 64, 4), CacheLevel()
    )
    @test modelled_blocking(nol3, 4, 6, 8, 8).n_block == (1MiB ÷ (682 * 8)) ÷ 6 * 6
    tiny = TargetProfile(
        :avx2, "", CacheLevel(64, 64, 1), CacheLevel(64, 64, 1), CacheLevel()
    )
    @test modelled_blocking(tiny, 8, 6, 8, 8) === Blocking(8, 1, 6)
    @test modelled_blocking(unknown_target(), 8, 6, 8, 8) === nothing
    @test modelled_blocking(
        TargetProfile(:avx2, "", CacheLevel(32KiB, 64, 1), CacheLevel(), CacheLevel()),
        8, 6, 8, 8
    ) === nothing
end

@testset "undetected caches: the fallback row scaled by packed reals" begin
    u = unknown_target()
    for T in (ComplexF64, ComplexF32), K in (PlanarKernel, OneMKernel, FMAddSubKernel)
        b = default_blocking(kernel_from_shape(first(kernel_shapes(T, K)), T, K), u)
        base = fallback_blocking(real(T))
        @test b.k_block === base.k_block
        @test b.m_block === base.m_block ÷ reals_per_element(pack_formats(K)[1])
        @test b.n_block === base.n_block ÷ reals_per_element(pack_formats(K)[2])
    end
    @test default_blocking(kernel_from_shape((24, 3, 8), ComplexF64, PlanarKernel), u) === Blocking(64, 256, 384)
end
