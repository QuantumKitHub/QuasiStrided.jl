# Line-by-line packing (`pack_split`): the planner's decision and the split plans end to end.

include("../helpers.jl")

@testset "line-by-line packing: the planner splits A of intensli_7 past the cache" begin
    plan_of(d) = plan_contract(_sp_views(Float64, SP_I7, ntuple(_ -> d, 6))...)
    p = plan_of(16)
    k, b = p.kernel, p.blocking
    splits(l2bytes) = is_split(pack_split(p.mgroup, p.kgroup, 1, tile_size(k, 1), pack_formats(typeof(k))[1], 8, b.k_block, b.m_block, b.m_block, default_blocking(k).k_block; l2bytes)[2])
    @test splits(2^20) && !splits(2^24)  # a 4 MB reuse window
    host = splits(QuasiStrided.split_capacity(target_profile(), true))
    @test is_split(p.mpack) == host
    host && @test _path_of(p) isa NestPath{false, <:Any, (true, false)}
    @test _path_of(plan_of(4)) isa NestPath{<:Any, <:Any, (false, false)}
end

@testset "line-by-line packing: contractions ($T)" for T in (Float64, Float32, ComplexF64, ComplexF32)
    kernels = T === ComplexF64 ? (nothing, OneMKernel(Val(4), Val(4), T), FMAddSubKernel(Val(8), Val(4), T)) : (nothing,)
    for (ind, ext, m_block) in (
                (SP_I7, (8, 8, 8, 3, 6, 5), nothing), (SP_I7, (6, 8, 11, 5, 10, 7), 48),
                (SP_BOTH, (20, 16, 9, 20, 8), nothing), (SP_BOTH, (13, 12, 5, 11, 12), 64),
            ), kernel in kernels, conjB in (T <: Complex ? (false, true) : (false,)), beta in (0, 0.7)
        T <: Complex && ind === SP_BOTH && continue  # K steps within a page: complex never splits
        Cv, Av, iA, Bv, iB, iC = _sp_views(T, ind, ext)
        iszero(beta) && fill!(Cv, NaN)
        Cref = _lo_reference(iszero(beta) ? zero(Array(Cv)) : Array(Cv), Av, iA, Bv, iB, iC; conjB, alpha = 1.3, beta)
        plan = _sp_forced_plan(Cv, Av, iA, Bv, iB, iC; conjB, m_block, kernel)
        @test _path_of(plan) isa NestPath{false, <:Any, (true, ind === SP_BOTH)}
        execute!(plan, 1.3, beta)
        @test Array(Cv) ≈ Cref rtol = 100 * eps(real(T))
    end
    plan = _sp_forced_plan(_sp_views(T, T <: Complex ? SP_I7 : SP_BOTH, (20, 16, 9, 20, 8, 5))...)
    execute!(plan, 1, 0)
    @test (@allocated execute!(plan, 1, 0)) == 0 skip = (VERSION < v"1.11")
end
