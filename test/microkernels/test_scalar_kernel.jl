include("helpers.jl")

@testset "ScalarKernel" begin
    for (MR, NR, T) in ((4, 3, Float64), (2, 2, Float32))
        mk_contract(ScalarKernel(Val(MR), Val(NR), T))
    end
end
