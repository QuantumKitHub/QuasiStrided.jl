# Uses `mk_store_fastpath` from test_planar_store_fastpath.jl.
using QuasiStrided: FMAddSubKernel, KERNEL_SHAPES_C64_FMADDSUB, KERNEL_SHAPES_C32_FMADDSUB

@testset "fmaddsub store fast path" begin
    mk_store_fastpath(FMAddSubKernel, ComplexF64, KERNEL_SHAPES_C64_FMADDSUB; beta0_exact = true)
    mk_store_fastpath(FMAddSubKernel, ComplexF32, KERNEL_SHAPES_C32_FMADDSUB; beta0_exact = true)
end
