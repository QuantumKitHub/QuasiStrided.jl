# Mixed-domain kernels (BLIS's mixed-domain method): a complex operand times a
# real one is a real product once the complex operand is viewed as real, so
# both run the real `SIMDKernel` verbatim, at 2 real FMAs per complex MAC.
#   * complex A x real B: interleaved A is a real `2MR x k_block_length`
#     panel; the accumulator is 1m/fmaddsub's interleaved layout.
#   * real A x complex B: interleaved B is a real `k_block_length x 2NR` panel;
#     accumulator column `2j-1` holds the real and `2j` the imaginary part of
#     column `j`.

inner(::ComplexRealKernel{MR, NR, T, W}) where {MR, NR, T, W} = SIMDKernel(Val(2 * MR), Val(NR), real(T), Val(W))
inner(::RealComplexKernel{MR, NR, T, W}) where {MR, NR, T, W} = SIMDKernel(Val(MR), Val(2 * NR), real(T), Val(W))
