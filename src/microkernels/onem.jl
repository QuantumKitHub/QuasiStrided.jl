# 1m (Van Zee's induced method): the real `SIMDKernel` of `2MR x NR`, run over
# `2*k_block_length` real K steps against 1e-packed A and planar B. At logical K
# step `p`, `OneEFormat` A holds `(re_0, im_0, re_1, im_1, ...)` then `(-im_0,
# re_0, -im_1, re_1, ...)`, and planar B `re_0..` then `im_0..`, so accumulator
# real row `2i-1` is the real and `2i` the imaginary part of complex row `i`.
# There is deliberately no FMA loop here: 1m reuses the real kernel body
# verbatim.

inner(::OneMKernel{MR, NR, T, W}) where {MR, NR, T, W} = SIMDKernel(Val(2 * MR), Val(NR), real(T), Val(W))
k_steps(::OneMKernel, k_block_length::Int) = 2 * k_block_length
