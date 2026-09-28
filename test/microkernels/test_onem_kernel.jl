# Exercises src/microkernels/onem.jl: the 1m (induced) complex microkernel.
#
# The oracles are deliberately *test-local* and independent of the engine:
#   * `onee_pack_a` writes the `OneEFormat` offsets out literally, rather than
#     calling the engine's packer or its `packed_a_plane_offset`;
#   * `planar_pack_b` does the same for `PlanarFormat` B;
#   * `onem_read` re-derives the tile reader from the layout claim, rather than
#     calling `store_tile!`;
#   * `scalar_complex_reference` is a scalar complex dot product in the
#     split-real form `wr += ar*br - ai*bi; wi += ar*bi + ai*br`.
# A kernel validated against the same code that feeds it validates nothing.
#
# Three things get specific attention:
#
#   The ADJACENCY CLAIM the tile reader rests on -- that the two halves of a
#   complex row are always adjacent lanes of one accumulator `Vec` -- is
#   verified structurally for every supported shape (`iseven(W)`,
#   `mod(2MR,W) == 0`, `MR == MV*(W÷2)`), and then pinned numerically with a
#   ramp accumulator whose every real row is distinguishable.
#
#   Cliff B (Julia heap-allocating a dynamically indexed NTuple above NV = 16)
#   by an `@allocated == 0` assertion on a *scattered* fixture, per shape.
#
#   Cliff A (architectural register spill) by `onem_register_pressure`, whose
#   docstring carries the measured `%rsp` traffic. 1m holds `MV*NR`
#   accumulators against planar's `2*MV*NR`, so it should be comfortable -- but
#   derived register arithmetic has proved optimistic for complex kernels, so
#   the numbers there are measured, not derived.

using Test
using Random
using QuasiStrided
using QuasiStrided: OneMKernel, OneMMethod, PlanarKernel, PlanarMethod,
    OneEFormat, PlanarFormat, ComplexKernelDescriptor, SIMDKernel,
    complex_method, realtype, packed_a_per_k, packed_b_per_k,
    a_format, b_format,
    mr, nr, scalartype, packed_a_length, packed_b_length,
    AffineAxis, ScatterAxis, DestinationTile, nrows, ncols,
    zero_accumulator, accumulate, scale_tile!, store_tile!, execute_tile!,
    lanewidth, avecs_per_column, target_profile, default_blocking,
    kernel_shapes, _kernel_from_shape, _default_method
using SIMD: Vec

const _QS = QuasiStrided

@testset "kernels/onem.jl (1m induced complex microkernel)" begin

    # ------------------------------------------------------------------
    # Test-local oracles
    # ------------------------------------------------------------------

    # OneEFormat ("1e"), written out by hand: per logical K step p, A's `MR`
    # complex rows contribute `4*MR` reals as two consecutive real K steps of
    # `2*MR`, holding the real 2x2 block [[re, -im], [im, re]]:
    #
    #     reals   0 .. 2MR-1 :  re_0, im_0, re_1, im_1, ...
    #     reals 2MR .. 4MR-1 : -im_0, re_0, -im_1, re_1, ...
    function onee_pack_a(MR::Int, Amat::AbstractMatrix, kc::Int)
        R = real(eltype(Amat))
        pa = zeros(R, 4 * MR * kc)
        for p in 0:(kc - 1), t in 0:(MR - 1)
            re, im = reim(Amat[t + 1, p + 1])
            base = p * 4 * MR
            pa[base + 2 * t + 1] = re
            pa[base + 2 * t + 2] = im
            pa[base + 2 * MR + 2 * t + 1] = -im
            pa[base + 2 * MR + 2 * t + 2] = re
        end
        return pa
    end

    # PlanarFormat ("1r") B, bit-identical to planar's: per logical K step, NR
    # reals of `re` then NR of `im`.
    function planar_pack_b(NR::Int, Bmat::AbstractMatrix, kc::Int)
        R = real(eltype(Bmat))
        pb = zeros(R, 2 * NR * kc)
        for p in 0:(kc - 1), j in 0:(NR - 1)
            re, im = reim(Bmat[p + 1, j + 1])
            pb[p * (2NR) + j + 1] = re
            pb[p * (2NR) + NR + j + 1] = im
        end
        return pb
    end

    onem_pack(k, Amat, Bmat, kc) =
        (onee_pack_a(mr(k), Amat, kc), planar_pack_b(nr(k), Bmat, kc))

    # The tile reader, re-derived here from the layout claim rather than taken
    # from the kernel: the accumulator is a real `2MR x NR` tile in which real
    # row `2i` is the real part and `2i+1` the imaginary part of complex row
    # `i`, and (W even) those two reals are adjacent lanes `2u+1`, `2u+2` of
    # accumulator vector `v = i ÷ (W÷2)`.
    function onem_read(acc, MR::Int, NR::Int, W::Int, i::Int, j::Int)
        MV = (2 * MR) ÷ W
        HW = W ÷ 2
        v, u = divrem(i, HW)
        vec = acc[v + MV * j + 1]
        return Complex(vec[2 * u + 1], vec[2 * u + 2])
    end

    # Scalar complex dot product, in the split-real form the method implements.
    function scalar_complex_reference(Amat, Bmat, kc::Int, MR::Int, NR::Int)
        T = eltype(Amat)
        R = real(T)
        out = zeros(T, MR, NR)
        for j in 1:NR, i in 1:MR
            wr = zero(R)
            wi = zero(R)
            for p in 1:kc
                ar, ai = reim(Amat[i, p])
                br, bi = reim(Bmat[p, j])
                wr += ar * br - ai * bi
                wi += ar * bi + ai * br
            end
            out[i, j] = Complex(wr, wi)
        end
        return out
    end

    # 1m's shape menus (src/planning/kernel_selection.jl), plus a small shape
    # for cheap exercises. Last entry of each is the AVX2-native rule shape
    # (2026-09-25, job 7107477).
    MENU64 = ((12, 8, 8), (16, 6, 8), (8, 8, 8), (4, 6, 4))
    MENU32 = ((24, 8, 16), (32, 6, 16), (16, 8, 16), (8, 6, 8))
    SMALL = (8, 4, 4)

    ALL64 = (MENU64..., SMALL)
    ALL32 = (MENU32..., SMALL)

    # Same tolerance as the real path at the same precision. 1m needs no
    # widening -- only 3m would, and it is out of scope. If one of these has to
    # be loosened, that is a bug signal, not a property of complex arithmetic.
    tol(::Type{ComplexF64}) = 1.0e-11
    tol(::Type{ComplexF32}) = 2.0f-4

    # ------------------------------------------------------------------
    # Type, construction, forwarding
    # ------------------------------------------------------------------

    @testset "construction and DescriptorKernel forwarding" begin
        k = OneMKernel(Val(12), Val(8), ComplexF64, Val(8))
        @test k isa OneMKernel{12, 8, ComplexF64, 8}
        @test k.descriptor isa
            ComplexKernelDescriptor{12, 8, ComplexF64, OneEFormat, PlanarFormat}
        # The inner kernel is the REAL one, at 2MR rows and the real eltype.
        @test k.inner isa SIMDKernel{24, 8, Float64, 8}
        @test isconcretetype(typeof(k))
        @test isconcretetype(fieldtype(typeof(k), :inner))

        # Shape bookkeeping: mr(k) is LOGICAL, mr(k.inner) is real.
        @test mr(k) == 12
        @test nr(k) == 8
        @test mr(k.inner) == 24
        @test nr(k.inner) == 8
        @test scalartype(k) === ComplexF64
        @test scalartype(k.inner) === Float64
        @test realtype(k) === Float64
        @test lanewidth(k) == 8
        @test avecs_per_column(k) == 3
        @test complex_method(k) === OneMMethod()
        @test a_format(k) === OneEFormat()
        @test b_format(k) === PlanarFormat()

        # Lengths take the LOGICAL kc and return a count of reals.
        @test packed_a_per_k(k) == 4 * 12
        @test packed_b_per_k(k) == 2 * 8
        @test packed_a_length(k, 7) == 4 * 12 * 7
        @test packed_b_length(k, 7) == 2 * 8 * 7

        # Default lane width comes from the REAL type.
        @test lanewidth(OneMKernel(Val(8), Val(4), ComplexF64)) == 4
        @test lanewidth(OneMKernel(Val(8), Val(4), ComplexF32)) == 8

        # MR = 12 at W = 8 is legal precisely because 2*12 = 24 divides by 8;
        # MR = 12 at W = 16 is not, because 24 does not.
        @test OneMKernel(Val(12), Val(8), ComplexF64, Val(8)) isa OneMKernel
        @test_throws ArgumentError OneMKernel(Val(12), Val(8), ComplexF64, Val(16))
        @test_throws ArgumentError OneMKernel(Val(8), Val(4), ComplexF64, Val(0))
        @test_throws ArgumentError OneMKernel(Val(8), Val(4), Float64)
        @test_throws ArgumentError OneMKernel(Val(8), Val(4), Float64, Val(4))
        # Odd W would break the lane-pair reader and is rejected outright,
        # even where `mod(2MR, W) == 0` would otherwise permit it (2*3 = 6).
        @test_throws ArgumentError OneMKernel(Val(3), Val(4), ComplexF64, Val(3))

        # A complex kernel must never be asked for a single-plane offset.
        @test_throws MethodError _QS.packed_a_offset(k, 0, 0)
        # The plane accessors forward to the descriptor: p*4MR + plane*MR + i.
        @test _QS.packed_a_plane_offset(k, 0, 5, 2) == 2 * 48 + 5
        @test _QS.packed_a_plane_offset(k, 2, 5, 2) == 2 * 48 + 24 + 5
        @test _QS.packed_b_plane_offset(k, 1, 3, 2) == 2 * 16 + 8 + 3

        # The inner-kernel type is pinned: nothing else can be stored there.
        wrong = SIMDKernel(Val(12), Val(8), Float64, Val(4))
        d = ComplexKernelDescriptor(Val(12), Val(8), ComplexF64, OneEFormat(), PlanarFormat())
        @test_throws ArgumentError OneMKernel{12, 8, ComplexF64, 8, typeof(wrong)}(d, wrong)
    end

    @testset "zero_accumulator is the inner REAL kernel's" begin
        for (T, menu) in ((ComplexF64, ALL64), (ComplexF32, ALL32))
            R = real(T)
            for (MR, NR, W) in menu
                k = OneMKernel(Val(MR), Val(NR), T, Val(W))
                acc = zero_accumulator(k)
                @test acc isa NTuple{((2 * MR) ÷ W) * NR, Vec{W, R}}
                @test acc === zero_accumulator(k.inner)
                @test all(v -> all(iszero, Tuple(v)), acc)
                # One accumulator plane over 2MR real rows, not two over MR:
                # exactly the real kernel's count at the doubled row extent.
                @test length(acc) == length(zero_accumulator(k.inner))
            end
        end
    end

    # ------------------------------------------------------------------
    # The adjacency claim the tile reader rests on
    # ------------------------------------------------------------------

    @testset "adjacency: a complex row is always two adjacent lanes of one Vec" begin
        for (T, menu) in ((ComplexF64, ALL64), (ComplexF32, ALL32))
            R = real(T)
            for (MR, NR, W) in menu
                # Structural preconditions, checked rather than assumed.
                @test iseven(W)                      # lane pairs exist
                @test mod(2 * MR, W) == 0            # vectors tile 2MR exactly
                MV = (2 * MR) ÷ W
                HW = W ÷ 2
                @test MR == MV * HW                  # complex rows partition cleanly
                # Every complex row's two real rows land in the SAME vector, at
                # adjacent lanes, for every row of every shape.
                for i in 0:(MR - 1)
                    v_re, lane_re = divrem(2 * i, W)
                    v_im, lane_im = divrem(2 * i + 1, W)
                    @test v_re == v_im               # same vector
                    @test lane_im == lane_re + 1     # adjacent lanes
                    @test iseven(lane_re)            # so 1-based lanes are (2u+1, 2u+2)
                    @test v_re == i ÷ HW             # matches the reader's v
                    @test lane_re == 2 * (i % HW)    # matches the reader's u
                end

                # Numerically pinned with a ramp accumulator: real row `r` of
                # column `j` carries the distinguishable value `1000j + r`, so
                # any mis-assignment of a lane to a row is visible.
                k = OneMKernel(Val(MR), Val(NR), T, Val(W))
                acc = ntuple(MV * NR) do idx
                    v, j = (idx - 1) % MV, (idx - 1) ÷ MV
                    Vec{W, R}(ntuple(l -> R(1000 * j + v * W + l - 1), W))
                end
                for j in 0:(NR - 1), i in 0:(MR - 1)
                    want = Complex(R(1000 * j + 2 * i), R(1000 * j + 2 * i + 1))
                    @test onem_read(acc, MR, NR, W, i, j) == want
                end
                # ... and the kernel's own reader agrees, via store_tile!.
                storage = zeros(T, MR * NR)
                dst = DestinationTile(storage, 0, AffineAxis(0, 1, MR), AffineAxis(0, MR, NR))
                store_tile!(dst, acc, one(T), zero(T), k)
                for j in 0:(NR - 1), i in 0:(MR - 1)
                    @test storage[i + j * MR + 1] ==
                        Complex(R(1000 * j + 2 * i), R(1000 * j + 2 * i + 1))
                end
            end
        end
    end

    # ------------------------------------------------------------------
    # Cliff A: architectural register pressure
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Numerical agreement: scalar oracle, and PlanarKernel
    # ------------------------------------------------------------------

    @testset "accumulate vs scalar complex dot product" begin
        for (T, menu) in ((ComplexF64, ALL64), (ComplexF32, ALL32))
            R = real(T)
            for (MR, NR, W) in menu
                k = OneMKernel(Val(MR), Val(NR), T, Val(W))
                rng = MersenneTwister(MR * 1000 + NR * 10 + W)
                for kc in (1, 5, 17)
                    Amat = rand(rng, T, MR, kc)
                    Bmat = rand(rng, T, kc, NR)
                    pa, pb = onem_pack(k, Amat, Bmat, kc)
                    @test length(pa) == packed_a_length(k, kc)
                    @test length(pb) == packed_b_length(k, kc)

                    acc = accumulate(k, zero_accumulator(k), pa, pb, kc)
                    expected = scalar_complex_reference(Amat, Bmat, kc, MR, NR)

                    maxerr = zero(R)
                    for j in 0:(NR - 1), i in 0:(MR - 1)
                        got = onem_read(acc, MR, NR, W, i, j)
                        maxerr = max(maxerr, abs(got - expected[i + 1, j + 1]))
                    end
                    @test maxerr <= tol(T)
                end
            end
        end
    end

    @testset "agrees with PlanarKernel on the same fixtures" begin
        # Wherever planar is constructible at the same (MR, NR, W) -- i.e.
        # `mod(MR, W) == 0`, which excludes 1m's deliberately "unaligned"
        # MR = 12 / MR = 24 entries -- the two methods must agree to the same
        # tolerance as each does against the scalar oracle.
        for (T, menu) in ((ComplexF64, ALL64), (ComplexF32, ALL32))
            R = real(T)
            for (MR, NR, W) in menu
                mod(MR, W) == 0 || continue
                km = OneMKernel(Val(MR), Val(NR), T, Val(W))
                kp = PlanarKernel(Val(MR), Val(NR), T, Val(W))
                rng = MersenneTwister(MR * 7 + NR * 3 + W)
                for kc in (1, 6, 19)
                    Amat = rand(rng, T, MR, kc) .- T(0.5, 0.5)
                    Bmat = rand(rng, T, kc, NR) .- T(0.5, 0.5)

                    stor_m = zeros(T, MR * NR)
                    dst_m = DestinationTile(
                        stor_m, 0, AffineAxis(0, 1, MR), AffineAxis(0, MR, NR)
                    )
                    pa_m, pb_m = onem_pack(km, Amat, Bmat, kc)
                    execute_tile!(km, dst_m, pa_m, pb_m, kc, one(T), zero(T))

                    # Planar's own packed A, written out locally too: 1m's B is
                    # bit-identical to planar's (their D14), so the SAME `pb`
                    # feeds both -- that is 1m's whole marginal cost being on
                    # the A side, asserted rather than asserted-about.
                    pa_p = zeros(R, 2 * MR * kc)
                    for p in 0:(kc - 1), i in 0:(MR - 1)
                        re, im = reim(Amat[i + 1, p + 1])
                        pa_p[p * (2MR) + i + 1] = re
                        pa_p[p * (2MR) + MR + i + 1] = im
                    end
                    stor_p = zeros(T, MR * NR)
                    dst_p = DestinationTile(
                        stor_p, 0, AffineAxis(0, 1, MR), AffineAxis(0, MR, NR)
                    )
                    execute_tile!(kp, dst_p, pa_p, pb_m, kc, one(T), zero(T))

                    @test maximum(abs.(stor_m .- stor_p)) <= tol(T)
                end
            end
        end
    end

    @testset "accumulate: kc == 0 is a no-op and reads nothing" begin
        k = OneMKernel(Val(8), Val(4), ComplexF64, Val(4))
        acc0 = zero_accumulator(k)
        @test accumulate(k, acc0, Float64[], Float64[], 0) === acc0
        # The error reports the LOGICAL kc the caller passed, not the doubled
        # real one: the `2*kc` must not leak into a diagnostic either.
        err = try
            accumulate(k, acc0, Float64[], Float64[], -3)
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("kc = -3", err.msg)
    end

    @testset "accumulate: composes additively across split kc" begin
        MR, NR, W, kc = 8, 4, 4, 4
        k = OneMKernel(Val(MR), Val(NR), ComplexF64, Val(W))
        rng = MersenneTwister(77)
        Amat = rand(rng, ComplexF64, MR, kc)
        Bmat = rand(rng, ComplexF64, kc, NR)
        pa, pb = onem_pack(k, Amat, Bmat, kc)

        acc_all = accumulate(k, zero_accumulator(k), pa, pb, kc)
        acc_split = zero_accumulator(k)
        for p in 0:(kc - 1)
            acc_split = accumulate(
                k, acc_split,
                view(pa, (p * 4MR + 1):((p + 1) * 4MR)),
                view(pb, (p * 2NR + 1):((p + 1) * 2NR)), 1
            )
        end
        for idx in 1:length(acc_all)
            @test all(Tuple(acc_all[idx]) .≈ Tuple(acc_split[idx]))
        end
    end

    # ------------------------------------------------------------------
    # execute_tile! end to end
    # ------------------------------------------------------------------

    function colmajor_tile(T, m, n, pad)
        storage = fill(T(NaN, NaN), m * n + 2pad)
        return storage, DestinationTile(storage, pad, AffineAxis(0, 1, m), AffineAxis(0, m, n))
    end

    @testset "execute_tile!: full tile vs A*B, both precisions, whole menu" begin
        for (T, menu) in ((ComplexF64, ALL64), (ComplexF32, MENU32))
            for (MR, NR, W) in menu
                k = OneMKernel(Val(MR), Val(NR), T, Val(W))
                rng = MersenneTwister(MR + NR + W)
                kc = 9
                Amat = rand(rng, T, MR, kc)
                Bmat = rand(rng, T, kc, NR)
                pa, pb = onem_pack(k, Amat, Bmat, kc)
                expected = scalar_complex_reference(Amat, Bmat, kc, MR, NR)

                pad = 3
                storage, dst = colmajor_tile(T, MR, NR, pad)
                execute_tile!(k, dst, pa, pb, kc, one(T), zero(T))
                for i in 0:(MR - 1), j in 0:(NR - 1)
                    @test abs(storage[pad + i + j * MR + 1] - expected[i + 1, j + 1]) <= tol(T)
                end
                @test all(isnan ∘ real, storage[1:pad])
                @test all(isnan ∘ real, storage[(end - pad + 1):end])
            end
        end
    end

    @testset "execute_tile!: alpha/beta contract, partial tiles, kc == 0" begin
        for (T, (MR, NR, W)) in ((ComplexF64, (12, 8, 8)), (ComplexF32, (24, 8, 16)))
            k = OneMKernel(Val(MR), Val(NR), T, Val(W))
            rng = MersenneTwister(5)

            for kc in (0, 1, 4, 13),
                    (alpha, beta) in (
                        (one(T), zero(T)), (T(2.5, -1.0), T(-1.75, 0.5)),
                        (one(T), one(T)), (zero(T), T(3.0, 1.0)), (one(T), T(2, 0)),
                    ),
                    (m, n) in ((MR, NR), (1, 1), (5, 4), (MR, 2), (3, NR), (0, 0))

                Amat = rand(rng, T, MR, max(kc, 1))
                Bmat = rand(rng, T, max(kc, 1), NR)
                pa, pb = onem_pack(k, Amat, Bmat, max(kc, 1))
                ref = kc == 0 ? zeros(T, MR, NR) :
                    scalar_complex_reference(Amat, Bmat, kc, MR, NR)

                Cold = rand(rng, T, max(m, 1), max(n, 1))
                storage = vec(copy(Cold))[1:(m * n)]
                dst = DestinationTile(storage, 0, AffineAxis(0, 1, m), AffineAxis(0, m, n))
                execute_tile!(k, dst, pa, pb, kc, alpha, beta)

                for i in 0:(m - 1), j in 0:(n - 1)
                    r = (kc == 0 || iszero(alpha)) ? zero(T) : ref[i + 1, j + 1]
                    want = (kc == 0 || iszero(alpha)) ? beta * Cold[i + 1, j + 1] :
                        alpha * r + beta * Cold[i + 1, j + 1]
                    @test abs(storage[i + j * m + 1] - want) <= tol(T) * max(1, abs(want))
                end
            end
        end
    end

    @testset "execute_tile!: validation and short-circuits" begin
        MR, NR, W = 8, 4, 4
        T = ComplexF64
        k = OneMKernel(Val(MR), Val(NR), T, Val(W))
        storage = zeros(T, MR * NR)
        dst = DestinationTile(storage, 0, AffineAxis(0, 1, MR), AffineAxis(0, MR, NR))

        @test_throws ArgumentError execute_tile!(k, dst, Float64[], Float64[], -1, one(T), zero(T))
        too_tall = DestinationTile(storage, 0, AffineAxis(0, 1, MR + 1), AffineAxis(0, MR, NR))
        @test_throws ArgumentError execute_tile!(k, too_tall, Float64[], Float64[], 1, one(T), zero(T))
        too_wide = DestinationTile(storage, 0, AffineAxis(0, 1, MR), AffineAxis(0, MR, NR + 1))
        @test_throws ArgumentError execute_tile!(k, too_wide, Float64[], Float64[], 1, one(T), zero(T))

        # kc == 0 and alpha == 0 never read the panels (empty buffers suffice).
        fill!(storage, T(2, 3))
        execute_tile!(k, dst, Float64[], Float64[], 0, one(T), T(2, 0))
        @test all(==(T(4, 6)), storage)
        fill!(storage, T(2, 3))
        execute_tile!(k, dst, Float64[], Float64[], 5, zero(T), zero(T))
        @test all(iszero, storage)

        # Undersized panels are a DimensionMismatch, checked before any read.
        # The needed A length is 4*MR*kc (twice planar's); B's is 2*NR*kc.
        @test_throws DimensionMismatch execute_tile!(
            k, dst, zeros(Float64, 4MR * 3 - 1), zeros(Float64, 2NR * 3), 3, one(T), zero(T)
        )
        @test_throws DimensionMismatch execute_tile!(
            k, dst, zeros(Float64, 4MR * 3), zeros(Float64, 2NR * 3 - 1), 3, one(T), zero(T)
        )
    end

    @testset "scattered and negative-stride axes agree with the regular path" begin
        MR, NR, W = 12, 8, 8
        T = ComplexF64
        k = OneMKernel(Val(MR), Val(NR), T, Val(W))
        rng = MersenneTwister(21)
        m, n, kc = 11, 5, 6
        Amat = rand(rng, T, MR, kc)
        Bmat = rand(rng, T, kc, NR)
        pa, pb = onem_pack(k, Amat, Bmat, kc)

        ref_storage = zeros(T, m * n)
        ref = DestinationTile(ref_storage, 0, AffineAxis(0, 1, m), AffineAxis(0, m, n))
        execute_tile!(k, ref, pa, pb, kc, one(T), zero(T))

        sc_storage = zeros(T, m * n)
        sc = DestinationTile(
            sc_storage, 0, ScatterAxis(collect(0:(m - 1)), m), AffineAxis(0, m, n)
        )
        execute_tile!(k, sc, pa, pb, kc, one(T), zero(T))
        @test sc_storage == ref_storage

        perm = [0, 3, 1, 7, 2, 9, 4, 10, 5, 6, 8]
        ir_storage = zeros(T, m * n)
        ir = DestinationTile(ir_storage, 0, ScatterAxis(perm, m), AffineAxis(0, m, n))
        execute_tile!(k, ir, pa, pb, kc, one(T), zero(T))
        for i in 0:(m - 1), j in 0:(n - 1)
            @test ir_storage[perm[i + 1] + m * j + 1] == ref_storage[i + m * j + 1]
        end

        neg_storage = zeros(T, m * n)
        neg = DestinationTile(
            neg_storage, (m - 1) + m * (n - 1), AffineAxis(0, -1, m), AffineAxis(0, -m, n)
        )
        execute_tile!(k, neg, pa, pb, kc, one(T), zero(T))
        for i in 0:(m - 1), j in 0:(n - 1)
            @test neg_storage[(m - 1 - i) + m * (n - 1 - j) + 1] == ref_storage[i + m * j + 1]
        end
    end

    @testset "padded-lane isolation: nonfinite padding never propagates" begin
        MR, NR, W = 12, 8, 8
        T = ComplexF64
        k = OneMKernel(Val(MR), Val(NR), T, Val(W))
        m, n, kc = 3, 2, 1

        # Poison every padding lane of both panels, in the packed 1e / 1r
        # layouts, leaving the valid lanes finite.
        pa = zeros(Float64, packed_a_length(k, kc))
        pb = zeros(Float64, packed_b_length(k, kc))
        for t in 0:(MR - 1)
            re = t < m ? Float64(t + 2) : Inf
            im = t < m ? Float64(t + 1) : Inf
            pa[2 * t + 1] = re
            pa[2 * t + 2] = im
            pa[2 * MR + 2 * t + 1] = -im
            pa[2 * MR + 2 * t + 2] = re
        end
        for j in 0:(NR - 1)
            pb[j + 1] = j < n ? Float64(7 + j) : Inf
            pb[NR + j + 1] = j < n ? Float64(2 + j) : Inf
        end

        acc = accumulate(k, zero_accumulator(k), pa, pb, kc)
        any_nonfinite = false
        for j in 0:(NR - 1), i in 0:(MR - 1)
            z = onem_read(acc, MR, NR, W, i, j)
            (i >= m || j >= n) && !isfinite(z) && (any_nonfinite = true)
        end
        @test any_nonfinite  # confirm the hazard is real

        storage = fill(T(1, 1), m * n)
        dst = DestinationTile(storage, 0, AffineAxis(0, 1, m), AffineAxis(0, m, n))
        store_tile!(dst, acc, one(T), zero(T), k)
        @test all(isfinite, storage)

        sc_storage = fill(T(1, 1), m * n)
        sc = DestinationTile(sc_storage, 0, ScatterAxis(collect(0:(m - 1)), m), AffineAxis(0, m, n))
        store_tile!(sc, acc, one(T), zero(T), k)
        @test all(isfinite, sc_storage)
        @test sc_storage == storage

        # beta == 0 never reads old C.
        poison = fill(T(Inf, NaN), m * n)
        pd = DestinationTile(poison, 0, AffineAxis(0, 1, m), AffineAxis(0, m, n))
        store_tile!(pd, acc, one(T), zero(T), k)
        @test all(isfinite, poison)

        # alpha == 0 never reads acc.
        NV = ((2 * MR) ÷ W) * NR
        nanacc = ntuple(_ -> Vec{W, Float64}(NaN), Val(NV))
        target = fill(T(2, 3), m * n)
        td = DestinationTile(target, 0, AffineAxis(0, 1, m), AffineAxis(0, m, n))
        store_tile!(td, nanacc, zero(T), T(2, 0), k)
        @test all(==(T(4, 6)), target)

        # Empty destination is a no-op in every branch.
        empty_storage = T[]
        ed = DestinationTile(empty_storage, 0, AffineAxis(0, 1, 0), AffineAxis(0, 0, 0))
        @test store_tile!(ed, acc, one(T), zero(T), k) === ed
        @test execute_tile!(k, ed, pa, pb, kc, one(T), zero(T)) === ed
    end

    # ------------------------------------------------------------------
    # Cliff B: allocations, measured on a SCATTERED fixture
    # ------------------------------------------------------------------

    @testset "Cliff B: zero allocations on a scattered fixture, every shape" begin
        run_accumulate(k, pa, pb, kc) = accumulate(k, zero_accumulator(k), pa, pb, kc)
        function run_execute(k, dst, pa, pb, kc, alpha, beta)
            execute_tile!(k, dst, pa, pb, kc, alpha, beta)
            return nothing
        end

        for (T, menu) in ((ComplexF64, ALL64), (ComplexF32, MENU32))
            for (MR, NR, W) in menu
                k = OneMKernel(Val(MR), Val(NR), T, Val(W))
                rng = MersenneTwister(MR * 31 + NR)
                kc = 8
                Amat = rand(rng, T, MR, kc)
                Bmat = rand(rng, T, kc, NR)
                pa, pb = onem_pack(k, Amat, Bmat, kc)

                offs = collect(0:(MR - 1))
                storage = zeros(T, MR * NR)
                dst = DestinationTile(
                    storage, 0, ScatterAxis(offs, MR), AffineAxis(0, MR, NR)
                )

                alpha, beta = T(2, -1), T(0.5, 0.25)
                run_accumulate(k, pa, pb, kc)                      # warm up
                run_execute(k, dst, pa, pb, kc, alpha, beta)       # warm up

                bytes_acc = @allocated run_accumulate(k, pa, pb, kc)
                bytes_exec = @allocated run_execute(k, dst, pa, pb, kc, alpha, beta)

                # Julia 1.10 (LTS) cannot keep an NTuple{NV,Vec} accumulator
                # register-resident; a compiler capability gap, kept visible as
                # a skip rather than hidden by a weaker assertion (on 1.10 this
                # fixture allocates ~0.5-1.6 KB per accumulate; execute_tile!
                # stays at 0 B).
                @test bytes_acc == 0 skip = (VERSION < v"1.11")
                @test bytes_exec == 0 skip = (VERSION < v"1.11")
                if VERSION >= v"1.11" && (bytes_acc != 0 || bytes_exec != 0)
                    @info "1m allocation" T MR NR W bytes_acc bytes_exec
                end
            end
        end
    end

    # ------------------------------------------------------------------
    # Packed-length ratios and derived blocking (their D14 / D13)
    # ------------------------------------------------------------------

    @testset "packed lengths: A is exactly twice planar's, B exactly equal" begin
        for (T, menu) in ((ComplexF64, ALL64), (ComplexF32, ALL32))
            for (MR, NR, W) in menu
                mod(MR, W) == 0 || continue
                km = OneMKernel(Val(MR), Val(NR), T, Val(W))
                kp = PlanarKernel(Val(MR), Val(NR), T, Val(W))
                @test packed_a_per_k(km) == 2 * packed_a_per_k(kp)
                @test packed_b_per_k(km) == packed_b_per_k(kp)
                for kc in (0, 1, 7, 256)
                    @test packed_a_length(km, kc) == 2 * packed_a_length(kp, kc)
                    @test packed_b_length(km, kc) == packed_b_length(kp, kc)
                    # The length arithmetic coincides exactly with running a
                    # real kernel of 2MR rows over 2kc real steps.
                    @test packed_a_length(km, kc) == (2 * MR) * (2 * kc)
                    @test packed_b_length(km, kc) == NR * (2 * kc)
                end
            end
        end
    end

    @testset "default_blocking gives 1m half planar's mc, same kc and nc" begin
        for T in (ComplexF64, ComplexF32)
            bm = default_blocking(OneMKernel(Val(8), Val(8), T, Val(8)))
            bp = default_blocking(PlanarKernel(Val(8), Val(8), T, Val(8)))
            @test bm.mc == bp.mc ÷ 2
            @test bm.kc == bp.kc
            @test bm.nc == bp.nc      # b_reals is 2 for both methods
            # Derived from the measured real row, never tabulated.
            br = default_blocking(SIMDKernel(Val(8), Val(6), real(T), Val(4)))
            @test bm.mc == max(1, br.mc ÷ 4)
            @test bp.mc == max(1, br.mc ÷ 2)
        end
    end

    # ------------------------------------------------------------------
    # Driver wiring: selectable ONLY by naming the kernel
    # ------------------------------------------------------------------

    @testset "_kernel_from_shape: the OneMMethod arm, and no auto-dispatch" begin
        for (T, menu) in ((ComplexF64, MENU64), (ComplexF32, MENU32))
            @test kernel_shapes(T, OneMMethod()) === menu
            for (MR, NR, W) in menu
                k = _kernel_from_shape((MR, NR, W), T, OneMMethod())
                @test k isa OneMKernel{MR, NR, T, W}
                @test complex_method(k) === OneMMethod()
            end
            # An off-menu shape throws, exactly as the planar arm does -- it
            # never silently builds a different shape or another method.
            @test_throws ArgumentError _kernel_from_shape((7, 7, 7), T, OneMMethod())
        end

        # 1m is NOT the default and no rule may make it one: method ranking
        # does not transfer between machines.
        @test _default_method(ComplexF64) === PlanarMethod()
        @test _default_method(ComplexF32) === PlanarMethod()
        for T in (ComplexF64, ComplexF32)
            @test complex_method(_QS._default_kernel(T, 1024, 1024)) === PlanarMethod()
            @test _QS._default_kernel(T, 1024, 1024) isa PlanarKernel
        end
    end

    @testset "end to end: a named OneMKernel plan matches planar and the oracle" begin
        for (T, (MR, NR, W)) in ((ComplexF64, (12, 8, 8)), (ComplexF32, (24, 8, 16)))
            rng = MersenneTwister(4242)
            # Extents chosen to straddle the register tile in both directions
            # and to exceed one kc panel, so multi-panel beta handling runs.
            M, N, K = 37, 23, 41
            Amat = rand(rng, T, M, K) .- T(0.5, 0.5)
            Bmat = rand(rng, T, K, N) .- T(0.5, 0.5)
            reference = Amat * Bmat

            km = OneMKernel(Val(MR), Val(NR), T, Val(W))
            alpha, beta = T(1.5, -0.25), T(-0.75, 0.5)

            Cinit = rand(rng, T, M, N)

            function run(kern, kc)
                C = copy(Cinit)
                plan = QuasiStrided.plan_contract(
                    _QS.StridedView(C), _QS.StridedView(Amat), (1, 2),
                    _QS.StridedView(Bmat), (2, 3), (1, 3);
                    kernel = kern, kc = kc
                )
                QuasiStrided.execute!(plan, alpha, beta)
                return C, plan
            end

            want = alpha .* reference .+ beta .* Cinit

            C1m, plan1m = run(km, 16)   # kc = 16 < K, so several K panels
            @test maximum(abs.(C1m .- want)) <= tol(T) * 64

            # The same plan through the independent macro-nest oracle.
            Ctw = copy(Cinit)
            plantw = QuasiStrided.plan_contract(
                _QS.StridedView(Ctw), _QS.StridedView(Amat), (1, 2),
                _QS.StridedView(Bmat), (2, 3), (1, 3);
                kernel = OneMKernel(Val(MR), Val(NR), T, Val(W)), kc = 16
            )
            QuasiStrided.execute_tilewise!(plantw, alpha, beta)
            @test maximum(abs.(Ctw .- C1m)) <= tol(T) * 64

            # And against the host's planar default kernel, named explicitly:
            # at M = 37 the extent demotion would pick FMAddSub for
            # ComplexF32 on AVX-512 (`_small_m_shape`).
            Cpl = copy(Cinit)
            planpl = QuasiStrided.plan_contract(
                _QS.StridedView(Cpl), _QS.StridedView(Amat), (1, 2),
                _QS.StridedView(Bmat), (2, 3), (1, 3);
                kernel = _QS._kernel_for(_QS.target_profile(), T)
            )
            @test planpl.kernel isa PlanarKernel
            QuasiStrided.execute!(planpl, alpha, beta)
            @test maximum(abs.(Cpl .- C1m)) <= tol(T) * 64

            # Conjugation goes through the existing pack-time transform seam,
            # which 1m inherits unchanged; pinned against the oracle too.
            for (cA, cB) in ((true, false), (false, true), (true, true))
                Cc = copy(Cinit)
                planc = QuasiStrided.plan_contract(
                    _QS.StridedView(Cc), _QS.StridedView(Amat), (1, 2),
                    _QS.StridedView(Bmat), (2, 3), (1, 3);
                    kernel = OneMKernel(Val(MR), Val(NR), T, Val(W)),
                    conjA = cA, conjB = cB
                )
                QuasiStrided.execute!(planc, alpha, beta)
                wantc = alpha .* ((cA ? conj.(Amat) : Amat) * (cB ? conj.(Bmat) : Bmat)) .+
                    beta .* Cinit
                @test maximum(abs.(Cc .- wantc)) <= tol(T) * 64

                Co = copy(Cinit)
                plano = QuasiStrided.plan_contract(
                    _QS.StridedView(Co), _QS.StridedView(Amat), (1, 2),
                    _QS.StridedView(Bmat), (2, 3), (1, 3);
                    kernel = OneMKernel(Val(MR), Val(NR), T, Val(W)),
                    conjA = cA, conjB = cB
                )
                QuasiStrided.execute_tilewise!(plano, alpha, beta)
                @test maximum(abs.(Co .- Cc)) <= tol(T) * 64
            end
        end
    end

end
