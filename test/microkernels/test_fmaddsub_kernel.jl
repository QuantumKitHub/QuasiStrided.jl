# Exercises src/microkernels/fmaddsub.jl: the interleaved-accumulator complex
# microkernel built on x86 `vfmaddsub`, and its `InterleavedFormat` A panel.
#
# Same discipline as test_onem_kernel.jl: every oracle here is TEST-LOCAL and
# independent of the engine --
#   * `interleaved_pack_a` / `planar_pack_b` write the packed offsets out
#     literally rather than calling the engine's packers or offset helpers;
#   * `ilv_read` re-derives the tile reader from the layout claim;
#   * `scalar_complex_reference` is a scalar complex dot product;
#   * the two lane primitives are checked lane by lane against Base's scalar
#     `fma`, which is the exact per-lane semantics claimed for them.
# End to end, a named FMAddSubKernel plan is compared against the tile-by-tile
# oracle (`execute_tilewise!`) run with the REFERENCE planar kernel, against
# its own oracle run, and against `A*B` -- always with a tolerance: the
# accumulation order differs from planar's in the real part, so bit equality
# would be asserting a rounding coincidence.

using Test
using Random
using QuasiStrided
using QuasiStrided: FMAddSubKernel, FMAddSubMethod, OneMKernel, OneMMethod,
    PlanarKernel, PlanarMethod, InterleavedFormat, PlanarFormat, OneEFormat,
    ComplexKernelDescriptor, complex_method, realtype, packed_a_per_k, packed_b_per_k,
    a_format, b_format,
    mr, nr, scalartype, packed_a_length, packed_b_length, reals_per_element,
    AffineAxis, ScatterAxis, DestinationTile, SourceTile, pack_a!, pack_b!,
    packed_panel, zero_accumulator, accumulate, store_tile!, execute_tile!,
    lanewidth, avecs_per_column, target_profile, default_blocking,
    kernel_shapes, _kernel_from_shape, _default_method, a_reals, b_reals,
    accumulator_planes, PackedPanel
using SIMD: Vec
using InteractiveUtils: code_native

const _QSF = QuasiStrided

@testset "kernels/fmaddsub.jl (interleaved fmaddsub complex microkernel)" begin

    # ------------------------------------------------------------------
    # Test-local oracles
    # ------------------------------------------------------------------

    # InterleavedFormat, by hand: per logical K step p, 2MR reals
    # `re_0, im_0, re_1, im_1, ...` -- Complex{T}'s own memory order.
    function interleaved_pack_a(MR::Int, Amat::AbstractMatrix, kc::Int)
        R = real(eltype(Amat))
        pa = zeros(R, 2 * MR * kc)
        for p in 0:(kc - 1), i in 0:(MR - 1)
            re, im = reim(Amat[i + 1, p + 1])
            pa[p * (2MR) + 2i + 1] = re
            pa[p * (2MR) + 2i + 2] = im
        end
        return pa
    end

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

    ilv_pack(k, Amat, Bmat, kc) =
        (interleaved_pack_a(mr(k), Amat, kc), planar_pack_b(nr(k), Bmat, kc))

    # Complex row `i` of column `j` is lanes (2u+1, 2u+2) of vector
    # `v + MV*j + 1`, `(v, u) = divrem(i, W÷2)`, `MV = 2MR÷W`.
    function ilv_read(acc, MR::Int, NR::Int, W::Int, i::Int, j::Int)
        MV = (2 * MR) ÷ W
        v, u = divrem(i, W ÷ 2)
        vec = acc[v + MV * j + 1]
        return Complex(vec[2 * u + 1], vec[2 * u + 2])
    end

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

    MENU64 = kernel_shapes(ComplexF64, FMAddSubMethod())
    MENU32 = kernel_shapes(ComplexF32, FMAddSubMethod())
    SMALL = (2, 3, 4)            # MV = 1, cheap
    ALL64 = (MENU64..., SMALL)
    ALL32 = (MENU32..., SMALL)

    # The same tolerances as test_onem_kernel.jl / the real path: two fused
    # ops per complex MAC per part, like planar and 1m. Needing to loosen
    # these would be a bug signal.
    tol(::Type{ComplexF64}) = 1.0e-11
    tol(::Type{ComplexF32}) = 2.0f-4

    # ------------------------------------------------------------------
    # Construction, forwarding, method traits
    # ------------------------------------------------------------------

    @testset "construction and DescriptorKernel forwarding" begin
        k = FMAddSubKernel(Val(12), Val(8), ComplexF64, Val(8))
        @test k isa FMAddSubKernel{12, 8, ComplexF64, 8}
        @test k.descriptor isa
            ComplexKernelDescriptor{12, 8, ComplexF64, InterleavedFormat, PlanarFormat}
        @test isconcretetype(typeof(k))
        @test mr(k) == 12
        @test nr(k) == 8
        @test scalartype(k) === ComplexF64
        @test realtype(k) === Float64
        @test lanewidth(k) == 8
        @test avecs_per_column(k) == 3
        @test complex_method(k) === FMAddSubMethod()
        @test a_format(k) === InterleavedFormat()
        @test b_format(k) === PlanarFormat()
        @test reals_per_element(InterleavedFormat()) == 2

        # Planar's footprint on both operands; half of 1e's on A.
        @test packed_a_per_k(k) == 2 * 12
        @test packed_b_per_k(k) == 2 * 8
        @test packed_a_length(k, 7) == 2 * 12 * 7
        @test packed_b_length(k, 7) == 2 * 8 * 7
        @test packed_a_per_k(k) * 2 == packed_a_per_k(OneMKernel(Val(12), Val(8), ComplexF64, Val(8)))
        @test packed_a_per_k(FMAddSubKernel(Val(8), Val(8), ComplexF64, Val(8))) ==
            packed_a_per_k(PlanarKernel(Val(8), Val(8), ComplexF64, Val(8)))

        @test a_reals(FMAddSubMethod()) == 2
        @test b_reals(FMAddSubMethod()) == 2
        @test accumulator_planes(FMAddSubMethod()) == 1

        @test lanewidth(FMAddSubKernel(Val(8), Val(4), ComplexF64)) == 4
        @test lanewidth(FMAddSubKernel(Val(8), Val(4), ComplexF32)) == 8

        @test_throws ArgumentError FMAddSubKernel(Val(12), Val(8), ComplexF64, Val(16))
        @test_throws ArgumentError FMAddSubKernel(Val(8), Val(4), ComplexF64, Val(0))
        @test_throws ArgumentError FMAddSubKernel(Val(8), Val(4), Float64)
        @test_throws ArgumentError FMAddSubKernel(Val(8), Val(4), Float64, Val(4))
        # Odd W is rejected even where mod(2MR, W) == 0 (2*3 = 6 at W = 3).
        @test_throws ArgumentError FMAddSubKernel(Val(3), Val(4), ComplexF64, Val(3))

        @test_throws MethodError _QSF.packed_a_offset(k, 0, 0)
        # Plane 0, index over reals: p*2MR + i.
        @test _QSF.packed_a_plane_offset(k, 0, 5, 2) == 2 * 24 + 5
        @test _QSF.packed_b_plane_offset(k, 1, 3, 2) == 2 * 16 + 8 + 3
    end

    @testset "zero_accumulator: 1m's layout and count" begin
        for (T, menu) in ((ComplexF64, ALL64), (ComplexF32, ALL32))
            R = real(T)
            for (MR, NR, W) in menu
                k = FMAddSubKernel(Val(MR), Val(NR), T, Val(W))
                acc = zero_accumulator(k)
                @test acc isa NTuple{((2 * MR) ÷ W) * NR, Vec{W, R}}
                @test all(v -> all(iszero, Tuple(v)), acc)
                if mod(2 * MR, W) == 0 && iseven(W)
                    @test acc === zero_accumulator(OneMKernel(Val(MR), Val(NR), T, Val(W)))
                end
            end
        end
    end

    # ------------------------------------------------------------------
    # The lane primitives and the algebra
    # ------------------------------------------------------------------

    @testset "_fmaddsub is lane-exact x86 fmaddsub; _swap_pairs swaps pairs" begin
        rng = MersenneTwister(0xADD5)
        for R in (Float64, Float32), N in (2, 4, 8, 16)
            for _ in 1:50
                x = Vec{N, R}(ntuple(_ -> randn(rng, R), N))
                y = Vec{N, R}(ntuple(_ -> randn(rng, R), N))
                c = Vec{N, R}(ntuple(_ -> randn(rng, R), N))
                r = _QSF._fmaddsub(x, y, c)
                # ONE fused rounding per lane: exactly Base's scalar fma, with
                # an exactly-negated addend in the even (0-based) lanes.
                want = ntuple(
                    l -> isodd(l) ? fma(x[l], y[l], -c[l]) : fma(x[l], y[l], c[l]), N
                )
                @test Tuple(r) === want
                s = _QSF._swap_pairs(x)
                @test Tuple(s) === ntuple(l -> isodd(l) ? x[l + 1] : x[l - 1], N)
            end
        end
        # Signed zeros and non-finite values pass through as fma defines them.
        x = Vec{4, Float64}((0.0, -0.0, Inf, 1.0))
        y = Vec{4, Float64}((1.0, 1.0, 1.0, NaN))
        c = Vec{4, Float64}((0.0, 0.0, 1.0, 1.0))
        @test isequal(
            Tuple(_QSF._fmaddsub(x, y, c)),
            (fma(0.0, 1.0, -0.0), fma(-0.0, 1.0, 0.0), fma(Inf, 1.0, -1.0), fma(1.0, NaN, 1.0))
        )
    end

    @testset "nesting order: swap(a)*bi must be the INNER op" begin
        # One complex MAC on a single element, done both ways. The correct
        # nesting gives c + a*b; the reversed one puts `ai*bi - ar*br` in the
        # real part -- a wrong answer, not a re-rounding -- so this pins the
        # composition identity the whole kernel rests on.
        for R in (Float64, Float32)
            a = Vec{2, R}((R(3), R(5)))        # 3 + 5im
            b = Complex{R}(R(7), R(-2))
            c = Vec{2, R}((R(11), R(13)))
            br = Vec{2, R}(real(b))
            bi = Vec{2, R}(imag(b))
            right = _QSF._fmaddsub(a, br, _QSF._fmaddsub(_QSF._swap_pairs(a), bi, c))
            want = Complex{R}(R(11), R(13)) + Complex{R}(R(3), R(5)) * b
            @test Complex(right[1], right[2]) == want
            wrong = _QSF._fmaddsub(_QSF._swap_pairs(a), bi, _QSF._fmaddsub(a, br, c))
            @test Complex(wrong[1], wrong[2]) != want
            @test wrong[1] == R(11) + R(5) * imag(b) - R(3) * real(b)
        end
    end

    # ------------------------------------------------------------------
    # Cliff A
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Numerical agreement
    # ------------------------------------------------------------------

    @testset "accumulate vs scalar complex dot product" begin
        for (T, menu) in ((ComplexF64, ALL64), (ComplexF32, ALL32))
            R = real(T)
            for (MR, NR, W) in menu
                k = FMAddSubKernel(Val(MR), Val(NR), T, Val(W))
                rng = MersenneTwister(MR * 1000 + NR * 10 + W)
                for kc in (1, 5, 17)
                    Amat = rand(rng, T, MR, kc) .- T(0.5, 0.5)
                    Bmat = rand(rng, T, kc, NR) .- T(0.5, 0.5)
                    pa, pb = ilv_pack(k, Amat, Bmat, kc)
                    @test length(pa) == packed_a_length(k, kc)
                    @test length(pb) == packed_b_length(k, kc)
                    acc = accumulate(k, zero_accumulator(k), pa, pb, kc)
                    expected = scalar_complex_reference(Amat, Bmat, kc, MR, NR)
                    maxerr = zero(R)
                    for j in 0:(NR - 1), i in 0:(MR - 1)
                        got = ilv_read(acc, MR, NR, W, i, j)
                        maxerr = max(maxerr, abs(got - expected[i + 1, j + 1]))
                    end
                    @test maxerr <= tol(T)
                end
            end
        end
    end

    @testset "PackedPanel operands (the driver's form) give the same bits as Vectors" begin
        for (T, (MR, NR, W)) in ((ComplexF64, (4, 5, 4)), (ComplexF32, (16, 8, 16)))
            k = FMAddSubKernel(Val(MR), Val(NR), T, Val(W))
            rng = MersenneTwister(3)
            kc = 9
            pa, pb = ilv_pack(k, rand(rng, T, MR, kc), rand(rng, T, kc, NR), kc)
            accv = accumulate(k, zero_accumulator(k), pa, pb, kc)
            accp = GC.@preserve pa pb accumulate(
                k, zero_accumulator(k),
                packed_panel(pa, 1, length(pa)), packed_panel(pb, 1, length(pb)), kc
            )
            @test accv === accp
        end
    end

    @testset "agrees with PlanarKernel and OneMKernel on the same fixtures" begin
        for (T, menu) in ((ComplexF64, ALL64), (ComplexF32, ALL32))
            R = real(T)
            for (MR, NR, W) in menu
                kf = FMAddSubKernel(Val(MR), Val(NR), T, Val(W))
                km = OneMKernel(Val(MR), Val(NR), T, Val(W))
                rng = MersenneTwister(MR * 7 + NR * 3 + W)
                for kc in (1, 6, 19)
                    Amat = rand(rng, T, MR, kc) .- T(0.5, 0.5)
                    Bmat = rand(rng, T, kc, NR) .- T(0.5, 0.5)
                    pa, pb = ilv_pack(kf, Amat, Bmat, kc)

                    tile() = (s = zeros(T, MR * NR); (s, DestinationTile(s, 0, AffineAxis(0, 1, MR), AffineAxis(0, MR, NR))))
                    sf, df = tile()
                    execute_tile!(kf, df, pa, pb, kc, one(T), zero(T))

                    # 1m: its own 1e A, the SAME planar B.
                    pa_m = zeros(R, 4 * MR * kc)
                    for p in 0:(kc - 1), i in 0:(MR - 1)
                        re, im = reim(Amat[i + 1, p + 1])
                        pa_m[p * 4MR + 2i + 1] = re
                        pa_m[p * 4MR + 2i + 2] = im
                        pa_m[p * 4MR + 2MR + 2i + 1] = -im
                        pa_m[p * 4MR + 2MR + 2i + 2] = re
                    end
                    sm, dm = tile()
                    execute_tile!(km, dm, pa_m, pb, kc, one(T), zero(T))
                    @test maximum(abs.(sf .- sm)) <= tol(T)

                    if mod(MR, W) == 0
                        kp = PlanarKernel(Val(MR), Val(NR), T, Val(W))
                        pa_p = zeros(R, 2 * MR * kc)
                        for p in 0:(kc - 1), i in 0:(MR - 1)
                            re, im = reim(Amat[i + 1, p + 1])
                            pa_p[p * 2MR + i + 1] = re
                            pa_p[p * 2MR + MR + i + 1] = im
                        end
                        sp, dp = tile()
                        execute_tile!(kp, dp, pa_p, pb, kc, one(T), zero(T))
                        @test maximum(abs.(sf .- sp)) <= tol(T)
                    end
                end
            end
        end
    end

    @testset "accumulate: kc == 0 is a no-op; negative kc throws" begin
        k = FMAddSubKernel(Val(4), Val(5), ComplexF64, Val(4))
        acc0 = zero_accumulator(k)
        @test accumulate(k, acc0, Float64[], Float64[], 0) === acc0
        @test_throws ArgumentError accumulate(k, acc0, Float64[], Float64[], -1)
    end

    @testset "accumulate: composes across split kc (bitwise: same op sequence)" begin
        MR, NR, W, kc = 8, 4, 4, 5
        k = FMAddSubKernel(Val(MR), Val(NR), ComplexF64, Val(W))
        rng = MersenneTwister(77)
        pa, pb = ilv_pack(k, rand(rng, ComplexF64, MR, kc), rand(rng, ComplexF64, kc, NR), kc)
        acc_all = accumulate(k, zero_accumulator(k), pa, pb, kc)
        acc_split = zero_accumulator(k)
        for p in 0:(kc - 1)
            acc_split = accumulate(
                k, acc_split,
                view(pa, (p * 2MR + 1):((p + 1) * 2MR)),
                view(pb, (p * 2NR + 1):((p + 1) * 2NR)), 1
            )
        end
        @test acc_all === acc_split
    end

    # ------------------------------------------------------------------
    # store_tile! and execute_tile!
    # ------------------------------------------------------------------

    @testset "store reader: bitwise identical to 1m's on one accumulator" begin
        for (T, menu) in ((ComplexF64, ALL64), (ComplexF32, ALL32))
            R = real(T)
            for (MR, NR, W) in menu
                kf = FMAddSubKernel(Val(MR), Val(NR), T, Val(W))
                km = OneMKernel(Val(MR), Val(NR), T, Val(W))
                MV = (2 * MR) ÷ W
                # Ramp: real row r of column j is 1000j + r, all distinguishable.
                acc = ntuple(MV * NR) do idx
                    v, j = (idx - 1) % MV, (idx - 1) ÷ MV
                    Vec{W, R}(ntuple(l -> R(1000 * j + v * W + l - 1), W))
                end
                # Scattered rows (same physical layout as unit stride): the
                # gate of the vectorized fast path excludes them, so this is
                # the scalar READER under test, as the testset name says. The
                # fast path has its own file,
                # test/microkernels/test_fmaddsub_store_fastpath.jl.
                rows_scattered = ScatterAxis(collect(0:(MR - 1)), MR)
                for (alpha, beta) in ((one(T), zero(T)), (T(2, -1), T(0.5, 0.25)))
                    s1 = [T(i, -i) for i in 1:(MR * NR)]
                    s2 = copy(s1)
                    d1 = DestinationTile(s1, 0, rows_scattered, AffineAxis(0, MR, NR))
                    d2 = DestinationTile(s2, 0, rows_scattered, AffineAxis(0, MR, NR))
                    store_tile!(d1, acc, alpha, beta, kf)
                    store_tile!(d2, acc, alpha, beta, km)
                    @test isequal(s1, s2)
                end
                storage = zeros(T, MR * NR)
                dst = DestinationTile(storage, 0, rows_scattered, AffineAxis(0, MR, NR))
                store_tile!(dst, acc, one(T), zero(T), kf)
                for j in 0:(NR - 1), i in 0:(MR - 1)
                    @test storage[i + j * MR + 1] ==
                        Complex(R(1000 * j + 2 * i), R(1000 * j + 2 * i + 1))
                    @test ilv_read(acc, MR, NR, W, i, j) == storage[i + j * MR + 1]
                end
            end
        end
    end

    @testset "execute_tile!: alpha/beta contract, partial tiles, kc == 0" begin
        for (T, (MR, NR, W)) in ((ComplexF64, (12, 8, 8)), (ComplexF64, (4, 5, 4)), (ComplexF32, (24, 8, 16)))
            k = FMAddSubKernel(Val(MR), Val(NR), T, Val(W))
            rng = MersenneTwister(5)
            for kc in (0, 1, 4, 13),
                    (alpha, beta) in (
                        (one(T), zero(T)), (T(2.5, -1.0), T(-1.75, 0.5)),
                        (one(T), one(T)), (zero(T), T(3.0, 1.0)), (one(T), T(2, 0)),
                    ),
                    (m, n) in ((MR, NR), (1, 1), (3, 2), (MR, 2), (3, NR), (0, 0))

                Amat = rand(rng, T, MR, max(kc, 1))
                Bmat = rand(rng, T, max(kc, 1), NR)
                pa, pb = ilv_pack(k, Amat, Bmat, max(kc, 1))
                ref = kc == 0 ? zeros(T, MR, NR) : scalar_complex_reference(Amat, Bmat, kc, MR, NR)
                Cold = rand(rng, T, max(m, 1), max(n, 1))
                storage = vec(copy(Cold))[1:(m * n)]
                dst = DestinationTile(storage, 0, AffineAxis(0, 1, m), AffineAxis(0, m, n))
                execute_tile!(k, dst, pa, pb, kc, alpha, beta)
                for i in 0:(m - 1), j in 0:(n - 1)
                    want = (kc == 0 || iszero(alpha)) ? beta * Cold[i + 1, j + 1] :
                        alpha * ref[i + 1, j + 1] + beta * Cold[i + 1, j + 1]
                    @test abs(storage[i + j * m + 1] - want) <= tol(T) * max(1, abs(want))
                end
            end
        end
    end

    @testset "execute_tile!: validation (undersized panels, oversized tiles)" begin
        MR, NR, W = 4, 5, 4
        T = ComplexF64
        k = FMAddSubKernel(Val(MR), Val(NR), T, Val(W))
        storage = zeros(T, MR * NR)
        dst = DestinationTile(storage, 0, AffineAxis(0, 1, MR), AffineAxis(0, MR, NR))
        @test_throws ArgumentError execute_tile!(k, dst, Float64[], Float64[], -1, one(T), zero(T))
        too_tall = DestinationTile(storage, 0, AffineAxis(0, 1, MR + 1), AffineAxis(0, MR, NR))
        @test_throws ArgumentError execute_tile!(k, too_tall, Float64[], Float64[], 1, one(T), zero(T))
        # A needs 2*MR*kc (planar's, half of 1m's 4*MR*kc); B 2*NR*kc.
        @test_throws DimensionMismatch execute_tile!(
            k, dst, zeros(2MR * 3 - 1), zeros(2NR * 3), 3, one(T), zero(T)
        )
        @test_throws DimensionMismatch execute_tile!(
            k, dst, zeros(2MR * 3), zeros(2NR * 3 - 1), 3, one(T), zero(T)
        )
        fill!(storage, T(2, 3))
        execute_tile!(k, dst, Float64[], Float64[], 5, zero(T), zero(T))
        @test all(iszero, storage)
    end

    @testset "scattered and negative-stride destinations agree with the regular path" begin
        MR, NR, W = 12, 8, 8
        T = ComplexF64
        k = FMAddSubKernel(Val(MR), Val(NR), T, Val(W))
        rng = MersenneTwister(21)
        m, n, kc = 11, 5, 6
        pa, pb = ilv_pack(k, rand(rng, T, MR, kc), rand(rng, T, kc, NR), kc)
        ref_storage = zeros(T, m * n)
        execute_tile!(k, DestinationTile(ref_storage, 0, AffineAxis(0, 1, m), AffineAxis(0, m, n)), pa, pb, kc, one(T), zero(T))
        perm = [0, 3, 1, 7, 2, 9, 4, 10, 5, 6, 8]
        ir_storage = zeros(T, m * n)
        execute_tile!(k, DestinationTile(ir_storage, 0, ScatterAxis(perm, m), AffineAxis(0, m, n)), pa, pb, kc, one(T), zero(T))
        for i in 0:(m - 1), j in 0:(n - 1)
            @test ir_storage[perm[i + 1] + m * j + 1] == ref_storage[i + m * j + 1]
        end
        neg_storage = zeros(T, m * n)
        execute_tile!(
            k, DestinationTile(neg_storage, (m - 1) + m * (n - 1), AffineAxis(0, -1, m), AffineAxis(0, -m, n)),
            pa, pb, kc, one(T), zero(T)
        )
        for i in 0:(m - 1), j in 0:(n - 1)
            @test neg_storage[(m - 1 - i) + m * (n - 1 - j) + 1] == ref_storage[i + m * j + 1]
        end
    end

    @testset "padded-lane isolation: nonfinite padding never propagates" begin
        MR, NR, W = 12, 8, 8
        T = ComplexF64
        k = FMAddSubKernel(Val(MR), Val(NR), T, Val(W))
        m, n, kc = 3, 2, 1
        pa = zeros(Float64, packed_a_length(k, kc))
        pb = zeros(Float64, packed_b_length(k, kc))
        for t in 0:(MR - 1)
            pa[2t + 1] = t < m ? Float64(t + 2) : Inf
            pa[2t + 2] = t < m ? Float64(t + 1) : Inf
        end
        for j in 0:(NR - 1)
            pb[j + 1] = j < n ? Float64(7 + j) : Inf
            pb[NR + j + 1] = j < n ? Float64(2 + j) : Inf
        end
        acc = accumulate(k, zero_accumulator(k), pa, pb, kc)
        @test any(!isfinite(ilv_read(acc, MR, NR, W, i, j)) for j in 0:(NR - 1), i in 0:(MR - 1) if i >= m || j >= n)
        storage = fill(T(1, 1), m * n)
        store_tile!(DestinationTile(storage, 0, AffineAxis(0, 1, m), AffineAxis(0, m, n)), acc, one(T), zero(T), k)
        @test all(isfinite, storage)
        for i in 0:(m - 1), j in 0:(n - 1)
            @test storage[i + m * j + 1] == Complex(Float64(i + 2), Float64(i + 1)) * Complex(Float64(7 + j), Float64(2 + j))
        end
        poison = fill(T(Inf, NaN), m * n)
        store_tile!(DestinationTile(poison, 0, AffineAxis(0, 1, m), AffineAxis(0, m, n)), acc, one(T), zero(T), k)
        @test all(isfinite, poison)
        nanacc = ntuple(_ -> Vec{W, Float64}(NaN), Val(((2 * MR) ÷ W) * NR))
        target = fill(T(2, 3), m * n)
        store_tile!(DestinationTile(target, 0, AffineAxis(0, 1, m), AffineAxis(0, m, n)), nanacc, zero(T), T(2, 0), k)
        @test all(==(T(4, 6)), target)
        ed = DestinationTile(T[], 0, AffineAxis(0, 1, 0), AffineAxis(0, 0, 0))
        @test store_tile!(ed, acc, one(T), zero(T), k) === ed
    end

    # ------------------------------------------------------------------
    # Cliff B
    # ------------------------------------------------------------------

    @testset "Cliff B: zero allocations on a scattered fixture, every shape" begin
        run_accumulate(k, pa, pb, kc) = accumulate(k, zero_accumulator(k), pa, pb, kc)
        function run_execute(k, dst, pa, pb, kc, alpha, beta)
            execute_tile!(k, dst, pa, pb, kc, alpha, beta)
            return nothing
        end
        for (T, menu) in ((ComplexF64, ALL64), (ComplexF32, ALL32))
            for (MR, NR, W) in menu
                k = FMAddSubKernel(Val(MR), Val(NR), T, Val(W))
                rng = MersenneTwister(MR * 31 + NR)
                kc = 8
                pa, pb = ilv_pack(k, rand(rng, T, MR, kc), rand(rng, T, kc, NR), kc)
                storage = zeros(T, MR * NR)
                dst = DestinationTile(storage, 0, ScatterAxis(collect(0:(MR - 1)), MR), AffineAxis(0, MR, NR))
                alpha, beta = T(2, -1), T(0.5, 0.25)
                run_accumulate(k, pa, pb, kc)
                run_execute(k, dst, pa, pb, kc, alpha, beta)
                @test (@allocated run_accumulate(k, pa, pb, kc)) == 0 skip = (VERSION < v"1.11")
                @test (@allocated run_execute(k, dst, pa, pb, kc, alpha, beta)) == 0 skip = (VERSION < v"1.11")
            end
        end
    end

    # ------------------------------------------------------------------
    # Instruction selection, on the host
    # ------------------------------------------------------------------

    @testset "instruction selection: vfmaddsub, no separate mul/add/sub, no spill" begin
        # Only meaningful where the host has FMA3 -- x86 AVX2/AVX-512. The
        # expectation is per K step at an AVX2-native and an AVX-512-native
        # shape, read off the hot loop of `accumulate` with the driver's
        # `PackedPanel` argument types (benchmark/probes/fmaddsub_codegen.jl
        # has the same count for every shape, including under `-C znver2`).
        #
        # Also skipped on CI (GitHub Actions sets CI=true): these assert an
        # EXACT instruction mix (down to spill counts) in the compiled hot
        # loop, which depends on the host's real, physically-exposed CPU
        # features -- not something a generic hosted CI runner reliably
        # matches (observed: mismatched counts on both ubuntu-latest and
        # macos-latest runners, presumably a virtualized/hypervisor-exposed
        # feature set that doesn't line up with this test's tuned
        # expectations). This is exactly the kind of measurement this
        # project otherwise takes on dedicated Slurm hardware, not shared CI.
        isa = target_profile().isa
        has_fma = Sys.ARCH === :x86_64 && isa in (:avx2, :avx512) && get(ENV, "CI", "false") != "true"
        function hot_loop(asm)
            lines = split(asm, '\n')
            labels = Dict{String, Int}()
            best = nothing
            for (n, l) in enumerate(lines)
                m = match(r"^(\.LBB\w+):", l)
                m === nothing || (labels[m[1]] = n)
                b = match(r"^\s+j\w+\s+(\.LBB\w+)", l)
                b !== nothing && haskey(labels, b[1]) && (best = (labels[b[1]], n))
            end
            return best === nothing ? "" : join(lines[best[1]:best[2]], '\n')
        end
        shapes = isa === :avx512 ?
            ((ComplexF64, (8, 8, 8)), (ComplexF32, (16, 8, 16)), (ComplexF64, (4, 5, 4))) :
            ((ComplexF64, (4, 5, 4)), (ComplexF32, (8, 5, 8)))
        for (T, (MR, NR, W)) in shapes
            k = FMAddSubKernel(Val(MR), Val(NR), T, Val(W))
            R = real(T)
            asm = sprint() do io
                code_native(
                    io, Base.accumulate,
                    (typeof(k), typeof(zero_accumulator(k)), PackedPanel{R}, PackedPanel{R}, Int);
                    debuginfo = :none, syntax = :intel
                )
            end
            loop = hot_loop(asm)
            MV = (2 * MR) ÷ W
            @test count(r"vfmaddsub\d+p", loop) == 2 * MV * NR skip = !has_fma
            @test count(r"vf(n?madd|n?msub)\d+p[sd]", loop) == 0 skip = !has_fma
            @test count(r"v(mul|add|sub)p[sd]", loop) == 0 skip = !has_fma
            @test count(r"v(shufp|permilp)", loop) == MV skip = !has_fma
            @test count(r"\[r[sb]p", loop) == 0 skip = !has_fma
        end
    end

    # ------------------------------------------------------------------
    # Packing: InterleavedFormat through the ENGINE's packer
    # ------------------------------------------------------------------

    @testset "pack_a! InterleavedFormat: scalar loop and fast path vs local layout" begin
        # Checked bitwise (`isequal`: separates -0.0 from +0.0) -- packing is a
        # copy and, under `conj`, a sign flip; there is no rounding here.
        for T in (ComplexF64, ComplexF32), (MR, NR, W) in (kernel_shapes(T, FMAddSubMethod())..., (3, 2, 2))
            R = real(T)
            k = FMAddSubKernel(Val(MR), Val(NR), T, Val(W))
            kc, lda = 7, MR + 3
            vals = [T(10i + 1, -(10i + 2)) for i in 1:(lda * kc + 4)]
            vals[5] = T(0, 0)
            vals[9] = T(R(-0.0), R(0))
            base = 2
            for f in (identity, conj), m in (MR, max(1, MR - 1))
                src = SourceTile(vals, base, AffineAxis(0, 1, m), AffineAxis(0, lda, kc))
                want = zeros(R, packed_a_length(k, kc))
                for p in 0:(kc - 1), t in 0:(m - 1)
                    z = f(vals[base + t + lda * p + 1])
                    want[p * 2MR + 2t + 1] = real(z)
                    want[p * 2MR + 2t + 2] = imag(z)
                end
                # Scalar loop: a plain Vector destination is excluded by the gate.
                bufv = fill(R(-777), packed_a_length(k, kc))
                pack_a!(bufv, src, k, f)
                @test isequal(bufv, want)
                # PackedPanel destination: the fast path wherever the gate opens.
                bufp = fill(R(-777), packed_a_length(k, kc))
                GC.@preserve bufp pack_a!(packed_panel(bufp, 1, length(bufp)), src, k, f)
                @test isequal(bufp, want)
            end
            # The gate opens for a full unit-stride sliver exactly when the
            # ISA gate does (never hardcoded to this host).
            src = SourceTile(vals, base, AffineAxis(0, 1, MR), AffineAxis(0, lda, kc))
            bufp = zeros(R, packed_a_length(k, kc))
            GC.@preserve bufp begin
                pp = packed_panel(bufp, 1, length(bufp))
                @test _QSF._pack_complex_contiguous_eligible(
                    pp, vals, src.rows, identity, InterleavedFormat(), MR, Val(MR), T
                ) == _QSF._complex_fastpath_isa_eligible()
            end
        end
    end

    # ------------------------------------------------------------------
    # Driver wiring and end to end against the oracle
    # ------------------------------------------------------------------

    @testset "_kernel_from_shape: the FMAddSubMethod arm, and no auto-dispatch" begin
        for (T, menu) in ((ComplexF64, MENU64), (ComplexF32, MENU32))
            for (MR, NR, W) in menu
                k = _kernel_from_shape((MR, NR, W), T, FMAddSubMethod())
                @test k isa FMAddSubKernel{MR, NR, T, W}
            end
            @test_throws ArgumentError _kernel_from_shape((7, 7, 7), T, FMAddSubMethod())
            @test _default_method(T) === PlanarMethod()
            @test !(_QSF._default_kernel(T, 1024, 1024) isa FMAddSubKernel)
        end
    end

    @testset "default_blocking: planar's (same packed reals on both operands)" begin
        for T in (ComplexF64, ComplexF32)
            bf = default_blocking(FMAddSubKernel(Val(8), Val(8), T, Val(8)))
            bp = default_blocking(PlanarKernel(Val(8), Val(8), T, Val(8)))
            bm = default_blocking(OneMKernel(Val(8), Val(8), T, Val(8)))
            @test (bf.mc, bf.kc, bf.nc) == (bp.mc, bp.kc, bp.nc)
            @test bf.mc == 2 * bm.mc
        end
    end

    @testset "end to end: named FMAddSubKernel vs execute_tilewise! with the planar reference" begin
        for (T, (MR, NR, W)) in (
                (ComplexF64, (12, 8, 8)), (ComplexF64, (4, 5, 4)),
                (ComplexF32, (24, 8, 16)), (ComplexF32, (8, 5, 8)),
            )
            rng = MersenneTwister(4242)
            M, N, K = 37, 23, 41      # straddle the tile both ways; several kc panels
            Amat = rand(rng, T, M, K) .- T(0.5, 0.5)
            Bmat = rand(rng, T, K, N) .- T(0.5, 0.5)
            Cinit = rand(rng, T, M, N)
            alpha, beta = T(1.5, -0.25), T(-0.75, 0.5)
            kref = T === ComplexF64 ? PlanarKernel(Val(4), Val(5), T, Val(4)) :
                PlanarKernel(Val(8), Val(5), T, Val(8))
            plan_with(C, kern; kw...) = QuasiStrided.plan_contract(
                _QSF.StridedView(C), _QSF.StridedView(Amat), (1, 2),
                _QSF.StridedView(Bmat), (2, 3), (1, 3); kernel = kern, kw...
            )
            for (cA, cB) in ((false, false), (true, false), (false, true), (true, true))
                want = alpha .* ((cA ? conj.(Amat) : Amat) * (cB ? conj.(Bmat) : Bmat)) .+ beta .* Cinit

                Cf = copy(Cinit)
                QuasiStrided.execute!(
                    plan_with(Cf, FMAddSubKernel(Val(MR), Val(NR), T, Val(W)); kc = 16, conjA = cA, conjB = cB),
                    alpha, beta
                )
                # The oracle, run with the REFERENCE (planar) kernel.
                Cref = copy(Cinit)
                QuasiStrided.execute_tilewise!(
                    plan_with(Cref, kref; kc = 16, conjA = cA, conjB = cB), alpha, beta
                )
                # ... and with this kernel, which isolates the driver wiring.
                Ctw = copy(Cinit)
                QuasiStrided.execute_tilewise!(
                    plan_with(Ctw, FMAddSubKernel(Val(MR), Val(NR), T, Val(W)); kc = 16, conjA = cA, conjB = cB),
                    alpha, beta
                )
                @test maximum(abs.(Cf .- Cref)) <= tol(T) * 64
                @test maximum(abs.(Cf .- Ctw)) <= tol(T) * 64
                @test maximum(abs.(Cf .- want)) <= tol(T) * 64
            end
        end
    end

    @testset "end to end: permuted/strided tensor contraction vs oracle" begin
        # A[a,k,b] with a non-unit-stride M composite and a transposed B, so
        # neither operand takes the contiguous pack fast path everywhere.
        T = ComplexF64
        rng = MersenneTwister(99)
        A = rand(rng, T, 5, 9, 7) .- T(0.5, 0.5)       # a, k, b
        B = rand(rng, T, 6, 9) .- T(0.5, 0.5)          # n, k
        C = zeros(T, 5, 6, 7)                          # a, n, b
        Cref = copy(C)
        for kern in (FMAddSubKernel(Val(4), Val(5), T, Val(4)), FMAddSubKernel(Val(12), Val(8), T, Val(8)))
            fill!(C, zero(T))
            plan = QuasiStrided.plan_contract(
                _QSF.StridedView(C), _QSF.StridedView(A), (1, 2, 3),
                _QSF.StridedView(B), (4, 2), (1, 4, 3); kernel = kern
            )
            QuasiStrided.execute!(plan, one(T), zero(T))
            fill!(Cref, zero(T))
            planr = QuasiStrided.plan_contract(
                _QSF.StridedView(Cref), _QSF.StridedView(A), (1, 2, 3),
                _QSF.StridedView(B), (4, 2), (1, 4, 3); kernel = PlanarKernel(Val(4), Val(5), T, Val(4))
            )
            QuasiStrided.execute_tilewise!(planr, one(T), zero(T))
            @test maximum(abs.(C .- Cref)) <= tol(T) * 64
            direct = [sum(A[a, k, b] * B[n, k] for k in 1:9) for a in 1:5, n in 1:6, b in 1:7]
            @test maximum(abs.(C .- direct)) <= tol(T) * 64
        end
    end
end
