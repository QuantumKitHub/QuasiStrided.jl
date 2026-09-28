# The vectorized complex store fast paths (planar here, fmaddsub in
# test_fmaddsub_store_fastpath.jl, which reuses `mk_store_fastpath`).
#
# Values are pinned two ways. On the elements the fast path vectorizes (full
# row blocks) it must match, bitwise, an independent transcription of Base's
# `Complex` `*`/`muladd` expression trees (the arithmetic `_axpby_tile!` does).
# Against the scalar store it is compared with a tolerance only: LLVM contracts
# Base's scalar complex `muladd` depending on inlining context, so the scalar
# path is not bit-reproducible even against itself.
#
# The fast path ships for AVX-512 only; every expectation derives from the live
# profile, so `test/forced_isa_runner.jl` checks the other ISAs too.

using QuasiStrided: PlanarKernel, FMAddSubKernel, KERNEL_SHAPES_C64_FMADDSUB, KERNEL_SHAPES_C32_FMADDSUB, PtrScatterAxis, TargetProfile, CacheLevel,
    target_profile, unknown_target, KERNEL_SHAPES_C64_PLANAR, KERNEL_SHAPES_C32_PLANAR

const STORE_FASTPATH_ON = QuasiStrided._complex_fastpath_isa_eligible()

# `@noinline` identity: keeps LLVM from contracting separately rounded products.
@noinline barrier(x) = x

#   *(z, w)        = Complex(zr*wr - zi*wi, zr*wi + zi*wr)
#   muladd(z, w, x) = Complex(muladd(zr, wr, -muladd(zi, wi, -xr)),
#                             muladd(zr, wi, muladd(zi, wr, xi)))
function ref_axpby(alpha::T, rr::R, ri::R, beta::T, cold::T) where {T, R}
    ar, ai = reim(alpha)
    iszero(beta) && return Complex(barrier(ar * rr) - barrier(ai * ri), barrier(ar * ri) + barrier(ai * rr))
    br, bi = reim(beta)
    cr, ci = reim(cold)
    if isone(beta)
        xr, xi = cr, ci
    else
        xr = barrier(br * cr) - barrier(bi * ci)
        xi = barrier(br * ci) + barrier(bi * cr)
    end
    return Complex(fma(ar, rr, barrier(-fma(ai, ri, -xr))), fma(ar, ri, barrier(fma(ai, rr, xi))))
end

# Every `_axpby_tile!` branch, plus purely imaginary and real non-unit beta.
mk_store_ab(T) = (
    (one(T), zero(T)), (T(-0.5, 0.25), zero(T)), (one(T), one(T)), (T(2, -1), one(T)),
    (T(2.5, -1), T(-1.75, 0.5)), (one(T), T(2, 0)), (T(1, 1), T(0, -3)),
)

# `beta0_exact`: the fast path also matches the scalar store bitwise at beta == 0.
function mk_store_fastpath(K, T, shapes; beta0_exact = false)
    R = real(T)
    tol = 8 * eps(R)
    return @testset "$K store fast path $T $s" for s in shapes
        MR, NR, W = s
        k = K(Val(MR), Val(NR), T, Val(W))
        blk = k isa PlanarKernel ? W : W ÷ 2  # complex rows per vector
        rng = MersenneTwister(97MR + NR)
        # Includes an extent straddling a block and one skipping a whole block.
        extents = unique(
            (
                (MR, NR), (1, 1), (MR, 1), (1, NR), (MR - 1, NR), (max(1, MR ÷ 2 + 1), max(1, NR - 1)),
                (max(1, MR ÷ blk - 1) * blk, NR), (max(1, MR - blk), NR),
            )
        )
        for (m, n) in extents
            acc = map(v -> typeof(v)(ntuple(_ -> R(2rand(rng) - 1), W)), zero_accumulator(k))
            cold = [Complex(R(2rand(rng) - 1), R(2rand(rng) - 1)) for _ in 1:(m * n)]
            for (alpha, beta) in mk_store_ab(T)
                fast = mk_dense(cold)
                dfast = DestinationTile(fast, 0, AffineAxis(0, 1, m), AffineAxis(0, m, n))
                @test QuasiStrided._complex_vector_eligible(dfast, T) == STORE_FASTPATH_ON
                store_tile!(dfast, acc, alpha, beta, k)
                scal = copy(cold)  # scattered rows: always the scalar store
                store_tile!(DestinationTile(scal, 0, ScatterAxis(collect(0:(m - 1)), m), AffineAxis(0, m, n)), acc, alpha, beta, k)
                want = [ref_axpby(alpha, reim(mk_read(k, acc, i, j))..., beta, cold[i + j * m + 1]) for i in 0:(m - 1), j in 0:(n - 1)]
                got = reshape(collect(fast), m, n)
                vectorized = STORE_FASTPATH_ON ? (1:((m ÷ blk) * blk)) : (1:0)
                @test isequal(got[vectorized, :], want[vectorized, :])
                @test all(abs.(got .- want) .<= tol .* max.(abs.(want), 1))
                @test all(abs.(vec(got) .- scal) .<= tol .* max.(abs.(scal), 1))
                beta0_exact && iszero(beta) && @test isequal(vec(got), scal)
            end
        end
    end
end

@testset "planar store fast path" begin
    mk_store_fastpath(PlanarKernel, ComplexF64, KERNEL_SHAPES_C64_PLANAR)
    mk_store_fastpath(PlanarKernel, ComplexF32, KERNEL_SHAPES_C32_PLANAR)

    @testset "eligibility gate: one violated condition at a time" begin
        T = ComplexF64
        m, n = 16, 4
        storage = zeros(T, 4 * m * n)
        eligible(s, rows, cols = AffineAxis(0, m, n)) =
            QuasiStrided._complex_vector_eligible(DestinationTile(s, 0, rows, cols), T)
        @test eligible(storage, AffineAxis(0, 1, m)) == STORE_FASTPATH_ON
        @test !eligible(storage, AffineAxis(0, 2, m), AffineAxis(0, 2m, n))
        @test !QuasiStrided._complex_vector_eligible(DestinationTile(storage, m - 1, AffineAxis(0, -1, m), AffineAxis(0, m, n)), T)
        @test !eligible(storage, ScatterAxis(collect(0:(m - 1)), m))
        ptr_rows = collect(0:(m - 1))
        GC.@preserve ptr_rows @test !eligible(storage, PtrScatterAxis(pointer(ptr_rows), m))
        @test !eligible(view(storage, 1:(m * n)), AffineAxis(0, 1, m))
        @test !eligible(reshape(storage, 4m, n), AffineAxis(0, 1, m))
        @test !eligible(zeros(ComplexF32, m * n), AffineAxis(0, 1, m))
        @test !eligible(zeros(Float64, m * n), AffineAxis(0, 1, m))
    end

    @testset "ISA gate: AVX-512 only" begin
        profile(key, vb, nreg) = TargetProfile(key, Sys.ARCH, "t", vb, nreg, CacheLevel(), CacheLevel(), CacheLevel())
        @test QuasiStrided._complex_fastpath_isa_eligible(profile(:avx512, 64, 32))
        for (key, vb, nreg) in ((:avx2, 32, 16), (:neon, 16, 32), (:unknown, 0, 0))
            @test !QuasiStrided._complex_fastpath_isa_eligible(profile(key, vb, nreg))
        end
        @test !QuasiStrided._complex_fastpath_isa_eligible(unknown_target())
        @test STORE_FASTPATH_ON == (target_profile().vector_bytes == QuasiStrided._isa_vector_bytes(Val(:avx512)))
    end
end


@testset "fmaddsub store fast path" begin
    mk_store_fastpath(FMAddSubKernel, ComplexF64, KERNEL_SHAPES_C64_FMADDSUB; beta0_exact = true)
    mk_store_fastpath(FMAddSubKernel, ComplexF32, KERNEL_SHAPES_C32_FMADDSUB; beta0_exact = true)
end
