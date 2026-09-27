# The vectorized FMAddSub store fast path (src/microkernels/fmaddsub.jl,
# `_store_tile_fmaddsub_vector!`): the interleaved-layout counterpart of the
# planar one, pinned the same four ways test_planar_store_fastpath.jl pins
# that -- values (bitwise against the independent reference on the lanes it
# vectorizes, tolerance against the scalar reader everywhere), the alpha/beta
# contract on the fast path specifically, the gate, and ISA portability. The
# reference `ref_axpby`, the `barrier` it is built on and `STORE_FASTPATH_ON`
# are test_planar_store_fastpath.jl's, which runtests.jl includes into this
# scope first: the arithmetic contract is `_axpby_tile!`'s, shared by every
# complex kernel, so one transcription of it serves both fast paths.

using Test
using Random
using QuasiStrided
using QuasiStrided: FMAddSubKernel, AffineAxis, ScatterAxis, DestinationTile, store_tile!,
    KERNEL_SHAPES_C64_FMADDSUB, KERNEL_SHAPES_C32_FMADDSUB
using SIMD: Vec

const _QSFS = QuasiStrided

# Complex row `i` of column `j` in an FMAddSub/1m accumulator: vector
# `v = i ÷ (W÷2)`, tuple index `v + MV*j + 1`, lanes `2u+1`/`2u+2` (1-based)
# with `u = i % (W÷2)`. Re-derived from the layout claim in `zero_accumulator`'s
# docstring, not from the store's own code.
function fsfp_lanes(acc, MR::Int, NR::Int, W::Int, i::Int, j::Int)
    HW = W ÷ 2
    MV = (2 * MR) ÷ W
    v, u = i ÷ HW, i % HW
    vec = acc[v + MV * j + 1]
    return vec[2 * u + 1], vec[2 * u + 2]
end

function fsfp_acc(::Type{T}, MR::Int, NR::Int, W::Int, seed::Int) where {T}
    R = real(T)
    rng = MersenneTwister(seed)
    NV = ((2 * MR) ÷ W) * NR
    return ntuple(_ -> Vec{W, R}(ntuple(_ -> R(2 * rand(rng) - 1), W)), NV)
end

fsfp_cold(::Type{T}, len::Int, seed::Int) where {T} =
    (
    rng = MersenneTwister(seed);
    T[Complex(real(T)(2 * rand(rng) - 1), real(T)(2 * rand(rng) - 1)) for _ in 1:len]
)

fsfp_ab(::Type{T}) where {T} = (
    (one(T), zero(T)),
    (T(-0.5, 0.25), zero(T)),
    (one(T), one(T)),
    (T(2, -1), one(T)),
    (T(2.5, -1), T(-1.75, 0.5)),
    (one(T), T(2, 0)),
    (T(1, 1), T(0, -3)),
)

const FSFP_MENUS = (
    (ComplexF64, KERNEL_SHAPES_C64_FMADDSUB),
    (ComplexF32, KERNEL_SHAPES_C32_FMADDSUB),
)

fsfp_kernel(::Type{T}, MR::Int, NR::Int, W::Int) where {T} =
    FMAddSubKernel(Val(MR), Val(NR), T, Val(W))

# ---------------------------------------------------------------------------
# 1. Values
# ---------------------------------------------------------------------------

@testset "fmaddsub store fast path: values ($T)" for (T, menu) in FSFP_MENUS
    R = real(T)
    reltol = 8 * eps(R)   # as for planar: one ULP measured, headroom for LLVM

    @testset "($MR,$NR,$W)" for (MR, NR, W) in menu
        HW = W ÷ 2
        # The full tile, one element, one row/column, one short of full, an
        # extent that is not a multiple of HW (a straddling block exists) and
        # one that is exactly a multiple of HW below MR (a block is skipped).
        extents = unique(
            (
                (MR, NR), (1, 1), (MR, 1), (1, NR), (MR - 1, NR),
                (max(1, MR ÷ 2 + 1), max(1, NR - 1)), (max(1, MR ÷ HW - 1) * HW, NR),
                (max(1, MR - HW), NR),
            )
        )
        for (m, n) in extents
            (m <= 0 || n <= 0) && continue
            acc = fsfp_acc(T, MR, NR, W, MR * 97 + NR * 13 + m * 7 + n)
            cold = fsfp_cold(T, m * n, MR * 31 + m * 5 + n)

            for (alpha, beta) in fsfp_ab(T)
                fast = copy(cold)
                dfast = DestinationTile(fast, 0, AffineAxis(0, 1, m), AffineAxis(0, m, n))
                @test _QSFS._complex_vector_eligible(dfast, T) == STORE_FASTPATH_ON
                store_tile!(dfast, acc, alpha, beta, fsfp_kernel(T, MR, NR, W))

                # Same physical layout, scattered rows: the gate excludes it,
                # so this is the scalar reader `_store_tile_fmaddsub!`.
                scal = copy(cold)
                dscal = DestinationTile(
                    scal, 0, ScatterAxis(collect(0:(m - 1)), m), AffineAxis(0, m, n)
                )
                @test !_QSFS._complex_vector_eligible(dscal, T)
                store_tile!(dscal, acc, alpha, beta, fsfp_kernel(T, MR, NR, W))

                for j in 0:(n - 1), i in 0:(m - 1)
                    rr, ri = fsfp_lanes(acc, MR, NR, W, i, j)
                    want = ref_axpby(alpha, rr, ri, beta, cold[i + j * m + 1])
                    got = fast[i + j * m + 1]
                    # (1a) bitwise where the fast path vectorizes: rows in a
                    # full HW-block; the straddling block's rows take the
                    # scalar tail, which inherits LLVM's contraction choice.
                    in_full_block = (i < (m ÷ HW) * HW)
                    if in_full_block && STORE_FASTPATH_ON
                        @test isequal(want, got)
                    end
                    # (1b) tolerance everywhere, against the scalar reader ...
                    ref = scal[i + j * m + 1]
                    @test abs(got - ref) <= reltol * max(abs(ref), one(R))
                    # ... and against the independent reference.
                    @test abs(got - want) <= reltol * max(abs(want), one(R))
                    # Exact agreement with the scalar reader where Base's tree
                    # has no contraction freedom: beta == 0 (unfused `*`).
                    if iszero(beta)
                        @test isequal(got, ref)
                    end
                end
            end
        end
    end
end

# ---------------------------------------------------------------------------
# 2. Contract preservation on the fast path
# ---------------------------------------------------------------------------

@testset "fmaddsub store fast path: contracts ($T)" for (T, menu) in FSFP_MENUS
    R = real(T)
    MR, NR, W = first(menu)
    k = fsfp_kernel(T, MR, NR, W)
    HW = W ÷ 2
    acc = fsfp_acc(T, MR, NR, W, 5)

    @testset "beta == 0 never reads old C (a full, vectorized block)" begin
        poison = fill(T(NaN, Inf), MR * NR)
        d = DestinationTile(poison, 0, AffineAxis(0, 1, MR), AffineAxis(0, MR, NR))
        @test _QSFS._complex_vector_eligible(d, T) == STORE_FASTPATH_ON
        store_tile!(d, acc, T(2, -1), zero(T), k)
        @test all(isfinite, poison)
    end

    @testset "alpha == 0 never reads acc" begin
        nanacc = ntuple(_ -> Vec{W, R}(R(NaN)), Val(((2 * MR) ÷ W) * NR))
        target = fill(T(2, 3), MR * NR)
        d = DestinationTile(target, 0, AffineAxis(0, 1, MR), AffineAxis(0, MR, NR))
        store_tile!(d, nanacc, zero(T), T(2, 0), k)
        @test all(==(T(4, 6)), target)
    end

    @testset "nothing outside the valid rectangle is touched" begin
        # A guard band around an (m, n) rectangle inside a larger dense array;
        # rows unit-stride, so the fast path fires on the full blocks.
        m, n = MR - 1, max(1, NR - 1)
        ld = MR + 3
        storage = fill(T(-7, 7), ld * (NR + 2))
        base = 1 + ld   # one row down, one column in
        d = DestinationTile(storage, base, AffineAxis(0, 1, m), AffineAxis(0, ld, n))
        @test _QSFS._complex_vector_eligible(d, T) == STORE_FASTPATH_ON
        store_tile!(d, acc, one(T), zero(T), k)
        for idx in 0:(length(storage) - 1)
            i, j = (idx - base) % ld, (idx - base) ÷ ld
            inside = idx >= base && 0 <= i < m && 0 <= j < n
            if inside
                rr, ri = fsfp_lanes(acc, MR, NR, W, i, j)
                @test storage[idx + 1] == Complex(rr, ri)
            else
                @test storage[idx + 1] == T(-7, 7)
            end
        end
    end

    @testset "beta is applied exactly once" begin
        cold = fsfp_cold(T, MR * NR, 11)
        got = copy(cold)
        d = DestinationTile(got, 0, AffineAxis(0, 1, MR), AffineAxis(0, MR, NR))
        alpha, beta = T(1.5, -0.5), T(-2, 0.25)
        store_tile!(d, acc, alpha, beta, k)
        for j in 0:(NR - 1), i in 0:(MR - 1)
            rr, ri = fsfp_lanes(acc, MR, NR, W, i, j)
            want = alpha * Complex(rr, ri) + beta * cold[i + j * MR + 1]
            @test abs(got[i + j * MR + 1] - want) <= 8 * eps(R) * max(abs(want), one(R))
        end
    end

    @testset "empty destination is a no-op" begin
        ed = DestinationTile(T[], 0, AffineAxis(0, 1, 0), AffineAxis(0, 0, 0))
        @test store_tile!(ed, acc, one(T), zero(T), k) === ed
    end
end

# ---------------------------------------------------------------------------
# 3. Allocation-free, every shape
# ---------------------------------------------------------------------------

@testset "fmaddsub store fast path: allocation-free ($T)" for (T, menu) in FSFP_MENUS
    for (MR, NR, W) in menu
        k = fsfp_kernel(T, MR, NR, W)
        acc = fsfp_acc(T, MR, NR, W, 3)
        storage = fsfp_cold(T, MR * NR, 4)
        d = DestinationTile(storage, 0, AffineAxis(0, 1, MR), AffineAxis(0, MR, NR))
        for (alpha, beta) in fsfp_ab(T)
            store_tile!(d, acc, alpha, beta, k)
            @test (@allocated store_tile!(d, acc, alpha, beta, k)) == 0
        end
    end
end
