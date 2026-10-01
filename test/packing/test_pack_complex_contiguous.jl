# The vectorized complex packing fast path (`pack_complex_contiguous!`),
# checked bitwise against test_pack_real.jl's reference layouts. Gate
# expectations derive from the live profile, so test/forced_isa_runner.jl
# checks the same outputs with the fast path off.

using QuasiStrided: target_profile, unknown_target, TargetProfile, CacheLevel,
    kernel_shapes, PlanarMethod, OneMMethod, FMAddSubMethod, SliverSpec, complex_contiguous_eligible,
    complex_fastpath_isa_eligible

const FASTPATH_ON = complex_fastpath_isa_eligible()

# The extremes of each method's menu plus extents that are not multiples of any
# lane width (interleaved shares 1e's shuffle); B is planar under every method.
_menu_extremes(T, method, i) = extrema(s[i] for s in kernel_shapes(T, method))
const FAST_A_CASES = [
    (T, fa, MR) for T in (ComplexF64, ComplexF32)
        for (fa, method, extra) in (
            (PlanarFormat(), PlanarMethod(), (1, 7)), (OneEFormat(), OneMMethod(), (1, 7)),
            (InterleavedFormat(), FMAddSubMethod(), ()),
        )
        for MR in unique((_menu_extremes(T, method, 1)..., extra...))
]
const FAST_B_NRS = (1, 3, 7, 8)

# Packs into a `PackedPanel` (the only destination the gate admits).
# `S` is the storage eltype; a mixed one holds values `T` must round.
function check_panel(spec, T, PD, fmt, lane, step, f; S = T)
    storage = S === T ? complex_storage(T, 40000) : S.(complex_storage(ComplexF64, 40000) ./ 3)
    src, g = pack_fixture(storage, 17, lane, step)
    len = _ref_rpe(fmt) * PD * length(step)
    got, canaries = pack_into(:panel, real(T), len, src, spec, f)
    @test canaries
    @test all(isequal.(got, ref_pack(fmt, T, PD, length(step), length(lane), g, f)))
    return storage, src
end

@testset "complex pack fast path: A panel, $T / $(typeof(fa)) / MR=$MR" for (T, fa, MR) in FAST_A_CASES
    kernel = Descriptor(Val(MR), Val(3), T, fa, PlanarFormat())
    for f in (identity, conj), S in (T, T === ComplexF64 ? ComplexF32 : ComplexF64)
        storage, src = check_panel(sliver_spec(kernel, 1), T, MR, fa, AffineAxis(0, 1, MR), AffineAxis(0, 997, 5), f; S)
        @test complex_contiguous_eligible(src, sliver_spec(kernel, 1), f, MR) == FASTPATH_ON
    end
end

@testset "complex pack fast path: B panel, $T / NR=$NR" for T in (ComplexF64, ComplexF32), NR in FAST_B_NRS
    kernel = Descriptor(Val(4), Val(NR), T, PlanarFormat(), PlanarFormat())
    for f in (identity, conj), S in (T, T === ComplexF64 ? ComplexF32 : ComplexF64)
        storage, src = check_panel(sliver_spec(kernel, 2), T, NR, PlanarFormat(), AffineAxis(0, 1, NR), AffineAxis(0, 997, 5), f; S)
        @test complex_contiguous_eligible(src, sliver_spec(kernel, 2), f, NR) == FASTPATH_ON
    end
end

@testset "complex pack fast path: scattered K steps, 1e B, ineligible shapes ($T)" for T in (ComplexF64, ComplexF32)
    MR, NR, k_block_length = 8, 6, 6
    koffs = [((p * 5) % 7) * 1013 + 3p for p in 0:(k_block_length - 1)]
    for fa in (PlanarFormat(), OneEFormat(), InterleavedFormat()), f in (identity, conj)
        kernel = Descriptor(Val(MR), Val(NR), T, fa, OneEFormat())
        # The lane axis must be unit-stride; the step axis may scatter.
        check_panel(sliver_spec(kernel, 1), T, MR, fa, AffineAxis(0, 1, MR), view(koffs, 1:k_block_length), f)
        check_panel(sliver_spec(kernel, 2), T, NR, OneEFormat(), AffineAxis(0, 1, NR), AffineAxis(0, 997, k_block_length), f)
        # Ineligible shapes fall back to the scalar loop, into the same panel.
        for lane in (AffineAxis(20, -1, MR), AffineAxis(0, 3, MR), AffineAxis(0, 1, MR - 3))
            check_panel(sliver_spec(kernel, 1), T, MR, fa, lane, AffineAxis(3000, -97, k_block_length), f)
        end
    end
end

@testset "complex pack fast path: eligibility gate, one violation at a time" begin
    T = ComplexF64
    MR = 8
    storage = complex_storage(T, 4000)
    offs = collect(0:(MR - 1))
    elig(;
        store = storage, ax = AffineAxis(0, 1, MR), tr = identity, fmt = PlanarFormat(), valid = MR,
    ) = complex_contiguous_eligible(Tile(store, 0, ax, AffineAxis(0, MR, 2)), SliverSpec{1, MR, typeof(fmt), T}(), tr, valid)

    @test elig() == FASTPATH_ON
    @test elig(tr = conj) == FASTPATH_ON
    @test elig(fmt = OneEFormat()) == FASTPATH_ON
    @test elig(fmt = InterleavedFormat()) == FASTPATH_ON
    @test !elig(store = view(storage, 1:100))                      # not a DenseVector
    @test !elig(store = collect(reshape(storage, 40, 100)))        # rank 2
    @test elig(store = ComplexF32.(storage)) == FASTPATH_ON         # converted lanes
    @test !elig(store = real.(storage))                            # real storage
    @test !elig(ax = AffineAxis(0, 2, MR))
    @test !elig(ax = AffineAxis(0, -1, MR))
    @test !elig(ax = view(offs, 1:MR))
    @test !elig(valid = MR - 1)                                    # padding lanes
    @test !elig(tr = z -> 2z)
    @test !elig(fmt = RealFormat())
end

@testset "complex pack fast path: ISA gate is a register-width question" begin
    @test complex_fastpath_isa_eligible(synthetic(:avx512))
    for key in (:avx2, :neon, :unknown)
        @test !complex_fastpath_isa_eligible(synthetic(key))
    end
    @test FASTPATH_ON == (target_profile().isa === :avx512)
end
