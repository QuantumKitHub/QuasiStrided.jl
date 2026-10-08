# The contiguous packing fast paths: the real one, and the vectorized complex
# one (`pack_complex_contiguous!`) checked bitwise against the reference layouts. Gate
# expectations derive from the live profile, so test/forced_isa_runner.jl
# checks the same outputs with the fast path off.

include("helpers.jl")

using QuasiStrided: target_profile, unknown_target, TargetProfile, CacheLevel,
    kernel_shapes, PlanarKernel, OneMKernel, FMAddSubKernel, SliverSpec, complex_contiguous_eligible,
    complex_fastpath_isa_eligible

const FASTPATH_ON = complex_fastpath_isa_eligible()

# The extremes of each kernel's menu plus extents that are not multiples of any
# lane width (interleaved shares 1e's shuffle); B is planar under every kernel.
_menu_extremes(T, K, i) = extrema(s[i] for s in kernel_shapes(T, K))
# Every 1m menu MR: no other test compiles the 1e A packing.
_menu_rows(T, K) = K === OneMKernel ? Tuple(s[1] for s in kernel_shapes(T, K)) : _menu_extremes(T, K, 1)
const FAST_A_CASES = [
    (T, fa, MR) for T in (ComplexF64, ComplexF32)
        for (fa, K, extra) in (
            (PlanarFormat(), PlanarKernel, (1, 7)), (OneEFormat(), OneMKernel, (1, 7)),
            (InterleavedFormat(), FMAddSubKernel, ()),
        )
        for MR in unique((_menu_rows(T, K)..., extra...))
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
    @test complex_fastpath_isa_eligible(synthetic(:avx512)) && complex_fastpath_isa_eligible(synthetic(:avx2))
    for key in (:neon, :unknown)
        @test !complex_fastpath_isa_eligible(synthetic(key))
    end
    @test FASTPATH_ON == (target_profile().isa in (:avx512, :avx2))
end

@testset "pack! contiguous fast path: fires exactly when eligible ($T, L=$L, side $i)" for
    T in (Float64, Float32), (L, i) in ((4, 1), (16, 1), (6, 2))

    kernel = Descriptor(Val(L), Val(L), T)
    vals = T.(collect(1.0:2000.0))
    mixed = (T === Float64 ? Float32 : Float64).(collect(1.0:2000.0) ./ 3)
    storages = @static isdefined(Base, :Memory) ? (vals, copyto!(Memory{T}(undef, 2000), vals), mixed) : (vals, mixed)
    koffs = [7, 900, 300, 1500]
    contig = collect(0:(L - 1))
    spec = sliver_spec(kernel, i)
    for storage in storages
        steps = Any[AffineAxis(0, L, 5), AffineAxis(1800, -L, 6), view(koffs, 1:4)]
        lanes = Any[
            (AffineAxis(0, 1, L), true), (AffineAxis(5, 1, L), true),
            (AffineAxis(0, 2, L), false), (AffineAxis(L + 3, -1, L), false),
            (AffineAxis(0, 1, L - 1), false), (view(contig, 1:L), false),
        ]
        for (lane, eligible) in lanes, step in steps, f in (identity, conj, x -> -x)
            src, g = pack_fixture(storage, 11, lane, step)
            k_block_length = length(step)
            @test real_contiguous_eligible(src, spec, f, length(lane)) ==
                (eligible && (f === identity || f === conj))
            got, canaries = pack_into(:panel, T, L * k_block_length, src, spec, f)
            @test got == ref_pack(RealFormat(), T, L, k_block_length, length(lane), g, f)
            @test canaries
        end
    end
    @test copies_unchanged(conj, T)
    @test !copies_unchanged(conj, complex(T))   # conj is not the identity on complex
end
