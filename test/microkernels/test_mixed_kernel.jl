using QuasiStrided: ComplexRealKernel, RealComplexKernel, ComplexRealMethod, RealComplexMethod, packed_panel

mk_optypes(k::ComplexRealKernel) = (scalartype(k), real(scalartype(k)))
mk_optypes(k::RealComplexKernel) = (real(scalartype(k)), scalartype(k))
mk_pack_a(::ComplexRealKernel, A) = mk_cols(c -> mk_ilv(real(c), imag(c)), A)
mk_pack_b(::ComplexRealKernel, B) = vec(permutedims(B))
mk_pack_a(::RealComplexKernel, A) = vec(A)
mk_pack_b(::RealComplexKernel, B) = mk_cols(c -> mk_ilv(real(c), imag(c)), permutedims(B))
function mk_read(k::ComplexRealKernel, acc, i, j)
    W = lanewidth(k)
    v, u = divrem(i, W ÷ 2)
    vec = acc[v + (2 * mr(k) ÷ W) * j + 1]
    return Complex(vec[2u + 1], vec[2u + 2])
end
function mk_read(k::RealComplexKernel, acc, i, j)
    W = lanewidth(k)
    MV = mr(k) ÷ W
    v, u = divrem(i, W)
    return Complex(acc[v + MV * 2j + 1][u + 1], acc[v + MV * (2j + 1) + 1][u + 1])
end

@testset "mixed-domain kernels" begin
    for (K, s, full) in (
            (ComplexRealKernel, (ComplexF64, 12, 8, 8), true), (ComplexRealKernel, (ComplexF32, 8, 5, 8), true),
            (ComplexRealKernel, (ComplexF64, 3, 2, 2), false),
            (RealComplexKernel, (ComplexF64, 24, 4, 8), true), (RealComplexKernel, (ComplexF32, 16, 3, 8), true),
            (RealComplexKernel, (ComplexF64, 2, 3, 2), false),
        )
        T, MR, NR, W = s
        mk_contract(K(Val(MR), Val(NR), T, Val(W)); full)
    end

    @testset "construction" begin
        cr = ComplexRealKernel(Val(8), Val(4), ComplexF32)
        rc = RealComplexKernel(Val(8), Val(4), ComplexF64)
        @test complex_method(cr) === ComplexRealMethod()
        @test complex_method(rc) === RealComplexMethod()
        @test (a_reals(ComplexRealMethod()), b_reals(ComplexRealMethod())) == (2, 1)
        @test (a_reals(RealComplexMethod()), b_reals(RealComplexMethod())) == (1, 2)
        @test accumulator_planes(ComplexRealMethod()) == accumulator_planes(RealComplexMethod()) == 1
        @test cr.inner isa SIMDKernel{16, 4, Float32, 8}
        @test rc.inner isa SIMDKernel{8, 8, Float64, 4}
        @test (packed_a_per_k(cr), packed_b_per_k(cr), packed_a_per_k(rc), packed_b_per_k(rc)) == (16, 4, 8, 8)
        @test_throws ArgumentError ComplexRealKernel(Val(3), Val(4), ComplexF64, Val(3))  # odd W
        @test_throws ArgumentError RealComplexKernel(Val(6), Val(4), ComplexF64, Val(4))
        @test_throws ArgumentError ComplexRealKernel(Val(8), Val(4), Float64)
        @test_throws ArgumentError RealComplexKernel(Val(8), Val(4), Float64, Val(4))
    end

    # Packing from storage of another precision, `conj` on both sides.
    @testset "pack -> accumulate -> store: $(nameof(K)) $T from $SA x $SB" for (K, T, SA, SB, MR, NR, W) in (
            (ComplexRealKernel, ComplexF32, ComplexF64, Float64, 8, 5, 8),
            (RealComplexKernel, ComplexF32, Float64, ComplexF64, 16, 3, 8),
            (ComplexRealKernel, ComplexF64, ComplexF64, Float32, 12, 8, 8),
            (RealComplexKernel, ComplexF64, Float32, ComplexF32, 24, 4, 8),
        )
        k = K(Val(MR), Val(NR), T, Val(W))
        R = real(T)
        rng = MersenneTwister(7)
        kc, lda = 6, MR + 2
        Av, Bv = rand(rng, SA, lda * kc + 1), rand(rng, SB, kc * NR)
        for f in (identity, conj), (m, n) in ((MR, NR), (MR - 1, NR - 1)), panel in (false, true)
            srcA = SourceTile(Av, 1, AffineAxis(0, 1, m), AffineAxis(0, lda, kc))
            srcB = SourceTile(Bv, 0, AffineAxis(0, NR, kc), AffineAxis(0, 1, n))
            A = zeros(T, MR, kc)
            B = zeros(T, kc, NR)
            A[1:m, :] = f.(reshape(Av[2:(1 + lda * kc)], lda, kc)[1:m, :])
            B[:, 1:n] = f.(permutedims(reshape(Bv, NR, kc))[:, 1:n])
            TA, TB = mk_optypes(k)
            pa, pb = zeros(R, packed_a_length(k, kc)), zeros(R, packed_b_length(k, kc))
            GC.@preserve pa pb begin
                dpa = panel ? packed_panel(pa, 1, length(pa)) : pa
                dpb = panel ? packed_panel(pb, 1, length(pb)) : pb
                pack_a!(dpa, srcA, k, f)
                pack_b!(dpb, srcB, k, f)
            end
            @test isequal((pa, pb), mk_pack(k, TA.(A), TB.(B)))
            alpha, beta = T(1.5, -0.5), T(0.25, 1)
            C0 = rand(rng, T, m * n)
            got = mk_dense(copy(C0))
            execute_tile!(k, DestinationTile(got, 0, AffineAxis(0, 1, m), AffineAxis(0, m, n)), pa, pb, kc, alpha, beta)
            @test mk_close(got, alpha .* vec((A * B)[1:m, 1:n]) .+ beta .* C0, T)
        end
    end

    @testset "end to end: $(nameof(typeof(k))) $TA x $TB -> $TC, B $mode" for (k, TA, TB, TC, acc, mode) in (
            (ComplexRealKernel(Val(12), Val(8), ComplexF64, Val(8)), ComplexF64, Float64, ComplexF64, nothing, :always),
            (ComplexRealKernel(Val(12), Val(8), ComplexF64, Val(8)), ComplexF64, Float64, ComplexF64, nothing, :never),
            (ComplexRealKernel(Val(8), Val(5), ComplexF32, Val(8)), ComplexF32, Float64, ComplexF32, Float32, :auto),
            (RealComplexKernel(Val(24), Val(4), ComplexF64, Val(8)), Float64, ComplexF64, ComplexF64, nothing, :auto),
            (RealComplexKernel(Val(16), Val(3), ComplexF32, Val(8)), Float64, ComplexF32, ComplexF32, Float32, :auto),
        )
        rng = MersenneTwister(4242)
        A, B, C0 = rand(rng, TA, 37, 41), rand(rng, TB, 41, 23), rand(rng, TC, 37, 23)
        alpha, beta = 1.5 - 0.25im, -0.75 + 0.5im
        saved = QuasiStrided._UNPACKED_B_MODE[]
        QuasiStrided._UNPACKED_B_MODE[] = mode
        try
            for (cA, cB) in ((false, false), (true, false), (false, true))
                C = copy(C0)
                plan = plan_contract(
                    StridedView(C), StridedView(A), (1, 2), StridedView(B), (2, 3), (1, 3);
                    kernel = k, kc = 16, conjA = cA, conjB = cB, accumulator = acc
                )
                execute!(plan, alpha, beta)
                want = alpha .* ((cA ? conj.(A) : A) * (cB ? conj.(B) : B)) .+ beta .* C0
                @test maximum(abs, C .- want) <= 64 * mk_tol(TC) * maximum(abs, want)
            end
        finally
            QuasiStrided._UNPACKED_B_MODE[] = saved
        end
    end
end
