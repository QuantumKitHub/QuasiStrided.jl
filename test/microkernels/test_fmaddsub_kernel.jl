using QuasiStrided: FMAddSubKernel, FMAddSubMethod, PlanarKernel, PlanarMethod, OneMKernel,
    InterleavedFormat, KernelMethod, lanewidth, kernel_shapes, packed_panel,
    _kernel_from_shape, _default_method, accumulator_planes, target_profile, PackedPanel
using SIMD: Vec
using InteractiveUtils: code_native

const _QSF = QuasiStrided

@testset "FMAddSubKernel" begin
    full = ((ComplexF64, (12, 8, 8)), (ComplexF32, (8, 5, 8)))
    for (T, (MR, NR, W)) in full
        mk_contract(FMAddSubKernel(Val(MR), Val(NR), T, Val(W)))
    end
    for T in (ComplexF64, ComplexF32), s in (kernel_shapes(T, FMAddSubMethod())..., (2, 3, 4))
        (T, s) in full || mk_contract(FMAddSubKernel(Val(s[1]), Val(s[2]), T, Val(s[3])); full = false)
    end

    @testset "construction" begin
        @test KernelMethod(FMAddSubKernel(Val(8), Val(4), ComplexF64)) === FMAddSubMethod()
        @test accumulator_planes(FMAddSubMethod()) == 1
        @test lanewidth(FMAddSubKernel(Val(8), Val(4), ComplexF32)) == 8
        @test_throws ArgumentError FMAddSubKernel(Val(12), Val(8), ComplexF64, Val(16))
        @test_throws ArgumentError FMAddSubKernel(Val(3), Val(4), ComplexF64, Val(3))  # odd W
        @test_throws ArgumentError FMAddSubKernel(Val(8), Val(4), ComplexF64, Val(0))
        @test_throws ArgumentError FMAddSubKernel(Val(8), Val(4), Float64)
        @test_throws ArgumentError FMAddSubKernel(Val(8), Val(4), Float64, Val(4))
    end

    @testset "fmaddsub is lane-exact x86 fmaddsub; swap_pairs swaps pairs" begin
        rng = MersenneTwister(0xADD5)
        for R in (Float64, Float32), N in (2, 4, 8, 16), _ in 1:10
            x, y, c = (Vec{N, R}(ntuple(_ -> randn(rng, R), N)) for _ in 1:3)
            # One fused rounding per lane, with an exactly negated addend in even lanes.
            @test Tuple(_QSF.fmaddsub(x, y, c)) ===
                ntuple(l -> isodd(l) ? fma(x[l], y[l], -c[l]) : fma(x[l], y[l], c[l]), N)
            @test Tuple(_QSF.swap_pairs(x)) === ntuple(l -> isodd(l) ? x[l + 1] : x[l - 1], N)
        end
        x = Vec{4, Float64}((0.0, -0.0, Inf, 1.0))
        y = Vec{4, Float64}((1.0, 1.0, 1.0, NaN))
        c = Vec{4, Float64}((0.0, 0.0, 1.0, 1.0))
        @test isequal(
            Tuple(_QSF.fmaddsub(x, y, c)),
            (fma(0.0, 1.0, -0.0), fma(-0.0, 1.0, 0.0), fma(Inf, 1.0, -1.0), fma(1.0, NaN, 1.0))
        )
    end

    @testset "nesting order: swap(a)*bi must be the inner op" begin
        for R in (Float64, Float32)
            a = Vec{2, R}((R(3), R(5)))
            b = Complex{R}(7, -2)
            c = Vec{2, R}((R(11), R(13)))
            br, bi = Vec{2, R}(real(b)), Vec{2, R}(imag(b))
            right = _QSF.fmaddsub(a, br, _QSF.fmaddsub(_QSF.swap_pairs(a), bi, c))
            @test Complex(right[1], right[2]) == Complex{R}(11, 13) + Complex{R}(3, 5) * b
            wrong = _QSF.fmaddsub(_QSF.swap_pairs(a), bi, _QSF.fmaddsub(a, br, c))
            @test wrong[1] == R(11) + R(5) * imag(b) - R(3) * real(b)
        end
    end

    @testset "PackedPanel operands (the driver's form) give the same bits as Vectors" begin
        for (T, (MR, NR, W)) in ((ComplexF64, (4, 5, 4)), (ComplexF32, (16, 8, 16)))
            k = FMAddSubKernel(Val(MR), Val(NR), T, Val(W))
            rng = MersenneTwister(3)
            pa, pb = mk_pack(k, rand(rng, T, MR, 9), rand(rng, T, 9, NR))
            accp = GC.@preserve pa pb mk_run_acc(k, packed_panel(pa, 1, length(pa)), packed_panel(pb, 1, length(pb)), 9)
            @test mk_run_acc(k, pa, pb, 9) === accp
        end
    end

    @testset "instruction selection: vfmaddsub, no separate mul/add/sub, no spill" begin
        # Exact instruction counts in the hot loop of `add_tile`, so only on
        # an FMA3 x86 host and not on hosted CI, whose virtualized CPU feature
        # sets do not reliably match.
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
                    io, QuasiStrided.add_tile,
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

    @testset "pack! InterleavedFormat A: scalar loop and fast path vs local layout" begin
        # Bitwise (`isequal` separates -0.0): packing is a copy, or a sign flip under conj.
        for T in (ComplexF64, ComplexF32), (MR, NR, W) in (kernel_shapes(T, FMAddSubMethod())..., (3, 2, 2))
            R = real(T)
            k = FMAddSubKernel(Val(MR), Val(NR), T, Val(W))
            k_block_length, lda, base = 7, MR + 3, 2
            vals = [T(10i + 1, -(10i + 2)) for i in 1:(lda * k_block_length + 4)]
            vals[5] = T(0, 0)
            vals[9] = T(R(-0.0), R(0))
            for f in (identity, conj), m in (MR, max(1, MR - 1))
                src = Tile(vals, base, AffineAxis(0, 1, m), AffineAxis(0, lda, k_block_length))
                A = zeros(T, MR, k_block_length)
                A[1:m, :] = f.(reshape(vals[(base + 1):(base + lda * k_block_length)], lda, k_block_length)[1:m, :])
                want = mk_pack_a(k, A)
                bufp = fill(R(-777), packed_a_length(k, k_block_length))
                pack!(bufp, src, sliver_spec(k, 1), f)
                @test isequal(bufp, want)
            end
            src = Tile(vals, base, AffineAxis(0, 1, MR), AffineAxis(0, lda, k_block_length))
            @test _QSF.complex_contiguous_eligible(src, sliver_spec(k, 1), identity, MR) ==
                _QSF.complex_fastpath_isa_eligible()
        end
    end

    @testset "blocking and selection" begin
        for T in (ComplexF64, ComplexF32)
            bf = default_blocking(FMAddSubKernel(Val(8), Val(8), T, Val(8)))
            bp = default_blocking(PlanarKernel(Val(8), Val(8), T, Val(8)))
            @test (bf.m_block, bf.k_block, bf.n_block) == (bp.m_block, bp.k_block, bp.n_block)  # planar's packed reals
            for s in kernel_shapes(T, FMAddSubMethod())
                @test _kernel_from_shape(s, T, FMAddSubMethod()) isa FMAddSubKernel{s[1], s[2], T, s[3]}
            end
            @test_throws ArgumentError _kernel_from_shape((7, 7, 7), T, FMAddSubMethod())
            @test !(_QSF._default_kernel(T, 1024, 1024) isa FMAddSubKernel)
        end
    end

    mk_e2e(ComplexF64, FMAddSubKernel(Val(4), Val(5), ComplexF64, Val(4)))
    mk_e2e(ComplexF32, FMAddSubKernel(Val(8), Val(5), ComplexF32, Val(8)))

    @testset "end to end: permuted/strided tensor contraction" begin
        T = ComplexF64
        rng = MersenneTwister(99)
        A = rand(rng, T, 5, 9, 7)  # a, k, b: non-unit-stride M composite
        B = rand(rng, T, 6, 9)     # n, k: transposed
        C = zeros(T, 5, 6, 7)
        plan = plan_contract(
            StridedView(C), StridedView(A), (1, 2, 3), StridedView(B), (4, 2), (1, 4, 3);
            kernel = FMAddSubKernel(Val(12), Val(8), T, Val(8))
        )
        execute!(plan, one(T), zero(T))
        want = [sum(A[a, k, b] * B[n, k] for k in 1:9) for a in 1:5, n in 1:6, b in 1:7]
        @test maximum(abs, C .- want) <= 64 * mk_tol(T)
    end
end
