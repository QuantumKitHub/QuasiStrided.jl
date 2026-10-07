include("helpers.jl")

using QuasiStrided: FMAddSubKernel, lanewidth, kernel_shapes, packed_panel,
    accumulator_planes, target_profile, PackedPanel
using SIMD: Vec
using InteractiveUtils: code_native

const _QSF = QuasiStrided

@testset "FMAddSubKernel" begin
    full = ((ComplexF64, (12, 8, 8)), (ComplexF32, (8, 5, 8)))
    for (T, (MR, NR, W)) in full
        mk_contract(FMAddSubKernel(Val(MR), Val(NR), T, Val(W)))
    end
    for T in (ComplexF64, ComplexF32), s in (kernel_shapes(T, FMAddSubKernel)..., (2, 3, 4))
        (T, s) in full || mk_contract(FMAddSubKernel(Val(s[1]), Val(s[2]), T, Val(s[3])); full = false)
    end

    @testset "construction" begin
        @test accumulator_planes(FMAddSubKernel) == 1
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
end
