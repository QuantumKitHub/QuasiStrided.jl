# Tile axes and Tile addressing, against hand-computed addresses.

include("../helpers.jl")

using QuasiStrided: checked_span_bounds, descriptor_offset_range, is_unit_stride

@testset "AffineAxis: offsets, validation" begin
    @test AffineAxis(10, 3, 5) == [10, 13, 16, 19, 22]
    @test AffineAxis(20, -4, 4) == [20, 16, 12, 8]
    @test AffineAxis(7, 0, 3) == [7, 7, 7]
    @test_throws BoundsError AffineAxis(7, 0, 3)[4]
    @test_throws ArgumentError AffineAxis(0, 1, -1)
    @test is_unit_stride(AffineAxis(3, 1, 2)) && !is_unit_stride(AffineAxis(3, 2, 2))
    @test !is_unit_stride(view([0, 1, 2], 1:3))
end

@testset "Tile: base + rows[i] + cols[j], every axis combination" begin
    rowoffs, coloffs = [0, 100, 5, -20], [3, -7, 42]
    for rows in (AffineAxis(2, 3, 4), view(rowoffs, 1:4)), cols in (AffineAxis(1, 10, 3), view(coloffs, 1:3))
        storage = zeros(200)
        t = Tile(storage, 50, rows, cols)
        @test size(t) == (4, 3)
        for i in 1:4, j in 1:3
            addr = 50 + rows[i] + cols[j]
            t[i, j] = 1000.0i + j
            @test storage[addr + 1] == t[i, j] == 1000.0i + j
        end
    end
    @test_throws BoundsError Tile(zeros(3), 0, AffineAxis(0, 1, 2), AffineAxis(0, 2, 2))[2, 2]
end

@testset "storage bounds: offset ranges and span checks" begin
    @test extrema(AffineAxis(10, -3, 4)) == (1, 10)
    @test_throws OverflowError extrema(AffineAxis(typemax(Int), 1, 2))
    @test_throws ArgumentError extrema(AffineAxis(5, 1, 0))
    buf = [99, 99, 7, 10, 13, 16, -1]
    @test descriptor_offset_range(describe_block(buf, 2, 4), buf, 2) == (7, 16)
    bufi = [-1, -1, 0, 5, 1, 6, -1]
    @test descriptor_offset_range(describe_block(bufi, 2, 4), bufi, 2) == (0, 6)
    @test descriptor_offset_range(describe_block(Int[], 0), Int[], 0) == (0, -1)
    storage = zeros(20)
    @test checked_tile_storage_bounds(Tile(storage, 0, AffineAxis(0, 1, 4), AffineAxis(0, 5, 4))) === nothing
    @test_throws BoundsError checked_tile_storage_bounds(Tile(storage, 2, AffineAxis(0, 1, 4), AffineAxis(0, 5, 4)))
    @test_throws BoundsError checked_tile_storage_bounds(Tile(storage, 0, AffineAxis(-1, 1, 4), AffineAxis(0, 5, 4)))
    @test_throws BoundsError checked_tile_storage_bounds(Tile(storage, 0, view([4, -2, 9], 1:2), AffineAxis(0, 5, 4)))
    # An empty axis reads nothing, whatever the other axis addresses.
    @test checked_tile_storage_bounds(Tile(storage, 0, AffineAxis(0, 1, 0), AffineAxis(500, 1, 3))) === nothing
    @test checked_span_bounds(0, (0, 9), (0, 10), 20) === nothing
    @test_throws BoundsError checked_span_bounds(0, (0, 9), (0, 11), 20)
    @test checked_span_bounds(-100, (0, -1), (0, 5), 20) === nothing
end

@testset "checked_span_bounds is equivalent to the per-tile check" begin
    # For a region split into slivers, checking the union range once accepts
    # exactly what checking each sliver accepts.
    Random.seed!(99)
    for trial in 1:150
        nrow = rand(1:9)
        ncol = rand(1:5)
        rowoffs = [rand(-20:20) for _ in 1:nrow]
        coloffs = [rand(-20:20) for _ in 1:ncol]
        base = rand(-5:30)
        len = rand(1:60)
        reg = rand(1:4)                      # sliver height
        nsliv = cld(nrow, reg)

        cols = view(coloffs, 1:ncol)
        persliver = true
        for s in 0:(nsliv - 1)
            first = s * reg
            cnt = min(reg, nrow - first)
            rows = view(rowoffs, (first + 1):(first + cnt))
            ok = try
                checked_tile_storage_bounds(base, rows, cols, len)
                true
            catch e
                e isa BoundsError || rethrow()
                false
            end
            persliver &= ok
        end

        blockrange = (minimum(rowoffs), maximum(rowoffs))
        colrange = (minimum(coloffs), maximum(coloffs))
        blockok = try
            QuasiStrided.checked_span_bounds(base, blockrange, colrange, len)
            true
        catch e
            e isa BoundsError || rethrow()
            false
        end
        @test blockok == persliver
    end

    @test QuasiStrided.checked_span_bounds(0, (0, -1), (0, 0), 1) === nothing
    @test QuasiStrided.checked_span_bounds(0, (0, 0), (0, -1), 1) === nothing
    @test_throws BoundsError QuasiStrided.checked_span_bounds(0, (0, 0), (0, 0), 0)
end

@testset "descriptor_offset_range agrees with extrema" begin
    Random.seed!(7)
    for trial in 1:100
        n = rand(1:8)
        buf = [rand(-30:30) for _ in 1:n]
        rand() < 0.4 && (buf = [3 + 5 * (t - 1) for t in 1:n])   # force a regular run
        d = describe_block(buf, 0, n)
        GC.@preserve buf @test QuasiStrided.descriptor_offset_range(d, buf, 0) == extrema(QuasiStrided.axis_of(d, buf, 0))
    end
    @test QuasiStrided.descriptor_offset_range(BlockDescriptor(0, 0, 0, true), Int[], 0) == (0, -1)
end

# `@inbounds` reaches the storage check only through an inlined call.
pcf_inbounds_pack!(args...) = @inbounds pack!(args...)
pcf_inbounds_execute_tile!(args...) = @inbounds execute_tile!(args...)

@testset "@inbounds skips only the storage bounds check" begin
    kernel = SIMDKernel(Val(8), Val(6), Float64)
    src = Tile(collect(1.0:64.0), 0, AffineAxis(0, 1, 8), AffineAxis(0, 8, 4))
    a = sliver_spec(kernel, 1)
    @test_throws DimensionMismatch pcf_inbounds_pack!(zeros(3), src, a, identity)
    @test_throws ArgumentError pcf_inbounds_pack!(zeros(Float32, 64), src, a, identity)
    toowide = Tile(collect(1.0:200.0), 0, AffineAxis(0, 1, 9), AffineAxis(0, 16, 4))
    @test_throws ArgumentError pcf_inbounds_pack!(zeros(200), toowide, a, identity)

    dest = Tile(zeros(8 * 6), 0, AffineAxis(0, 1, 8), AffineAxis(0, 8, 6))
    @test_throws DimensionMismatch pcf_inbounds_execute_tile!(
        kernel, dest, zeros(3), zeros(6 * 4), 4, 1.0, 0.0
    )
    @test_throws ArgumentError pcf_inbounds_execute_tile!(
        kernel, dest, zeros(8 * 4), zeros(6 * 4), -1, 1.0, 0.0
    )
end
