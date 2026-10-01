# Tile axes and Tile addressing, against hand-computed addresses.

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
