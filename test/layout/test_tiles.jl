# Tile axes and QSTile addressing, against hand-computed addresses.

using QuasiStrided: PtrScatterAxis, QSTile, axis_offset, tile_offset, tile_load, tile_store!,
    checked_span_bounds, descriptor_offset_range

@testset "tile axes: addressing, borrowing, validation" begin
    @test [axis_offset(AffineAxis(10, 3, 5), t) for t in 0:4] == [10, 13, 16, 19, 22]
    @test [axis_offset(AffineAxis(20, -4, 4), t) for t in 0:3] == [20, 16, 12, 8]
    @test [axis_offset(AffineAxis(7, 0, 3), t) for t in 0:2] == [7, 7, 7]
    offs = [5, -3, 100, 0, 42]
    ax = ScatterAxis(offs, 3)                 # a strict prefix of `offs`
    @test axis_length(ax) == 3
    @test [axis_offset(ax, t) for t in 0:2] == offs[1:3]
    offs[2] = 999                              # borrowed, not copied
    @test axis_offset(ax, 1) == 999
    GC.@preserve offs begin
        pax = PtrScatterAxis(pointer(offs), 5)
        @test [axis_offset(pax, t) for t in 0:4] == offs
        @test axis_offset_range(pax) == (0, 999)
    end
    @test_throws ArgumentError AffineAxis(0, 1, -1)
    @test_throws ArgumentError ScatterAxis(offs, -1)
    @test_throws DimensionMismatch ScatterAxis(offs, 6)
    @test_throws ArgumentError PtrScatterAxis(pointer(offs), -1)
end

@testset "axis_from_descriptor: regular -> AffineAxis, irregular -> borrowed ScatterAxis" begin
    buf = [99, 99, 7, 10, 13, 16, -1]
    ax = axis_from_descriptor(describe_block(buf, 2, 4), buf, 2)
    @test ax isa AffineAxis && (ax.base, ax.stride, axis_length(ax)) == (7, 3, 4)
    bufi = [-1, -1, 0, 5, 1, 6, -1]
    d = describe_block(bufi, 2, 4)
    axi = axis_from_descriptor(d, bufi, 2)
    @test axi isa ScatterAxis && [axis_offset(axi, t) for t in 0:3] == bufi[3:6]
    @test descriptor_offset_range(d, bufi, 2) == axis_offset_range(axi) == (0, 6)
    bufi[3] = 555
    @test axis_offset(axi, 0) == 555
    @test axis_from_descriptor(describe_block(Int[], 0), Int[]) isa AffineAxis
end

@testset "QSTile: base + row_offset(i) + col_offset(j), every axis combination" begin
    rowoffs, coloffs = [0, 100, 5, -20], [3, -7, 42]
    for rows in (AffineAxis(2, 3, 4), ScatterAxis(rowoffs, 4)), cols in (AffineAxis(1, 10, 3), ScatterAxis(coloffs, 3))
        storage = zeros(200)
        t = SourceTile(storage, 50, rows, cols)
        @test (nrows(t), ncols(t)) == (4, 3)
        for i in 0:3, j in 0:2
            addr = 50 + (rows isa AffineAxis ? 2 + 3i : rowoffs[i + 1]) + (cols isa AffineAxis ? 1 + 10j : coloffs[j + 1])
            @test tile_offset(t, i, j) == addr
            tile_store!(t, i, j, 1000.0i + j)
            @test storage[addr + 1] == tile_load(t, i, j) == 1000.0i + j
        end
    end
end

@testset "storage bounds: offset ranges and span checks" begin
    @test axis_offset_range(AffineAxis(10, -3, 4)) == (1, 10)
    @test axis_offset_range(ScatterAxis([4, -2, 9], 2)) == (-2, 4)
    @test axis_offset_range(AffineAxis(5, 1, 0)) == (0, -1)
    @test_throws OverflowError axis_offset_range(AffineAxis(typemax(Int), 1, 2))
    storage = zeros(20)
    @test checked_tile_storage_bounds(SourceTile(storage, 0, AffineAxis(0, 1, 4), AffineAxis(0, 5, 4))) === nothing
    @test_throws BoundsError checked_tile_storage_bounds(SourceTile(storage, 2, AffineAxis(0, 1, 4), AffineAxis(0, 5, 4)))
    @test_throws BoundsError checked_tile_storage_bounds(SourceTile(storage, 0, AffineAxis(-1, 1, 4), AffineAxis(0, 5, 4)))
    # An empty axis reads nothing, whatever the other axis addresses.
    @test checked_tile_storage_bounds(SourceTile(storage, 0, AffineAxis(0, 1, 0), AffineAxis(500, 1, 3))) === nothing
    @test checked_span_bounds(0, (0, 9), (0, 10), 20) === nothing
    @test_throws BoundsError checked_span_bounds(0, (0, 9), (0, 11), 20)
    @test checked_span_bounds(-100, (0, -1), (0, 5), 20) === nothing
end
