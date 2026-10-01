# The microkernel contract shared by every kernel (`mk_contract`), plus the
# ScalarKernel runs. Each kernel file calls `mk_contract` on its own shapes.
#
# The oracles are test-local: packed panels are written out literally from each
# format's layout and the accumulator is read back from its documented layout,
# never through the engine's packers or offset accessors.

using Random
using QuasiStrided: SIMDKernel, PlanarKernel, OneMKernel, FMAddSubKernel
using SIMD: Vec
using StridedViews: StridedView

# `[x0, y0, x1, y1, ...]`
mk_ilv(x, y) = vec(permutedims(hcat(x, y)))
mk_cols(f, M) = reduce(vcat, [f(c) for c in eachcol(M)])

# Packed A (MR x k_block_length) and B (k_block_length x NR), one K step after another.
mk_pack_a(::Union{ScalarKernel, SIMDKernel}, A) = vec(A)
mk_pack_a(::PlanarKernel, A) = mk_cols(c -> [real(c); imag(c)], A)
mk_pack_a(::OneMKernel, A) =
    mk_cols(c -> [mk_ilv(real(c), imag(c)); mk_ilv(-imag(c), real(c))], A)
mk_pack_a(::FMAddSubKernel, A) = mk_cols(c -> mk_ilv(real(c), imag(c)), A)
mk_pack_b(::Union{ScalarKernel, SIMDKernel}, B) = vec(permutedims(B))
mk_pack_b(k, B) = mk_cols(c -> [real(c); imag(c)], permutedims(B))
mk_pack(k, A, B) = (mk_pack_a(k, A), mk_pack_b(k, B))

# Element (i, j), zero-based, of an accumulator.
mk_read(::ScalarKernel, acc, i, j) = acc[i + 1, j + 1]
function mk_read(k::SIMDKernel, acc, i, j)
    W = lanewidth(k)
    return acc[i ÷ W + (tile_size(k, 1) ÷ W) * j + 1][i % W + 1]
end
function mk_read(k::PlanarKernel, acc, i, j)
    W = lanewidth(k)
    idx = i ÷ W + (tile_size(k, 1) ÷ W) * j + 1
    return Complex(acc[idx][i % W + 1], acc[length(acc) ÷ 2 + idx][i % W + 1])
end
function mk_read(k::Union{OneMKernel, FMAddSubKernel}, acc, i, j)
    W = lanewidth(k)
    v, u = divrem(i, W ÷ 2)
    vec = acc[v + (2 * tile_size(k, 1) ÷ W) * j + 1]
    return Complex(vec[2u + 1], vec[2u + 2])
end

mk_fill_acc(k, x) = (acc = zero_accumulator(k); acc isa AbstractMatrix ? fill(x, size(acc)) : map(v -> typeof(v)(x), acc))
mk_nan(T) = T <: Complex ? T(NaN, NaN) : T(NaN)
mk_inf(T) = T <: Complex ? T(Inf, Inf) : T(Inf)
mk_tol(T) = real(T) === Float64 ? 1.0e-11 : 2.0f-4
mk_close(got, want, T) = all(abs.(got .- want) .<= mk_tol(T) .* max.(1, abs.(want)))
mk_alphabeta(T) = T <: Complex ?
    ((one(T), zero(T)), (T(2.5, -1), T(-1.75, 0.5)), (one(T), one(T)), (zero(T), T(3, 1)), (T(1, 1), T(2, 0))) :
    ((one(T), zero(T)), (T(2.5), T(-1.75)), (one(T), one(T)), (zero(T), T(3)), (T(1.5), T(2)))

# The storage the driver hands `store_tile!`: `Memory{T}` on Julia >= 1.11.
mk_dense(v::AbstractVector{T}) where {T} = copyto!(parent(StridedView(zeros(T, length(v)))), v)

# Element types of the test operands A and B; the mixed-domain kernels differ.
mk_optypes(k) = (scalartype(k), scalartype(k))

mk_run_acc(k, pa, pb, k_block_length) = accumulate(k, zero_accumulator(k), pa, pb, k_block_length)
mk_run_exec(k, dst, pa, pb, k_block_length, alpha, beta) = (execute_tile!(k, dst, pa, pb, k_block_length, alpha, beta); nothing)

# `full = false` checks only accumulate against the reference and allocations.
# Separate functions, so the light check does not compile the full one.
function mk_contract(k; full::Bool = true)
    return @testset "$(nameof(typeof(k))){$(tile_size(k, 1)),$(tile_size(k, 2)),$(scalartype(k))}" begin
        mk_contract_light(k)
        full && mk_contract_full(k)
    end
end

function mk_contract_light(k)
    T = scalartype(k)
    TA, TB = mk_optypes(k)
    MR, NR = tile_size(k)
    rng = MersenneTwister(100MR + NR)
    k_block_length = 5
    A = rand(rng, TA, MR, k_block_length)
    B = rand(rng, TB, k_block_length, NR)
    pa, pb = mk_pack(k, A, B)
    @test (length(pa), length(pb)) == (packed_a_length(k, k_block_length), packed_b_length(k, k_block_length))
    acc = mk_run_acc(k, pa, pb, k_block_length)
    @test mk_close([mk_read(k, acc, i, j) for i in 0:(MR - 1), j in 0:(NR - 1)], A * B, T)

    # Allocation-free, including a scattered destination and a dense one
    # with a partial row block. Julia 1.10 does not keep the tuple
    # accumulator in registers.
    m = max(1, MR - 1)
    if !(k isa ScalarKernel)
        skip = VERSION < v"1.11"
        ab = (T(2), T(0.5))
        scat = Tile(zeros(T, MR * NR), 0, view(collect(0:(MR - 1)), 1:MR), AffineAxis(0, MR, NR))
        part = Tile(mk_dense(zeros(T, m * NR)), 0, AffineAxis(0, 1, m), AffineAxis(0, m, NR))
        mk_run_acc(k, pa, pb, k_block_length)
        @test (@allocated mk_run_acc(k, pa, pb, k_block_length)) == 0 skip = skip
        for dst in (scat, part)
            mk_run_exec(k, dst, pa, pb, k_block_length, ab...)
            @test (@allocated mk_run_exec(k, dst, pa, pb, k_block_length, ab...)) == 0 skip = skip
        end
    end
    return nothing
end

function mk_contract_full(k)
    T = scalartype(k)
    TA, TB = mk_optypes(k)
    R = real(T)
    MR, NR = tile_size(k)
    rng = MersenneTwister(100MR + NR + 1)
    m = max(1, MR - 1)
    k_block_length = 5
    A = rand(rng, TA, MR, k_block_length)
    B = rand(rng, TB, k_block_length, NR)
    pa, pb = mk_pack(k, A, B)
    acc = mk_run_acc(k, pa, pb, k_block_length)

    acc0 = zero_accumulator(k)
    @test all(iszero(mk_read(k, acc0, i, j)) for i in 0:(MR - 1), j in 0:(NR - 1))
    @test accumulate(k, acc0, R[], R[], 0) === acc0
    @test_throws ArgumentError accumulate(k, acc0, R[], R[], -1)
    # Splitting k_block_length across calls composes.
    la, lb = length(pa) ÷ k_block_length, length(pb) ÷ k_block_length
    accs = zero_accumulator(k)
    for p in 0:(k_block_length - 1)
        accs = accumulate(k, accs, view(pa, (p * la + 1):((p + 1) * la)), view(pb, (p * lb + 1):((p + 1) * lb)), 1)
    end
    @test all(mk_read(k, accs, i, j) ≈ mk_read(k, acc, i, j) for i in 0:(MR - 1), j in 0:(NR - 1))

    # alpha*A*B + beta*C on the vector-store and scalar-store paths, every
    # alpha/beta branch, partial tiles; old C is NaN whenever beta == 0.
    pad, sentinel = 2, T(-77)
    extents = unique(((MR, NR), (1, 1), (m, NR), (MR, max(1, NR - 1)), (MR ÷ 2 + 1, min(2, NR)), (0, 0)))
    for k_block_length in (0, 1, 5), (alpha, beta) in mk_alphabeta(T), (m, n) in extents, scattered in (false, true)
        A = rand(rng, TA, MR, k_block_length)
        B = rand(rng, TB, k_block_length, NR)
        pa, pb = k_block_length == 0 ? (R[], R[]) : mk_pack(k, A, B)
        Cold = iszero(beta) ? fill(mk_nan(T), m * n) : rand(rng, T, m * n)
        storage = [fill(sentinel, pad); Cold; fill(sentinel, pad)]
        scattered || (storage = mk_dense(storage))
        rows = scattered ? view(collect(0:(m - 1)), 1:m) : AffineAxis(0, 1, m)
        execute_tile!(k, Tile(storage, pad, rows, AffineAxis(0, m, n)), pa, pb, k_block_length, alpha, beta)
        AB = (A * B)[1:m, 1:n]
        want = iszero(beta) ? alpha .* vec(AB) : alpha .* vec(AB) .+ beta .* Cold
        @test mk_close(storage[(pad + 1):(pad + m * n)], want, T)
        @test all(==(sentinel), storage[[1:pad; (pad + m * n + 1):(2pad + m * n)]])
    end

    # Nonfinite padding lanes (rows >= m, columns >= n) never reach C.
    m, n = max(1, MR - 1), max(1, NR - 1)
    A = rand(rng, TA, MR, 2)
    B = rand(rng, TB, 2, NR)
    A[(m + 1):end, :] .= mk_inf(TA)
    B[:, (n + 1):end] .= mk_inf(TB)
    pa, pb = mk_pack(k, A, B)
    acc = mk_run_acc(k, pa, pb, 2)
    @test any(!isfinite(mk_read(k, acc, i, j)) for i in 0:(MR - 1), j in 0:(NR - 1))
    for (storage, rows) in ((mk_dense(fill(mk_nan(T), m * n)), AffineAxis(0, 1, m)), (fill(mk_nan(T), m * n), view(collect(0:(m - 1)), 1:m)))
        execute_tile!(k, Tile(storage, 0, rows, AffineAxis(0, m, n)), pa, pb, 2, one(T), zero(T))
        @test mk_close(storage, vec(A[1:m, :] * B[:, 1:n]), T)
    end

    # alpha == 0 never reads acc or the panels; k_block_length == 0 never reads the panels.
    fulltile(s) = Tile(s, 0, AffineAxis(0, 1, MR), AffineAxis(0, MR, NR))
    storage = mk_dense(fill(T(2), MR * NR))
    store_tile!(fulltile(storage), mk_fill_acc(k, R(NaN)), zero(T), T(3), k)
    @test all(==(T(6)), storage)
    execute_tile!(k, fulltile(storage), R[], R[], 5, zero(T), zero(T))
    @test all(iszero, storage)
    empty = Tile(T[], 0, AffineAxis(0, 1, 0), AffineAxis(0, 0, 0))
    @test store_tile!(empty, acc, one(T), zero(T), k) === empty
    @test execute_tile!(k, empty, pa, pb, 2, one(T), one(T)) === empty

    # Rejected inputs.
    st = zeros(T, (MR + 1) * (NR + 1))
    @test_throws ArgumentError execute_tile!(k, Tile(st, 0, AffineAxis(0, 1, MR + 1), AffineAxis(0, MR + 1, NR)), pa, pb, 2, one(T), zero(T))
    @test_throws ArgumentError execute_tile!(k, Tile(st, 0, AffineAxis(0, 1, MR), AffineAxis(0, MR, NR + 1)), pa, pb, 2, one(T), zero(T))
    @test_throws ArgumentError execute_tile!(k, fulltile(st), pa, pb, -1, one(T), zero(T))
    @test_throws DimensionMismatch execute_tile!(k, fulltile(st), pa[1:(end - 1)], pb, 2, one(T), zero(T))
    @test_throws DimensionMismatch execute_tile!(k, fulltile(st), pa, pb[1:(end - 1)], 2, one(T), zero(T))

    # Destination layouts: each lands on exactly its own addresses.
    A = rand(rng, TA, MR, k_block_length)
    B = rand(rng, TB, k_block_length, NR)
    pa, pb = mk_pack(k, A, B)
    AB = A * B
    ld = m + 3
    perm = randperm(rng, ld)[1:m] .- 1
    cperm = ld .* (randperm(rng, n) .- 1)
    nanbuf(len) = fill(mk_nan(T), len)
    layouts = (
        # (storage, base, rows, cols, zero-based address of (i, j))
        (nanbuf(ld * n), 0, view(perm, 1:m), AffineAxis(0, ld, n), (i, j) -> perm[i + 1] + ld * j),
        (mk_dense(nanbuf(ld * n)), (m - 1) + ld * (n - 1), AffineAxis(0, -1, m), AffineAxis(0, -ld, n), (i, j) -> (m - 1 - i) + ld * (n - 1 - j)),
        (mk_dense(nanbuf(2ld * n)), 0, AffineAxis(0, 2, m), AffineAxis(0, 2ld, n), (i, j) -> 2i + 2ld * j),
        (mk_dense(nanbuf(ld * n)), 0, AffineAxis(0, 1, m), AffineAxis(0, ld, n), (i, j) -> i + ld * j),
        (mk_dense(nanbuf(ld * n)), 0, AffineAxis(0, 1, m), view(cperm, 1:n), (i, j) -> i + cperm[j + 1]),
        (view(nanbuf(ld * n + 4), 3:(ld * n + 2)), 0, AffineAxis(0, 1, m), AffineAxis(0, ld, n), (i, j) -> i + ld * j),
    )
    for (storage, base, rows, cols, addr) in layouts
        execute_tile!(k, Tile(storage, base, rows, cols), pa, pb, k_block_length, one(T), zero(T))
        got = [storage[addr(i, j) + 1] for i in 0:(m - 1), j in 0:(n - 1)]
        @test mk_close(got, AB[1:m, 1:n], T)
        @test count(!isnan, storage) == m * n
    end
    return nothing
end

# A named-kernel plan end to end: several K panels, tiles straddling both ways,
# conjugation.
function mk_e2e(T, kernel)
    rng = MersenneTwister(4242)
    M, N, K = 37, 23, 41
    A = rand(rng, T, M, K)
    B = rand(rng, T, K, N)
    Cinit = rand(rng, T, M, N)
    alpha, beta = T <: Complex ? (T(1.5, -0.25), T(-0.75, 0.5)) : (T(1.5), T(-0.75))
    conjs = T <: Complex ? ((false, false), (true, false), (false, true)) : ((false, false),)
    return @testset "end to end $(kernel === nothing ? "default kernel" : nameof(typeof(kernel))) $T" begin
        for (cA, cB) in conjs
            C = copy(Cinit)
            plan = plan_contract(
                StridedView(C), StridedView(A), (1, 2), StridedView(B), (2, 3), (1, 3);
                kernel, k_block = 16, conjA = cA, conjB = cB
            )
            execute!(plan, alpha, beta)
            want = alpha .* ((cA ? conj.(A) : A) * (cB ? conj.(B) : B)) .+ beta .* Cinit
            @test maximum(abs, C .- want) <= 64 * mk_tol(T) * maximum(abs, want)
        end
    end
end

@testset "ScalarKernel" begin
    for (MR, NR, T) in ((4, 3, Float64), (2, 2, Float32))
        mk_contract(ScalarKernel(Val(MR), Val(NR), T))
    end
end
