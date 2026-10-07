# Fixtures and bindings shared by all test files. Every test file includes this
# file or its folder's helpers.jl, which includes it.

using Test
using Random
using QuasiStrided
using QuasiStrided: AxisGroup, axis_length, fill_offsets!, BlockDescriptor,
    describe_block, block_descriptors!, Descriptor, tile_size,
    scalartype, packed_a_offset, packed_b_offset, packed_a_length, packed_b_length,
    AffineAxis, Tile, checked_tile_storage_bounds, pack!, sliver_spec,
    zero_accumulator, add_tile, store_tile!, execute_tile!, lanewidth, contract!,
    Blocking, default_blocking, ScalarKernel, SIMDKernel, TargetProfile, CacheLevel,
    target_profile, cache_topology, unknown_target, detect_isa, detect_target,
    derived_shape, fallback_shape, shape_override, kernel_from_shape,
    fallback_blocking, kernel_shapes, parse_size, count_cpu_list, NR_DEFAULT,
    rule_applies, isa_nregisters, sliver_width, realtype,
    PlanarKernel, OneMKernel, FMAddSubKernel, accumulator_planes, pack_formats,
    reals_per_element, modelled_blocking, kernel_blocking,
    planar_pressure, pack_split, is_split, NestPath,
    plan_contract, execute!, ContractPlan
using StridedViews: StridedView, offset
import TensorOperations as TO

_path_of(plan) = plan.path

# Offsets of coordinate `q` of an `AxisGroup` in every map, by direct decoding.
function offsets(g::AxisGroup{D, P}, q::Int) where {D, P}
    c = Tuple(CartesianIndices(g.lengths)[q + 1])
    return ntuple(p -> sum((c[d] - 1) * g.strides[p][d] for d in 1:D; init = 0), P)
end

# Plan for the matmul C[m,n] = sum_k A[m,k]*B[k,n].
function _mm_plan(Cmat, Amat, Bmat; kwargs...)
    return plan_contract(
        StridedView(Cmat), StridedView(Amat), (1, 2), StridedView(Bmat), (2, 3), (1, 3);
        kwargs...
    )
end

# Steady-state allocation of one `run!(plan, 1.0, 0.0)` call, after a warm-up.
function _steady_allocs!(run!, plan, Cmat)
    run!(plan, 1.0, 0.0)
    fill!(Cmat, 0.0)
    return @allocated run!(plan, 1.0, 0.0)
end

# C[a,n,b] = sum_k A[a,k,b]*B[k,n] with sizes a=3, k=5, b=2, n=4 and labels
# a=1, k=2, b=3, n=4.
function _worked_fixture()
    A = reshape(collect(1.0:30.0), 3, 5, 2)
    B = reshape(collect(1.0:20.0), 5, 4)
    Cref = zeros(Float64, 3, 4, 2)
    for b in 1:2, n in 1:4, a in 1:3
        Cref[a, n, b] = sum(A[a, k, b] * B[k, n] for k in 1:5)
    end
    return A, B, Cref
end

const INDA = (1, 2, 3)
const INDB = (2, 4)
const INDC = (1, 4, 3)

const VALID_ISAS = (:avx512, :avx2, :neon, :unknown)
synthetic(isakey) = TargetProfile(isakey, "synthetic", CacheLevel(), CacheLevel(), CacheLevel())

# The kernel `plan_contract` picks for `T` on this host: at its own shape, and
# for an M extent of `m_length` with C contiguous.
host_kernel(::Type{T}) where {T} = kernel_from_shape(derived_shape(target_profile(), T), T)
function auto_kernel(::Type{T}, m_length::Int) where {T}
    shape, K = QuasiStrided.select_shape(T, QuasiStrided.default_kernel_type(T), m_length, 0, m_length)
    return kernel_from_shape(shape, T, K)
end

# Permuted A (with a zero-stride axis), negative-stride B, offset sliced C.
function scattered_fixture(::Type{T}, a_n = 32, k_n = 32, b_n = 8, n_n = 32) where {T}
    A2 = randn(MersenneTwister(11), T, a_n, k_n)
    Aperm = permutedims(StridedView(vec(A2), (a_n, k_n, b_n), (1, a_n, 0), 0), (2, 3, 1))
    Bneg = StridedView(randn(MersenneTwister(12), T, k_n * n_n), (k_n, n_n), (-1, k_n), k_n - 1)
    Cbig = zeros(T, a_n + 2, n_n + 3, b_n + 1)
    Cv = StridedView(view(Cbig, 2:(a_n + 1), 2:(n_n + 1), 1:b_n))
    return (Cv, Aperm, (2, 3, 1), Bneg, (2, 4), (1, 4, 3))
end

# Steady-state allocation of one `pack!` call. `::F where {F}` forces
# specialization on the pass-through `transform`; without it the dynamic call
# allocates on Julia 1.10.
function steady_pack_allocs(packed, source, spec, transform::F) where {F}
    pack!(packed, source, spec, transform)
    return @allocated pack!(packed, source, spec, transform)
end

# Engine-free reference: alpha * sum_K conj?(A) conj?(B) + beta * C, one loop
# over every label's range (`getindex` applies a view's `op`, the flag goes on top).
function _lo_reference(Cstart, Av, indA, Bv, indB, indC; conjA = false, conjB = false, alpha = 1, beta = 0)
    labels = unique((indA..., indB...))
    ext = Dict{Int, Int}()
    for (l, s) in zip(indA, size(Av))
        ext[l] = s
    end
    for (l, s) in zip(indB, size(Bv))
        ext[l] = s
    end
    T = eltype(Cstart)
    acc = zeros(T, size(Cstart))
    pA = map(l -> Int(findfirst(==(l), labels)), indA)
    pB = map(l -> Int(findfirst(==(l), labels)), indB)
    pC = map(l -> Int(findfirst(==(l), labels)), indC)
    dims = Tuple(ext[l] for l in labels)
    _lo_reference_loop!(acc, Av, pA, Bv, pB, pC, dims, conjA, conjB)
    return alpha .* acc .+ beta .* Cstart
end

function _lo_reference_loop!(
        acc, Av, pA::NTuple{NA, Int}, Bv, pB::NTuple{NB, Int}, pC::NTuple{NC, Int},
        dims::NTuple{NL, Int}, conjA::Bool, conjB::Bool
    ) where {NA, NB, NC, NL}
    for I in CartesianIndices(dims)
        a = Av[ntuple(d -> I[pA[d]], Val(NA))...]
        b = Bv[ntuple(d -> I[pB[d]], Val(NB))...]
        conjA && (a = conj(a))
        conjB && (b = conj(b))
        acc[ntuple(d -> I[pC[d]], Val(NC))...] += a * b
    end
    return acc
end

# Line-by-line packing. intensli_7, C[e,c,b,f,a] = A[a,b,c,d,e] * B[d,f], splits A
# (K steps of a page or more, every eltype); C[a1,au,f1,f2] = A[au,k,a1] * B[f2,k,f1]
# splits both operands (K steps within a page, real only).
const SP_I7 = ((1, 2, 3, 4, 5), (4, 6), (5, 3, 2, 6, 1))
const SP_BOTH = ((2, 3, 1), (5, 3, 4), (1, 2, 4, 5))

function _sp_views(T, (iA, iB, iC), ext, TA = T)
    arr(S, I) = StridedView(randn(S, map(l -> ext[l], I)))
    return (arr(T, iC), arr(TA, iA), iA, arr(T, iB), iB, iC)
end

# The plan with every structurally eligible group split, whatever this host's L2:
# the planner's decision at a zero L2 threshold, replanned at its block extents.
function _sp_forced_plan(Cv, Av, iA, Bv, iB, iC; kw...)
    p = plan_contract(Cv, Av, iA, Bv, iB, iC; kw...)
    k, k_block, d, T = p.kernel, p.blocking.k_block, default_blocking(p.kernel), eltype(Cv)
    function split(g, i, eff, req)
        R = tile_size(k, i)
        return pack_split(g, p.kgroup, i, R, pack_formats(typeof(k))[i], sizeof(T), k_block, eff, cld(req, R) * R, d.k_block; l2bytes = 0)
    end
    (m_block, ms) = split(p.mgroup, 1, p.blocking.m_block, something(get(kw, :m_block, nothing), d.m_block))
    (n_block, ns) = split(p.ngroup, 2, p.blocking.n_block, d.n_block)
    q = plan_contract(Cv, Av, iA, Bv, iB, iC; kw..., kernel = k, m_block, n_block)
    @assert q.mgroup == p.mgroup && q.ngroup == p.ngroup
    path = QuasiStrided.nest_path(false, q.mgroup, q.ngroup, q.kgroup, is_split(ms), is_split(ns))
    return ContractPlan(q; path, mpack = ms, npack = ns)
end
