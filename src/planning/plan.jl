# The plan: everything `execute!` needs, resolved once so it can be reused.
# `plan_contract` runs the planning stages (labels, conjugation, kernel
# selection, blocking, execution path) and sizes the `ContractWorkspace`.

"""
    ContractPlan

Reusable, concretely typed plan from [`plan_contract`](@ref): the M/N/K
`AxisGroup`s, kernel, execution path, operand storages, effective
[`Blocking`](@ref), packing transforms and the [`ContractWorkspace`](@ref)
holding every buffer [`execute!`](@ref) needs. Reusing a plan reuses its
buffers. A plan, and so its workspace, must be used by one task at a time.
Field layout is not part of the public interface; after an M/N orientation
swap the `A*` fields describe the original `B`.
"""
struct ContractPlan{
        T, Kern, P, GM <: AxisGroup, GN <: AxisGroup, GK <: AxisGroup, SA, SB, SC,
        TA, TB, VT <: AbstractVector, PT <: AbstractVector,
    }
    kernel::Kern
    # The singleton selecting the `execute_path!` method (src/execution/paths.jl).
    path::P
    mgroup::GM
    ngroup::GN
    kgroup::GK
    blocking::Blocking
    Astorage::SA
    Abase::Int
    Bstorage::SB
    Bbase::Int
    Cstorage::SC
    Cbase::Int

    # GUARDRAIL: `identity`/`conj` as singleton values, not a `Bool`, which
    # would cost a dynamic dispatch at the pack site.
    atransform::TA
    btransform::TB

    workspace::ContractWorkspace{T, VT, PT}

    # Line-by-line packing of A (`mpack`) and B (`npack`); the groups above keep
    # their natural order, and the nest path enumerates the split ones.
    mpack::PackSplit
    npack::PackSplit
end

# `p` with some fields replaced.
@inline ContractPlan(
    p::ContractPlan; path = p.path, mgroup = p.mgroup, ngroup = p.ngroup,
    Astorage = p.Astorage, Abase = p.Abase, Bstorage = p.Bstorage, Bbase = p.Bbase,
    Cstorage = p.Cstorage, Cbase = p.Cbase, atransform = p.atransform, btransform = p.btransform,
    mpack = p.mpack, npack = p.npack
) = ContractPlan(
    p.kernel, path, mgroup, ngroup, p.kgroup, p.blocking, Astorage, Abase, Bstorage, Bbase,
    Cstorage, Cbase, atransform, btransform, p.workspace, mpack, npack
)

# Internal, for tests and benchmarks: `:never` rules out the dot, outer-product
# or unpacked-B path, `:always` forces unpacked B where the kernel admits it.
Base.@kwdef struct PathModes
    dot::Symbol = :auto
    outer::Symbol = :auto
    unpacked_b::Symbol = :auto
end

"""
    plan_contract(C::StridedView, A::StridedView, indA::NTuple{NA,Int},
                  B::StridedView, indB::NTuple{NB,Int},
                  indC::NTuple{NC,Int};
                  kernel = nothing,
                  conjA = false, conjB = false,
                  m_block = nothing, k_block = nothing, n_block = nothing,
                  allocator = TensorOperations.DefaultAllocator(),
                  accumulator = nothing) -> ContractPlan

Plan `C[indC] = A[indA] * B[indB]` (every label in exactly two operands):
resolve the labels into M/N/K `AxisGroup`s, validate axis lengths and eltypes,
choose the kernel and blocking, and preallocate every buffer
[`execute!`](@ref) needs. Throws `ArgumentError`/`DimensionMismatch` on
invalid input; `C` must not share memory with `A` or `B`.

  * Each operand's eltype is one of `Float32`, `Float64`, `ComplexF32`,
    `ComplexF64`; a complex `A` or `B` needs a complex `C`. The compute type
    `T` is `promote_type` of the three, or, for `accumulator = Float32` or
    `Float64`, that precision in the domain (real or complex) of the promoted
    type. Operands are converted to `T` on load and `alpha*AB + beta*C` is
    evaluated in `T`, rounding to `eltype(C)` once. For an `eltype(C)` of
    lower precision than `T`, a K longer than `k_block` accumulates in a panel
    of `T` in the workspace, of `M * min(N, n_block)` elements.
  * `kernel = nothing` picks one from the hardware profile and the extents: a
    [`SIMDKernel`](@ref) for a real `T`, a [`PlanarKernel`](@ref) for a
    complex one (an [`FMAddSubKernel`](@ref) for a short M on AVX-512), a
    [`ComplexRealKernel`](@ref)/[`RealComplexKernel`](@ref) for a complex `T`
    with a real `B`/`A`. [`OneMKernel`](@ref) is used only when named.
  * Labels within the M and N composites are ordered by their stride in `C`;
    the K order follows a cost model of the two packs. For a real `T` the
    operand roles are swapped (B feeds M) when only the N side gives `C` a
    unit-stride run long enough for the kernel's register tile; the result is
    the same either way.
  * `m_block`/`k_block`/`n_block` override the fields of
    `default_blocking(kernel)` (each `>= 1`); `m_block`/`n_block` are rounded
    up to multiples of the tile size and all three are clamped to the extents.
  * An operand whose register slivers would read one element per cache line,
    with the lines' other elements needed only after more lines than fit L2,
    is packed line by line (`PackSplit`); its `m_block` (or `n_block`) then
    becomes a whole number of line groups, up to `requested k_block / k_block`
    times the request.
  * The plan's [`ContractWorkspace`](@ref) takes its packed panels from the
    TensorOperations `allocator`; for a non-default one, [`release!`](@ref)
    is left to the caller.
  * `conjA`/`conjB` conjugate A's/B's elements (never `alpha`/`beta`) and
    compose by XOR with a view's own `conj`/`adjoint` `op`. A conjugated `C`
    is rejected.
"""
function plan_contract(
        C::StridedView, A::StridedView, indA::NTuple{NA, Int},
        B::StridedView, indB::NTuple{NB, Int},
        indC::NTuple{NC, Int};
        kernel = nothing,
        conjA::Bool = false,
        conjB::Bool = false,
        m_block::Union{Int, Nothing} = nothing,
        k_block::Union{Int, Nothing} = nothing,
        n_block::Union{Int, Nothing} = nothing,
        allocator = TO.DefaultAllocator(),
        accumulator::Union{Nothing, Type{Float32}, Type{Float64}} = nothing,
        path_modes::PathModes = PathModes()
    ) where {NA, NB, NC}
    return planned(
        C, A, indA, B, indB, indC, nothing, nothing;
        kernel, conjA, conjB, m_block, k_block, n_block, allocator, accumulator, path_modes
    )
end

# Everything planning needs that is concretely typed before the kernel is
# known (`run` is the chosen M composite's unit-stride run in C). `T` is the
# phantom compute type; `alpha`/`beta` are `nothing` to build the plan only.
struct PlanRequest{
        T, S <: Union{T, Nothing}, GM <: AxisGroup, GN <: AxisGroup, GK <: AxisGroup, SA, SB, SC, AL,
    }
    alpha::S
    beta::S
    mgroup::GM
    ngroup::GN
    kgroup::GK
    Astorage::SA
    Abase::Int
    Bstorage::SB
    Bbase::Int
    Cstorage::SC
    Cbase::Int
    run::Int
    m_block::Union{Int, Nothing}
    k_block::Union{Int, Nothing}
    n_block::Union{Int, Nothing}
    allocator::AL
    modes::PathModes
end

@inline function plan_request(
        ::Type{T}, alpha::S, beta::S, mgroup::GM, ngroup::GN, kgroup::GK,
        Astorage::SA, Abase::Int, Bstorage::SB, Bbase::Int, Cstorage::SC, Cbase::Int,
        run::Int, m_block, k_block, n_block, allocator::AL, modes::PathModes
    ) where {T, S, GM, GN, GK, SA, SB, SC, AL}
    return PlanRequest{T, S, GM, GN, GK, SA, SB, SC, AL}(
        alpha, beta, mgroup, ngroup, kgroup, Astorage, Abase, Bstorage, Bbase, Cstorage, Cbase,
        run, m_block, k_block, n_block, allocator, modes
    )
end

# Conjugation: folding `conjA`/`conjB` with each view's `.op`. The engine never
# indexes through a `StridedView`, so an unfolded `.op` would be silently dropped.

# GUARDRAIL: a TOTAL table with a throwing fallback, not an `op === conj` test,
# which classifies a directly constructed `adjoint` view as unconjugated.
op_conjugates(::typeof(identity)) = false
op_conjugates(::typeof(conj)) = true
# Elementwise on a `Number`, these are identity/conj; the axes are already resolved.
op_conjugates(::typeof(transpose)) = false
op_conjugates(::typeof(adjoint)) = true
@noinline op_conjugates(f) = throw(
    ArgumentError(
        "unsupported StridedView.op $f: QuasiStrided folds a view's `op` into the " *
            "packing transform and recognizes only identity/conj/transpose/adjoint"
    )
)

# GUARDRAIL: `⊻`, not `||`: the flag and the view's `op` are independent
# conjugations and `conj` is involutive. Always `false` for real `T`, so the
# real path never gets a `conj` specialization.
isconj(v::StridedView{T}, flag::Bool) where {T} =
    (T <: Complex) && (flag ⊻ op_conjugates(v.op))

# `plan_contract`'s body. With `alpha`/`beta` (of the compute type) the plan
# is executed and released behind the kernel barrier, where its type is
# concrete, and `nothing` is returned; with `nothing` the plan is returned.
function planned(
        C::StridedView, A::StridedView, indA::NTuple{NA, Int},
        B::StridedView, indB::NTuple{NB, Int}, indC::NTuple{NC, Int},
        alpha::S, beta::S;
        kernel = nothing, conjA::Bool = false, conjB::Bool = false,
        m_block::Union{Int, Nothing} = nothing, k_block::Union{Int, Nothing} = nothing,
        n_block::Union{Int, Nothing} = nothing, allocator = TO.DefaultAllocator(),
        accumulator::AC = nothing, path_modes::PathModes = PathModes()
    ) where {NA, NB, NC, S, AC}
    T = compute_type(eltype(A), eltype(B), eltype(C), accumulator)
    K = default_kernel_type(T, eltype(A), eltype(B))
    check_kernel_domain(kernel, eltype(A), eltype(B))

    # GUARDRAIL: a conjugated `C` is rejected; the engine writes through to
    # its parent, so there is nowhere to absorb its `op`.
    isconj(C, false) && throw(
        ArgumentError(
            "plan_contract: cannot write into a conjugated view (C has op $(C.op)); " *
                "writing a conjugated output is not supported"
        )
    )
    (Base.mightalias(C, A) || Base.mightalias(C, B)) &&
        throw(ArgumentError("plan_contract: C must not share memory with A or B"))
    atransform = isconj(A, conjA) ? conj : identity
    btransform = isconj(B, conjB) ? conj : identity

    mlabels, nlabels, klabels = classify_labels(indA, indB, indC)

    morder = order_free_labels(mlabels, indC, C)
    norder = order_free_labels(nlabels, indC, C)

    # Only real `T` swaps and only real kernels run-demote; `0` is a placeholder.
    run_m = T <: Real || K isa MixedKernel ? leading_unit_run(morder, indC, C) : 0
    run_n = T <: Real ? leading_unit_run(norder, indC, C) : 0

    mgroup = AxisGroup(morder, (indA, A), (indC, C))  # maps: (A, C)
    ngroup = AxisGroup(norder, (indB, B), (indC, C))  # maps: (B, C)

    m_length = axis_length(mgroup)
    n_length = axis_length(ngroup)

    korder = order_contract_labels(klabels, indA, A, morder, indB, B, norder, m_length, n_length)
    kgroup = AxisGroup(korder, (indA, A), (indB, B))  # maps: (A, B)

    k_length = axis_length(kgroup)

    m_tile_asis, m_tile_swapped = candidate_m_tiles(T, K, kernel, m_length, n_length, run_m, run_n)

    if T <: Real && prefer_swap(run_m, run_n, m_tile_asis, m_tile_swapped)
        # B takes the M role: groups, K maps, storages, run and transforms move
        # together; the sum is unchanged.
        kgroup_swapped = AxisGroup(korder, (indB, B), (indA, A))  # maps: (B, A)
        req_swapped = plan_request(
            T, alpha, beta, ngroup, mgroup, kgroup_swapped,
            parent(B), offset(B), parent(A), offset(A), parent(C), offset(C),
            run_n, m_block, k_block, n_block, allocator, path_modes
        )
        return plan_with_kernel(kernel, K, btransform, atransform, req_swapped)
    end
    req = plan_request(
        T, alpha, beta, mgroup, ngroup, kgroup,
        parent(A), offset(A), parent(B), offset(B), parent(C), offset(C),
        run_m, m_block, k_block, n_block, allocator, path_modes
    )
    return plan_with_kernel(kernel, K, atransform, btransform, req)
end

const SUPPORTED_ELTYPES = (Float32, Float64, ComplexF32, ComplexF64)

# A named mixed-domain kernel's `RealFormat` side packs a real operand only.
@inline check_kernel_domain(kernel, ::Type, ::Type) = nothing
@inline check_kernel_domain(kernel::ComplexRealKernel, ::Type, ::Type{TB}) where {TB} =
    TB <: Real || throw_kernel_domain(kernel, "B", TB)
@inline check_kernel_domain(kernel::RealComplexKernel, ::Type{TA}, ::Type) where {TA} =
    TA <: Real || throw_kernel_domain(kernel, "A", TA)

@noinline throw_kernel_domain(kernel, side, T) = throw(
    ArgumentError("plan_contract: $(typeof(kernel)) needs a real $side, got eltype $T")
)

# Fold to the compute type, or a throw, at compile time.
@inline function compute_type(::Type{TA}, ::Type{TB}, ::Type{TC}, ::Nothing) where {TA, TB, TC}
    (TA in SUPPORTED_ELTYPES && TB in SUPPORTED_ELTYPES && TC in SUPPORTED_ELTYPES) ||
        throw_eltypes(TA, TB, TC)
    (TC <: Real && !(TA <: Real && TB <: Real)) && throw_complex_into_real(TA, TB, TC)
    return promote_type(TA, TB, TC)
end
@inline compute_type(::Type{TA}, ::Type{TB}, ::Type{TC}, ::Type{R}) where {TA, TB, TC, R <: Union{Float32, Float64}} =
    compute_type(TA, TB, TC, nothing) <: Complex ? Complex{R} : R

@noinline throw_eltypes(TA, TB, TC) = throw(
    ArgumentError(
        "plan_contract: eltypes (A, B, C) = ($TA, $TB, $TC); each must be one of " *
            "Float32, Float64, ComplexF32, ComplexF64"
    )
)
@noinline throw_complex_into_real(TA, TB, TC) = throw(
    ArgumentError("plan_contract: a complex operand (A: $TA, B: $TB) needs a complex C, got $TC")
)

# `m_tile` of the kernel each orientation would run, for the swap decision: a
# named kernel either way, else `select_shape`'s pick at that orientation's
# extent and C run. The swap is judged before `fit_to_run`'s K-dependent
# divisor step, which an unbounded `k_length` skips.
@inline candidate_m_tiles(::Type{T}, ::Type, kernel, m_length::Int, n_length::Int, run_m::Int, run_n::Int) where {T} =
    (tile_size(kernel, 1), tile_size(kernel, 1))
@inline function candidate_m_tiles(::Type{T}, ::Type{K}, ::Nothing, m_length::Int, n_length::Int, run_m::Int, run_n::Int) where {T, K}
    m_tile_asis = select_shape(T, K, m_length, typemax(Int), run_m)[1][1]
    m_tile_swapped = T <: Real ? select_shape(T, K, n_length, typemax(Int), run_n)[1][1] : m_tile_asis
    return m_tile_asis, m_tile_swapped
end

# Kernel resolution. A named kernel goes straight through, never demoted. An
# automatic one is chosen as a `(shape, kernel type)` value. The blocking and
# execution path follow from the kernel type and tile, so they are resolved
# here, before the kernel exists; the plan is then built across a dispatch
# barrier specialised on the one concrete kernel and path, so only their code
# is compiled (holding the menu-wide kernel Union would box the request; a
# static ladder would compile every menu kernel). All barrier arguments are
# singletons or heap objects (the request in a `Ref`): one method-cache hit.
@inline function plan_with_kernel(kernel, ::Type, atransform, btransform, req::PlanRequest{T}) where {T}
    scalartype(kernel) === T ||
        throw(ArgumentError("kernel scalar type $(scalartype(kernel)) does not match the compute type $T"))
    return plan_across_barrier(kernel, typeof(kernel), tile_size(kernel)..., atransform, btransform, req)
end
@inline function plan_with_kernel(::Nothing, ::Type{K}, atransform, btransform, req::PlanRequest{T}) where {K, T}
    shape, kernel_type = select_shape(T, K, axis_length(req.mgroup), axis_length(req.kgroup), req.run)
    return plan_at_shape(shape, kernel_type, atransform, btransform, req)
end

# Nothing here is called on the kernel `Val`: a call union-split on the kernel
# type is emitted out of line and boxes `req`.
@inline plan_at_shape(shape, ::Val{K}, atransform, btransform, req::PlanRequest{T}) where {K, T} =
    plan_across_barrier(menu_val(shape, T, K), K, shape[1], shape[2], atransform, btransform, req)

@inline function plan_across_barrier(
        kernel, ::Type{K}, m_tile::Int, n_tile::Int, atransform, btransform, req::PlanRequest
    ) where {K}
    resolved = resolve_blocking(K, m_tile, n_tile, req)
    path = select_path(K, req, resolved)
    return Base.inferencebarrier(build_plan)(
        kernel, path, atransform, btransform, Base.RefValue((req, resolved))
    )
end

# The effective blocking for a kernel of type `K` with an `m_tile x n_tile`
# tile, its line-by-line packing (`mpack`/`npack`) and whether partial sums
# live in a panel of C (`panel`).
function resolve_blocking(::Type{K}, m_tile::Int, n_tile::Int, req::PlanRequest{T}) where {K, T}
    m_length = axis_length(req.mgroup)
    n_length = axis_length(req.ngroup)
    k_length = axis_length(req.kgroup)

    defaults = kernel_blocking(target_profile(), T, K, m_tile, n_tile)
    requested = Blocking(
        something(req.m_block, defaults.m_block),
        something(req.k_block, defaults.k_block),
        something(req.n_block, defaults.n_block)
    )

    # Empty extents: the drivers never read these; the floors keep them valid.
    m_block_rounded = roundup(requested.m_block, m_tile)
    n_block_rounded = roundup(requested.n_block, n_tile)
    m_block = m_length == 0 ? m_tile : min(m_block_rounded, roundup(m_length, m_tile))
    n_block = n_length == 0 ? n_tile : min(n_block_rounded, roundup(n_length, n_tile))
    k_block = k_length == 0 ? 1 : min(requested.k_block, k_length)
    panel = c_panel_needed(T, req.Cstorage, k_length, k_block)
    mpack = npack = NO_SPLIT
    # Cache lines hold each operand's storage eltype, and the block walk costs
    # what its packed format's scatter does. B is not split under a panel of C:
    # the panel holds N blocks in C's own N order, which a split N group does
    # not enumerate contiguously.
    if m_length > 0 && n_length > 0 && k_length > 0
        a_format, b_format = pack_formats(K)
        m_block, mpack = pack_split(
            req.mgroup, req.kgroup, 1, m_tile, a_format, sizeof(eltype(req.Astorage)),
            k_block, m_block, m_block_rounded, requested.k_block
        )
        if !panel
            n_block, npack = pack_split(
                req.ngroup, req.kgroup, 2, n_tile, b_format, sizeof(eltype(req.Bstorage)),
                k_block, n_block, n_block_rounded, requested.k_block
            )
        end
    end
    return (blocking = Blocking(m_block, k_block, n_block), mpack, npack, panel)
end

# Whether the partial sums between K blocks must live in a compute-type panel
# rather than in a C of narrower eltype. Static `false` unless C is narrower.
@inline function c_panel_needed(::Type{T}, Cstorage, k_length::Int, k_block::Int) where {T}
    sizeof(real(eltype(Cstorage))) < sizeof(real(T)) || return false
    return k_length > k_block
end

# The path `execute!` runs for a nonzero `alpha`: none for an empty C, scaling C
# for an empty K, else the dot path, the outer-product path or the nest (B
# packed or read in place).
function select_path(::Type{K}, req::PlanRequest{T}, resolved) where {K, T}
    (; mgroup, ngroup, kgroup, modes) = req
    (; mpack, npack, panel) = resolved
    m_length = axis_length(mgroup)
    n_length = axis_length(ngroup)
    k_length = axis_length(kgroup)
    (m_length == 0 || n_length == 0) && return EmptyPath()
    k_length == 0 && return ScalePath()
    W = vector_lanes(target_profile(), real(T))
    if !panel && modes.dot !== :never &&
            dot_applicable(T, req.Astorage, req.Bstorage, kgroup, m_length, n_length, k_length, W)
        return m_length == 1 ? lane_path(DotPath{true}, W) : lane_path(DotPath{false}, W)
    end
    k_length == 1 && modes.outer !== :never && outer_applicable(T, req.Astorage, req.Cstorage, mgroup, m_length, W) &&
        return lane_path(OuterPath, W)
    unpacked_b = reads_b_by_element(K) && unpacked_b_rule(modes.unpacked_b, mgroup, kgroup)
    return nest_path(unpacked_b, mgroup, ngroup, kgroup, is_split(mpack), is_split(npack), panel)
end

# The dot path (dot.jl): a degenerate free extent, a matrix operand with
# unit-ramp K in dense storage (raw-pointer loads), and at least one vector
# (`W` lanes) of K.
function dot_applicable(::Type{T}, Astorage, Bstorage, kgroup::AxisGroup, m_length::Int, n_length::Int, k_length::Int, W::Int) where {T}
    (m_length == 1 || n_length == 1) || return false
    k_length >= W || return false
    if m_length == 1
        Bstorage isa DenseVector{T} || return false
        map_ramp_step(kgroup, 2) == 1 || return false
    else
        Astorage isa DenseVector{T} || return false
        map_ramp_step(kgroup, 1) == 1 || return false
    end
    return true
end

# The outer-product path (outer.jl): real `T`, dense A and C with M a unit
# ramp in both, at least one vector of M. N and B may have any layout.
function outer_applicable(::Type{T}, Astorage, Cstorage, mgroup::AxisGroup, m_length::Int, W::Int) where {T}
    T <: Real || return false
    (Astorage isa DenseVector{T} && Cstorage isa DenseVector{T}) || return false
    m_length >= W || return false
    map_ramp_step(mgroup, 1) == 1 || return false
    return map_ramp_step(mgroup, 2) == 1
end

# Whether B is read in place (given a kernel that reads B by element): small
# M, and every B column contiguous along K. Packing pays off only through
# reuse across M slivers, while `n_tile` contiguous columns read in place cost
# the kernel nothing. A large K stride in B (each step its own cache line, and
# for a power of two only a few L1 sets) is what packing exists for. `mode`:
# `:always`/`:never` override the rule (`PathModes`).
@inline function unpacked_b_rule(mode::Symbol, mgroup::AxisGroup, kgroup::AxisGroup)
    mode === :always && return true
    mode === :never && return false
    axis_length(mgroup) <= UNPACKED_B_MMAX || return false
    (k_ramp, k_step) = affine_ramp(kgroup)
    return k_ramp && abs(k_step[2]) == 1
end

# Beyond this M the packed B's reuse wins.
const UNPACKED_B_MMAX = 256

# Type parameters on the transforms force specialisation on them: the
# compiler does not specialise on a `Function` argument it only passes on.
build_plan(::Val{Kern}, path, atransform::TA, btransform::TB, slot::Base.RefValue) where {Kern, TA, TB} =
    build_plan(Kern(), path, atransform, btransform, slot)

# The plan on a concrete kernel, path and transform pair (a named kernel's
# transform Unions die here).
function build_plan(
        kernel::Microkernel, path::P, atransform::TA, btransform::TB, slot::Base.RefValue{R}
    ) where {P, TA, TB, R}
    T = scalartype(kernel)
    req, (; blocking, mpack, npack, panel) = slot[]
    panel_length = panel ? axis_length(req.mgroup) * min(blocking.n_block, axis_length(req.ngroup)) : 0
    ws = ContractWorkspace(T, kernel, blocking; allocator = req.allocator, panel = panel_length, path)
    plan = ContractPlan(
        kernel, path, req.mgroup, req.ngroup, req.kgroup, blocking,
        req.Astorage, req.Abase, req.Bstorage, req.Bbase, req.Cstorage, req.Cbase,
        atransform, btransform, ws, mpack, npack
    )
    return execute_request(plan, req.alpha, req.beta, req.allocator)
end

execute_request(plan::ContractPlan, ::Nothing, ::Nothing, allocator) = plan
function execute_request(plan::ContractPlan{T}, alpha::T, beta::T, allocator) where {T}
    execute!(plan, alpha, beta)
    release!(plan, allocator)
    return nothing
end

release!(plan::ContractPlan, allocator) = release!(plan.workspace, allocator)
