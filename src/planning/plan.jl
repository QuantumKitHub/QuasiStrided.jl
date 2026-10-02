# The plan: everything `execute!` needs, resolved once so it can be reused.
# `plan_contract` runs the planning stages (labels, conjugation, kernel
# selection, blocking) and sizes the `ContractWorkspace`.

"""
    ContractPlan

Reusable, concretely typed plan from [`plan_contract`](@ref): the M/N/K
`AxisGroup`s, kernel, operand storages, effective [`Blocking`](@ref), packing
transforms and the [`ContractWorkspace`](@ref) holding every buffer
[`execute!`](@ref) needs. Reusing a plan reuses its buffers. A plan, and so
its workspace, must be used by one task at a time. Field layout is not part
of the public interface; after an M/N orientation swap the `A*` fields
describe the original `B`.
"""
struct ContractPlan{
        T, Kern, GM <: AxisGroup, GN <: AxisGroup, GK <: AxisGroup, SA, SB, SC,
        TA, TB, VT <: AbstractVector, PT <: AbstractVector,
    }
    kernel::Kern
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
invalid input.

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
        accumulator::Union{Nothing, Type{Float32}, Type{Float64}} = nothing
    ) where {NA, NB, NC}
    return _planned(
        identity, C, A, indA, B, indB, indC,
        kernel, conjA, conjB, m_block, k_block, n_block, allocator, accumulator
    )
end

# Everything `_plan_contract` needs that is concretely typed before the kernel
# is known (`run` is the chosen M composite's unit-stride run in C), as one
# value that crosses the kernel barrier. `T` is the phantom compute type.
struct _PlanRequest{
        T, F, GM <: AxisGroup, GN <: AxisGroup, GK <: AxisGroup, SA, SB, SC, AL,
    }
    f::F
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
end

@inline function _plan_request(
        ::Type{T}, f::F, mgroup::GM, ngroup::GN, kgroup::GK,
        Astorage::SA, Abase::Int, Bstorage::SB, Bbase::Int, Cstorage::SC, Cbase::Int,
        run::Int, m_block, k_block, n_block, allocator::AL
    ) where {T, F, GM, GN, GK, SA, SB, SC, AL}
    return _PlanRequest{T, F, GM, GN, GK, SA, SB, SC, AL}(
        f, mgroup, ngroup, kgroup, Astorage, Abase, Bstorage, Bbase, Cstorage, Cbase,
        run, m_block, k_block, n_block, allocator
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

# `plan_contract`'s body, positional, with a continuation `f` applied to the
# plan inside the barrier, where its type is concrete (the TensorOperations
# adapter passes an executor).
function _planned(
        f::F, C::StridedView, A::StridedView, indA::NTuple{NA, Int},
        B::StridedView, indB::NTuple{NB, Int}, indC::NTuple{NC, Int},
        kernel, conjA::Bool, conjB::Bool,
        m_block::Union{Int, Nothing}, k_block::Union{Int, Nothing}, n_block::Union{Int, Nothing},
        allocator, accumulator::AC
    ) where {F, NA, NB, NC, AC}
    T = _compute_type(eltype(A), eltype(B), eltype(C), accumulator)
    K = default_kernel_type(T, eltype(A), eltype(B))
    _check_kernel_domain(kernel, eltype(A), eltype(B))

    # GUARDRAIL: a conjugated `C` is rejected; the engine writes through to
    # its parent, so there is nowhere to absorb its `op`.
    isconj(C, false) && throw(
        ArgumentError(
            "plan_contract: cannot write into a conjugated view (C has op $(C.op)); " *
                "writing a conjugated output is not supported"
        )
    )
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

    m_tile_asis, m_tile_swapped = _candidate_m_tiles(T, K, kernel, m_length, n_length, run_m, run_n)

    if T <: Real && prefer_swap(run_m, run_n, m_tile_asis, m_tile_swapped)
        # B takes the M role: groups, K maps, storages, run and transforms move
        # together; the sum is unchanged.
        kgroup_swapped = AxisGroup(korder, (indB, B), (indA, A))  # maps: (B, A)
        req_swapped = _plan_request(
            T, f, ngroup, mgroup, kgroup_swapped,
            parent(B), offset(B), parent(A), offset(A), parent(C), offset(C),
            run_n, m_block, k_block, n_block, allocator
        )
        return _plan_with_kernel(kernel, K, btransform, atransform, req_swapped)
    end
    req = _plan_request(
        T, f, mgroup, ngroup, kgroup,
        parent(A), offset(A), parent(B), offset(B), parent(C), offset(C),
        run_m, m_block, k_block, n_block, allocator
    )
    return _plan_with_kernel(kernel, K, atransform, btransform, req)
end

const _QS_ELTYPES = (Float32, Float64, ComplexF32, ComplexF64)

# A named mixed-domain kernel's `RealFormat` side packs a real operand only.
@inline _check_kernel_domain(kernel, ::Type, ::Type) = nothing
@inline _check_kernel_domain(kernel::ComplexRealKernel, ::Type, ::Type{TB}) where {TB} =
    TB <: Real || _throw_kernel_domain(kernel, "B", TB)
@inline _check_kernel_domain(kernel::RealComplexKernel, ::Type{TA}, ::Type) where {TA} =
    TA <: Real || _throw_kernel_domain(kernel, "A", TA)

@noinline _throw_kernel_domain(kernel, side, T) = throw(
    ArgumentError("plan_contract: $(typeof(kernel)) needs a real $side, got eltype $T")
)

# Fold to the compute type, or a throw, at compile time.
@inline function _compute_type(::Type{TA}, ::Type{TB}, ::Type{TC}, ::Nothing) where {TA, TB, TC}
    (TA in _QS_ELTYPES && TB in _QS_ELTYPES && TC in _QS_ELTYPES) ||
        _throw_eltypes(TA, TB, TC)
    (TC <: Real && !(TA <: Real && TB <: Real)) && _throw_complex_into_real(TA, TB, TC)
    return promote_type(TA, TB, TC)
end
@inline _compute_type(::Type{TA}, ::Type{TB}, ::Type{TC}, ::Type{R}) where {TA, TB, TC, R <: Union{Float32, Float64}} =
    _compute_type(TA, TB, TC, nothing) <: Complex ? Complex{R} : R

@noinline _throw_eltypes(TA, TB, TC) = throw(
    ArgumentError(
        "plan_contract: eltypes (A, B, C) = ($TA, $TB, $TC); each must be one of " *
            "Float32, Float64, ComplexF32, ComplexF64"
    )
)
@noinline _throw_complex_into_real(TA, TB, TC) = throw(
    ArgumentError("plan_contract: a complex operand (A: $TA, B: $TB) needs a complex C, got $TC")
)

# `m_tile` of the kernel each orientation would run, for the swap decision: a
# named kernel either way, else `select_shape`'s pick at that orientation's
# extent and C run. The swap is judged before `fit_to_run`'s K-dependent
# divisor step, which an unbounded `k_length` skips.
@inline _candidate_m_tiles(::Type{T}, ::Type, kernel, m_length::Int, n_length::Int, run_m::Int, run_n::Int) where {T} =
    (tile_size(kernel, 1), tile_size(kernel, 1))
@inline function _candidate_m_tiles(::Type{T}, ::Type{K}, ::Nothing, m_length::Int, n_length::Int, run_m::Int, run_n::Int) where {T, K}
    m_tile_asis = select_shape(T, K, m_length, typemax(Int), run_m)[1][1]
    m_tile_swapped = T <: Real ? select_shape(T, K, n_length, typemax(Int), run_n)[1][1] : m_tile_asis
    return m_tile_asis, m_tile_swapped
end

# Kernel resolution. A named kernel goes straight through, never demoted. An
# automatic one is chosen as a `(shape, kernel type)` value and the plan is
# built across a dispatch barrier specialised on that one concrete kernel type,
# so only the chosen kernel's code is compiled (holding the menu-wide kernel
# Union would box the request; a static ladder would compile every menu
# kernel). All barrier arguments are singletons or heap objects (the request
# in a `Ref`): one method-cache hit.
@inline _plan_with_kernel(kernel, ::Type, atransform, btransform, req::_PlanRequest) =
    _plan_contract(kernel, atransform, btransform, req, nothing)
@inline function _plan_with_kernel(::Nothing, ::Type{K}, atransform, btransform, req::_PlanRequest{T}) where {K, T}
    shape, kernel_type = select_shape(T, K, axis_length(req.mgroup), axis_length(req.kgroup), req.run)
    return _plan_at_shape(shape, kernel_type, atransform, btransform, req)
end

@inline function _plan_at_shape(shape, ::Val{K}, atransform, btransform, req::_PlanRequest{T}) where {K, T}
    vkernel = menu_val(shape, T, K)
    # The execution path, predicted so the callee is specialised on it. From
    # the shape and kernel type, not the kernel: a call union-split on the
    # kernel type is emitted out of line and boxes `req`.
    hint = _path_hint(req.f, req, shape, Val(K))
    return Base.inferencebarrier(_plan_resolved)(
        vkernel, atransform, btransform, hint, Base.RefValue{typeof(req)}(req)
    )
end

function _plan_resolved(
        ::Val{Kern}, atransform::TA, btransform::TB, hint::H, slot::Base.RefValue{R}
    ) where {Kern, TA, TB, H, R <: _PlanRequest}
    return _plan_contract(Kern(), atransform, btransform, slot[], hint)
end

# `nothing` for a continuation that does not execute (see src/execution/execute.jl).
@inline _path_hint(f, req::_PlanRequest, shape, kernel_type) = nothing

# Plan construction on a concrete kernel and transform pair (a named kernel's
# transform Unions die here). `req.f` runs in here, on the concrete plan type.
function _plan_contract(
        kernel::K, atransform::TA, btransform::TB, req::_PlanRequest{T}, hint::H
    ) where {K, TA, TB, T, H}
    scalartype(kernel) === T ||
        throw(ArgumentError("kernel scalar type $(scalartype(kernel)) does not match the compute type $T"))

    m_length = axis_length(req.mgroup)
    n_length = axis_length(req.ngroup)
    k_length = axis_length(req.kgroup)

    defaults = default_blocking(kernel)
    requested = Blocking(
        req.m_block === nothing ? defaults.m_block : req.m_block,
        req.k_block === nothing ? defaults.k_block : req.k_block,
        req.n_block === nothing ? defaults.n_block : req.n_block
    )

    m_tile, n_tile = tile_size(kernel)

    # Empty extents: the drivers never read these; the floors keep them valid.
    m_block_rounded = roundup(requested.m_block, m_tile)
    n_block_rounded = roundup(requested.n_block, n_tile)
    m_block = m_length == 0 ? m_tile : min(m_block_rounded, roundup(m_length, m_tile))
    n_block = n_length == 0 ? n_tile : min(n_block_rounded, roundup(n_length, n_tile))
    k_block = k_length == 0 ? 1 : min(requested.k_block, k_length)
    panel = _c_panel_needed(T, req.Cstorage, k_length, k_block)
    mpack = npack = _NO_SPLIT
    # Cache lines hold each operand's storage eltype, and the block walk costs
    # what its packed format's scatter does. B is not split under a panel of C:
    # the panel holds N blocks in C's own N order, which a split N group does
    # not enumerate contiguously.
    if m_length > 0 && n_length > 0 && k_length > 0
        m_block, mpack = pack_split(
            req.mgroup, req.kgroup, sliver_spec(kernel, 1), sizeof(eltype(req.Astorage)),
            k_block, m_block, m_block_rounded, requested.k_block
        )
        if !panel
            n_block, npack = pack_split(
                req.ngroup, req.kgroup, sliver_spec(kernel, 2), sizeof(eltype(req.Bstorage)),
                k_block, n_block, n_block_rounded, requested.k_block
            )
        end
    end
    blocking = Blocking(m_block, k_block, n_block)
    ws = ContractWorkspace(
        T, kernel, blocking; allocator = req.allocator, panel = panel ? m_length * min(n_block, n_length) : 0
    )

    plan = ContractPlan(
        kernel, req.mgroup, req.ngroup, req.kgroup, blocking,
        req.Astorage, req.Abase, req.Bstorage, req.Bbase, req.Cstorage, req.Cbase,
        atransform, btransform, ws, mpack, npack
    )
    return _continue(req.f, plan, hint)
end

# `hint` is used only by an executing continuation (src/execution/execute.jl).
@inline _continue(f::F, plan::ContractPlan, hint) where {F} = f(plan)

release!(plan::ContractPlan, allocator) = release!(plan.workspace, allocator)
