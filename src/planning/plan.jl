# The plan: everything `execute!` needs, resolved once so it can be reused.
# `plan_contract` runs the planning stages (labels, conjugation, kernel
# selection, blocking) and sizes the `ContractWorkspace`.

"""
    ContractPlan

Reusable, concretely typed plan from [`plan_contract`](@ref): the M/N/K
`AxisGroup`s, kernel, operand storages, effective [`Blocking`](@ref), packing
transforms and the [`ContractWorkspace`](@ref) holding every buffer
[`execute!`](@ref) needs. Field layout is not part of the public interface;
after an M/N orientation swap the `A*` fields describe the original `B`.
"""
struct ContractPlan{
        T, Kern, GM <: AxisGroup, GN <: AxisGroup, GK <: AxisGroup, SA, SB, SC,
        TA, TB, VT <: AbstractVector,
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

    workspace::ContractWorkspace{T, VT}
end

"""
    plan_contract(C::StridedView, A::StridedView, indA::NTuple{NA,Int},
                  B::StridedView, indB::NTuple{NB,Int},
                  indC::NTuple{NC,Int};
                  kernel = nothing,
                  conjA = false, conjB = false,
                  mc = nothing, kc = nothing, nc = nothing,
                  workspace = nothing,
                  allocator = TensorOperations.DefaultAllocator(),
                  oracle = true, accumulator = nothing) -> ContractPlan

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
    evaluated in `T`, rounding to `eltype(C)` on store. `C` holds the partial
    sums between `kc` blocks of K, so a K longer than `kc` rounds once per
    block.
  * `kernel = nothing` picks one from the hardware profile and the extents: a
    [`SIMDKernel`](@ref) for a real `T`, a [`PlanarKernel`](@ref) for a
    complex one (an [`FMAddSubKernel`](@ref) for a short M on AVX-512).
    [`OneMKernel`](@ref) is used only when named.
  * Labels within the M and N composites are ordered by their stride in `C`;
    the K order follows a cost model of the two packs. For a real `T` the
    operand roles are swapped (B feeds M) when only the N side gives `C` a
    unit-stride run long enough for the kernel's register tile; the result is
    the same either way.
  * `mc`/`kc`/`nc` override the fields of `default_blocking(kernel)` (each
    `>= 1`); `mc`/`nc` are rounded up to `mr`/`nr` multiples and all three are
    clamped to the extents.
  * `workspace` reuses an existing [`ContractWorkspace`](@ref) for compute
    type `T`, grown by [`reserve!`](@ref) as needed. A non-default
    TensorOperations `allocator` sizes the buffers once via `tensoralloc`,
    forbids `workspace`, and leaves [`release!`](@ref) to the caller.
  * `oracle = false` skips `execute_tilewise!`'s buffers.
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
        mc::Union{Int, Nothing} = nothing,
        kc::Union{Int, Nothing} = nothing,
        nc::Union{Int, Nothing} = nothing,
        workspace::Union{Nothing, ContractWorkspace} = nothing,
        allocator = TO.DefaultAllocator(),
        oracle::Bool = true,
        accumulator::Union{Nothing, Type{Float32}, Type{Float64}} = nothing
    ) where {NA, NB, NC}
    return _planned(
        identity, C, A, indA, B, indB, indC,
        kernel, conjA, conjB, mc, kc, nc, workspace, allocator, oracle, accumulator
    )
end

# Everything `_plan_contract` needs that is concretely typed before the kernel
# is known (`run` is the chosen M composite's unit-stride run in C), as one
# value that crosses the kernel barrier. `T` is the phantom compute type.
struct _PlanRequest{
        T, F, GM <: AxisGroup, GN <: AxisGroup, GK <: AxisGroup, SA, SB, SC,
        WS <: Union{Nothing, ContractWorkspace}, AL,
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
    mc::Union{Int, Nothing}
    kc::Union{Int, Nothing}
    nc::Union{Int, Nothing}
    workspace::WS
    allocator::AL
    oracle::Bool
end

@inline function _plan_request(
        ::Type{T}, f::F, mgroup::GM, ngroup::GN, kgroup::GK,
        Astorage::SA, Abase::Int, Bstorage::SB, Bbase::Int, Cstorage::SC, Cbase::Int,
        run::Int, mc, kc, nc, workspace::WS, allocator::AL, oracle::Bool
    ) where {T, F, GM, GN, GK, SA, SB, SC, WS, AL}
    return _PlanRequest{T, F, GM, GN, GK, SA, SB, SC, WS, AL}(
        f, mgroup, ngroup, kgroup, Astorage, Abase, Bstorage, Bbase, Cstorage, Cbase,
        run, mc, kc, nc, workspace, allocator, oracle
    )
end

# `plan_contract`'s body, positional, with a continuation `f` applied to the
# plan inside the barrier, where its type is concrete (the TensorOperations
# adapter passes an executor).
function _planned(
        f::F, C::StridedView, A::StridedView, indA::NTuple{NA, Int},
        B::StridedView, indB::NTuple{NB, Int}, indC::NTuple{NC, Int},
        kernel, conjA::Bool, conjB::Bool,
        mc::Union{Int, Nothing}, kc::Union{Int, Nothing}, nc::Union{Int, Nothing},
        workspace::Union{Nothing, ContractWorkspace}, allocator, oracle::Bool,
        accumulator::AC
    ) where {F, NA, NB, NC, AC}
    T = _compute_type(eltype(A), eltype(B), eltype(C), accumulator)

    # GUARDRAIL: a conjugated `C` is rejected; the engine writes through to
    # its parent, so there is nowhere to absorb its `op`.
    _qs_isconj(C, false) && throw(
        ArgumentError(
            "plan_contract: cannot write into a conjugated view (C has op $(C.op)); " *
                "writing a conjugated output is not supported"
        )
    )
    atransform = _qs_isconj(A, conjA) ? conj : identity
    btransform = _qs_isconj(B, conjB) ? conj : identity

    mlabels, nlabels, klabels = _classify_labels(indA, indB, indC)

    morder = _order_free_labels(mlabels, indC, C)
    norder = _order_free_labels(nlabels, indC, C)

    # Only real `T` swaps or run-demotes; `0` is a placeholder.
    run_m = T <: Real ? _leading_unit_run(morder, indC, C) : 0
    run_n = T <: Real ? _leading_unit_run(norder, indC, C) : 0

    mgroup = _build_pair_group(morder, indA, A, indC, C)  # maps: (A, C)
    ngroup = _build_pair_group(norder, indB, B, indC, C)  # maps: (B, C)

    Qm = axis_length(mgroup)
    Qn = axis_length(ngroup)

    korder = _order_contract_labels(klabels, indA, A, morder, indB, B, norder, Qm, Qn)
    kgroup = _build_pair_group(korder, indA, A, indB, B)  # maps: (A, B)

    Qk = axis_length(kgroup)

    mr_asis, mr_swapped = _candidate_mrs(T, kernel, Qm, Qn, run_m, run_n)

    if T <: Real && _prefer_swap(run_m, run_n, mr_asis, mr_swapped)
        # B takes the M role: groups, K maps, storages, run and transforms move
        # together; the sum is unchanged.
        kgroup_swapped = _build_pair_group(korder, indB, B, indA, A)  # maps: (B, A)
        req_swapped = _plan_request(
            T, f, ngroup, mgroup, kgroup_swapped,
            parent(B), offset(B), parent(A), offset(A), parent(C), offset(C),
            run_n, mc, kc, nc, workspace, allocator, oracle
        )
        return _plan_with_kernel(kernel, btransform, atransform, req_swapped)
    end
    req = _plan_request(
        T, f, mgroup, ngroup, kgroup,
        parent(A), offset(A), parent(B), offset(B), parent(C), offset(C),
        run_m, mc, kc, nc, workspace, allocator, oracle
    )
    return _plan_with_kernel(kernel, atransform, btransform, req)
end

const _QS_ELTYPES = (Float32, Float64, ComplexF32, ComplexF64)

# Fold to the compute type, or a throw, at compile time.
@inline function _compute_type(::Type{TA}, ::Type{TB}, ::Type{TC}, ::Nothing) where {TA, TB, TC}
    (TA in _QS_ELTYPES && TB in _QS_ELTYPES && TC in _QS_ELTYPES) ||
        _throw_eltypes(TA, TB, TC)
    (TC <: Real && !(TA <: Real && TB <: Real)) && _throw_complex_into_real(TA, TB, TC)
    return promote_type(TA, TB, TC)
end
@inline function _compute_type(::Type{TA}, ::Type{TB}, ::Type{TC}, ::Type{R}) where {TA, TB, TC, R}
    return _compute_type(TA, TB, TC, nothing) <: Complex ? Complex{R} : R
end

@noinline _throw_eltypes(TA, TB, TC) = throw(
    ArgumentError(
        "plan_contract: eltypes (A, B, C) = ($TA, $TB, $TC); each must be one of " *
            "Float32, Float64, ComplexF32, ComplexF64"
    )
)
@noinline _throw_complex_into_real(TA, TB, TC) = throw(
    ArgumentError("plan_contract: a complex operand (A: $TA, B: $TB) needs a complex C, got $TC")
)

# `mr` of the kernel each orientation would run, for the swap decision: a named
# kernel either way, else `_default_shape`'s pick at that orientation's extents
# and C run.
@inline _candidate_mrs(::Type{T}, kernel, Qm::Int, Qn::Int, run_m::Int, run_n::Int) where {T} =
    (mr(kernel), mr(kernel))
@inline function _candidate_mrs(::Type{T}, ::Nothing, Qm::Int, Qn::Int, run_m::Int, run_n::Int) where {T}
    mr_asis = _default_shape(T, Qm, Qn, run_m)[1][1]
    mr_swapped = T <: Real ? _default_shape(T, Qn, Qm, run_n)[1][1] : mr_asis
    return mr_asis, mr_swapped
end

# Kernel resolution. A named kernel goes straight through, never demoted. An
# automatic one is chosen as a `(shape, method)` value and the plan is built
# across a dispatch barrier specialised on that one shape, so only the chosen
# kernel's code is compiled (holding the menu-wide kernel Union would box the
# request; a static ladder would compile every menu kernel). All barrier
# arguments are singletons or heap objects: 0 B, one method-cache hit.
@inline _plan_with_kernel(kernel, atransform, btransform, req::_PlanRequest) =
    _plan_contract(kernel, atransform, btransform, req, nothing)
@inline function _plan_with_kernel(::Nothing, atransform, btransform, req::_PlanRequest{T}) where {T}
    Qm = axis_length(req.mgroup)
    shape, method = _default_shape(T, Qm, axis_length(req.ngroup), req.run)
    shape = _demote_shape_for_run(T, shape, method, req.run, Qm, axis_length(req.kgroup))
    vshape = _menu_val(shape, T, method)
    # The execution path, predicted so the callee is specialised on it. Only a
    # `Bool` crosses: a call union-split on `method` is emitted out of line and
    # boxes `req`.
    hint = _path_hint(req.f, req, _unpacked_b_method_eligible(method))
    core = _strip_storage(req)
    slot = _barrier_slot!(req.workspace, typeof(core))
    slot[] = core
    return Base.inferencebarrier(_plan_resolved)(
        vshape, method, atransform, btransform, hint, slot,
        req.Astorage, req.Bstorage, req.Cstorage
    )
end

function _plan_resolved(
        ::Val{S}, method::M, atransform::TA, btransform::TB, hint::H,
        slot::Base.RefValue{R}, Astorage::SA, Bstorage::SB, Cstorage::SC
    ) where {S, M, TA, TB, H, T, R <: _PlanRequest{T}, SA, SB, SC}
    req = _with_storage(slot[], Astorage, Bstorage, Cstorage)
    return _plan_contract(_kernel_from_shape(S, T, method), atransform, btransform, req, hint)
end

# The storages cross the barrier as arguments, so no slot retains a user array.
@inline _strip_storage(req::_PlanRequest{T}) where {T} = _plan_request(
    T, req.f, req.mgroup, req.ngroup, req.kgroup,
    nothing, req.Abase, nothing, req.Bbase, nothing, req.Cbase,
    req.run, req.mc, req.kc, req.nc, req.workspace, req.allocator, req.oracle
)
@inline _with_storage(req::_PlanRequest{T}, Astorage, Bstorage, Cstorage) where {T} = _plan_request(
    T, req.f, req.mgroup, req.ngroup, req.kgroup,
    Astorage, req.Abase, Bstorage, req.Bbase, Cstorage, req.Cbase,
    req.run, req.mc, req.kc, req.nc, req.workspace, req.allocator, req.oracle
)

# `nothing` for a continuation that does not execute (see src/execution/execute.jl).
@inline _path_hint(f, req::_PlanRequest, unpack_ok::Bool) = nothing

# Plan construction on a concrete kernel and transform pair (a named kernel's
# transform Unions die here). `req.f` runs in here, on the concrete plan type.
function _plan_contract(
        kernel::K, atransform::TA, btransform::TB, req::_PlanRequest{T}, hint::H
    ) where {K, TA, TB, T, H}
    scalartype(kernel) === T ||
        throw(ArgumentError("kernel scalar type $(scalartype(kernel)) does not match the compute type $T"))

    Qm = axis_length(req.mgroup)
    Qn = axis_length(req.ngroup)
    Qk = axis_length(req.kgroup)

    defaults = default_blocking(kernel)
    mc, kc, nc = req.mc, req.kc, req.nc
    requested = Blocking(
        mc === nothing ? defaults.mc : mc,
        kc === nothing ? defaults.kc : kc,
        nc === nothing ? defaults.nc : nc
    )

    MRk = mr(kernel)
    NRk = nr(kernel)

    # Empty extents: the drivers never read these; the floors keep them valid.
    mc_eff = Qm == 0 ? MRk : min(_roundup(requested.mc, MRk), _roundup(Qm, MRk))
    nc_eff = Qn == 0 ? NRk : min(_roundup(requested.nc, NRk), _roundup(Qn, NRk))
    kc_eff = Qk == 0 ? 1 : min(requested.kc, Qk)
    blocking = Blocking(mc_eff, kc_eff, nc_eff)
    ws = _resolve_workspace(T, req.workspace, kernel, blocking, req.oracle, req.allocator)

    plan = ContractPlan(
        kernel, req.mgroup, req.ngroup, req.kgroup, blocking,
        req.Astorage, req.Abase, req.Bstorage, req.Bbase, req.Cstorage, req.Cbase,
        atransform, btransform, ws
    )
    return _continue(req.f, plan, hint)
end

# `hint` is used only by an executing continuation (src/execution/execute.jl).
@inline _continue(f::F, plan::ContractPlan, hint) where {F} = f(plan)
