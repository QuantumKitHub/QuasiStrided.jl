# The plan: everything `execute!` needs, resolved once so it can be reused.
#   plan_contract(...) -> ContractPlan; execute!(plan, alpha, beta); contract! = both.
# `plan_contract` orchestrates the other planning stages -- labels
# (src/planning/labels.jl), conjugation, kernel selection, blocking -- and
# sizes the `ContractWorkspace` (src/execution/workspace.jl).

"""
    ContractPlan

Reusable plan from [`plan_contract`](@ref): resolved M/N/K `AxisGroup`s,
kernel, operand storage/base, the effective [`Blocking`](@ref), and the
[`ContractWorkspace`](@ref) holding every buffer
[`execute!`](@ref)/[`execute_tilewise!`](@ref) need -- sized once during
planning, never (re)allocated during execution, plus the per-operand packing
transforms `atransform`/`btransform`. `VT` is the workspace's packed-panel
vector type (`Vector{real(T)}` on the default allocator path, so `Vector{T}`
on the real path), a `where`-bound parameter resolved at construction, so
every plan instance is concretely typed. Field layout is an implementation
detail, not part of the public interface.

Note the M/N orientation swap:
after a swap, `Astorage`/`Abase`/`atransform` describe the ORIGINAL `B`
operand and `Bstorage`/`Bbase`/`btransform` describe the original `A`, so
`plan.Astorage === parent(A)` does not hold in general -- do not assume the
field name still tracks the user-facing argument it is named after.
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

    # `identity` or `conj`, as singleton function VALUES with their own type
    # parameters. GUARDRAIL: not a `Bool` field and not a `Val{Bool}` -- either
    # would cross `_pack_sliver!` as a `Union` or need mapping to a function at
    # the pack site, costing a dynamic dispatch (~80 B) per call.
    atransform::TA
    btransform::TB

    # Every buffer both drivers use.
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
                  oracle = true) -> ContractPlan

`kernel = nothing` (the default) resolves the kernel *after* the M/N/K groups
are built, via `_default_kernel(T, Qm, Qn)`, because the extent-aware demotion
needs `Qm`. For a real element type that is a [`SIMDKernel`](@ref) at the
hardware-derived shape; for a complex one a [`PlanarKernel`](@ref) at the
measured shape on AVX-512, and at a shape fitted to the register file
elsewhere (see src/planning/kernel_selection.jl). Pass `kernel` explicitly to
override. [`OneMKernel`](@ref) is never selected automatically; naming it is
the only way to use 1m. On AVX-512 a complex `Qm` below the planar tile's `MR`
demotes to an [`FMAddSubKernel`](@ref) instead (`_small_m_shape`).

Planning phase of [`contract!`](@ref): resolves labels into M/N/K
`AxisGroup`s, validates matched axis lengths and eltypes, and preallocates
every buffer [`execute!`](@ref) needs.

Label order and orientation: the
labels inside the M composite (A's free labels) and the N composite (B's free
labels) are each stable-sorted by `abs(stride)` of the label's axis *in `C`*,
ascending, ties keeping the operand's own axis order -- so each composite is
enumerated with `C`'s fastest axis fastest, whatever A's or B's layout is. The
K composite keeps `indA` order unless a cost model of the two packs (page-
crossing K walks, cache lines refetched once the lines in flight exceed L2)
prefers the order sorted by `abs(stride)` in `A` or in `B` (see
`_order_contract_labels`). Then, if the sorted M list does *not* begin
with a unit-stride run of at least `mr(kernel)` elements in `C` while the sorted
N list does, the operand roles are swapped: `B` feeds M and `A` feeds N, and
the K maps, the storage/base fields and the conjugation transforms move with
them (so `plan.Astorage` may be `parent(B)`). The result is unchanged either
way; only the walk through `C` is. `mc`/`kc`/`nc` are the macro-blocking
factors (see [`Blocking`](@ref)); a `nothing` keyword takes the corresponding
field of `default_blocking(kernel)`. Each must be `>= 1`, and is then rounded
and clamped into the *effective* blocking stored on the plan: `mc`/`nc` round
up to a whole `mr(kernel)`/`nr(kernel)` multiple, then cap at the M/N extent
(likewise rounded up); `kc` caps at the K extent. Throws
`ArgumentError`/`DimensionMismatch` on invalid input.

Buffers:

  * `workspace = nothing` builds a fresh [`ContractWorkspace`](@ref); passing
    an existing one reuses it, grown as needed by [`reserve!`](@ref), even
    across differently shaped contractions.
  * `allocator` is a TensorOperations allocator. `DefaultAllocator` gives a
    plain, GC-owned, `reserve!`-able `ContractWorkspace{T,Vector{T}}`; any
    other one sizes the packed panels exactly once via
    `TensorOperations.tensoralloc`, forbids `workspace` as well, and leaves
    [`release!`](@ref) to the caller.
  * `oracle = false` skips `execute_tilewise!`'s own buffers entirely, making
    that oracle unavailable for this plan. `execute!` is unaffected.

Conjugation: `conjA`/`conjB` request `conj` on A's/B's elements, TensorOperations'
semantics (`alpha`/`beta` are never conjugated). Each flag is folded here with
the corresponding view's `op` -- they compose with XOR, so a `conj`-wrapped view
with `conjA = true` is unconjugated -- and the result is stored on the plan as a
singleton transform applied during packing. A conjugated *output* view is
rejected: the engine addresses `parent(C)` directly and would silently ignore
it. For a real element type both transforms are `identity` no matter what the
flags say, so the real path gains no specialization.
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
        oracle::Bool = true
    ) where {NA, NB, NC}
    return _planned(
        identity, C, A, indA, B, indB, indC,
        kernel, conjA, conjB, mc, kc, nc, workspace, allocator, oracle
    )
end

# Everything `_plan_contract` needs that is already concretely typed BEFORE
# the kernel is known: the continuation, the three groups (their ranks are
# fixed by the label tuples' lengths, so `GM`/`GN`/`GK` are concrete), the
# operand storage/base pairs, the leading unit-stride run of the chosen M
# composite (for `_demote_for_run`), the blocking overrides and the
# workspace/allocator choice. One value, so the kernel-resolution ladder
# (`_plan_with_kernel`) forwards a single argument, and so the swapped and
# as-is orientations differ only in how it is filled.
#
# `T` is a phantom parameter (the storage element type, `eltype(C)`), carried
# so the barrier can check the kernel against it without re-deriving it.
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

# `plan_contract`'s body, with a continuation: `f(plan)` is applied INSIDE the
# `_plan_contract` barrier, where the plan's type is concrete, and its result
# returned. `plan_contract` passes `identity`; the TensorOperations adapter
# (src/integrations/tensoroperations.jl) passes an executor, so that the plan
# is built and consumed in one concretely typed frame. Positional throughout:
# this is the hot path, and the keyword handling is `plan_contract`'s job.
function _planned(
        f::F, C::StridedView, A::StridedView, indA::NTuple{NA, Int},
        B::StridedView, indB::NTuple{NB, Int}, indC::NTuple{NC, Int},
        kernel, conjA::Bool, conjB::Bool,
        mc::Union{Int, Nothing}, kc::Union{Int, Nothing}, nc::Union{Int, Nothing},
        workspace::Union{Nothing, ContractWorkspace}, allocator, oracle::Bool
    ) where {F, NA, NB, NC}
    T = eltype(C)
    eltype(A) === T ||
        throw(ArgumentError("eltype(A) = $(eltype(A)) does not match eltype(C) = $T"))
    eltype(B) === T ||
        throw(ArgumentError("eltype(B) = $(eltype(B)) does not match eltype(C) = $T"))

    # Fold each flag with its view's `op`. GUARDRAIL: a conjugated `C` is
    # REJECTED, not supported -- there is nowhere to absorb its `op` (the
    # engine writes through to the parent), so it would be silently wrong;
    # supporting it would also thread a flag through `store_tile!` and force
    # re-deriving the beta-applied-once argument. The two transforms are
    # `Union{typeof(identity),typeof(conj)}` here and die at the
    # `_plan_contract` barrier below, as the kernel's Union does.
    _qs_isconj(C, false) && throw(
        ArgumentError(
            "plan_contract: cannot write into a conjugated view (C has op $(C.op)); " *
                "writing a conjugated output is not supported"
        )
    )
    atransform = _qs_isconj(A, conjA) ? conj : identity
    btransform = _qs_isconj(B, conjB) ? conj : identity

    # Statically sized label tuples (src/planning/labels.jl): the three
    # composite ranks follow from `NA`/`NB`/`NC`, so the groups built below
    # are concretely typed and nothing here allocates.
    mlabels, nlabels, klabels = _classify_labels(indA, indB, indC)

    # C's layout, not A's/B's, decides the order within each composite.
    morder = _order_free_labels(mlabels, indC, C)
    norder = _order_free_labels(nlabels, indC, C)

    # Each composite's own leading unit-stride run length, computed once: the
    # swap decision and the run-length demotion below both consume these two
    # values, keyed on the composite (M or N), not on which orientation ends
    # up feeding M. Real `T` only: both consumers are `T <: Real`-gated, so a
    # complex plan skips the two `_leading_unit_run` calls (`0` is a
    # placeholder).
    run_m = T <: Real ? _leading_unit_run(morder, indC, C) : 0
    run_n = T <: Real ? _leading_unit_run(norder, indC, C) : 0

    mgroup = _build_pair_group(morder, indA, A, indC, C)  # maps: (A, C)
    ngroup = _build_pair_group(norder, indB, B, indC, C)  # maps: (B, C)

    Qm = axis_length(mgroup)
    Qn = axis_length(ngroup)

    # The K order is decided by a cost model of the two packs
    # (`_order_contract_labels`, src/planning/labels.jl), which needs each
    # operand's free-label order (whole-line slivers or not) and free extent.
    korder = _order_contract_labels(klabels, indA, A, morder, indB, B, norder, Qm, Qn)
    kgroup = _build_pair_group(korder, indA, A, indB, B)  # maps: (A, B)

    Qk = axis_length(kgroup)

    # The register width each orientation would run at (`_candidate_mrs`):
    # only this much of the kernel choice is needed here, for the swap
    # decision. The kernel itself is resolved by `_plan_with_kernel`, after
    # the orientation is fixed -- and resolved as a `(shape, method)` value,
    # not a kernel object, so no menu-wide Union is ever held in this frame.
    mr_asis, mr_swapped = _candidate_mrs(T, kernel, Qm, Qn)

    # The swap is for real element types only. Extending it to complex
    # kernels, which now also have a vectorized store, is a deliberately
    # deferred, unmeasured follow-up. Real kernels (`SIMDKernel` and
    # `ScalarKernel`) keep it. Uses the precomputed `run_m`/`run_n` directly.
    if T <: Real && _prefer_swap(run_m, run_n, mr_asis, mr_swapped)
        # B takes the M role and A the N role. Everything operand-bound moves
        # together: the groups (each already carries its own C map), the K
        # group's two maps, the storage/base pair `_plan_contract` reads off
        # the request's A/B fields, the run length the demotion is keyed on
        # (N's own, i.e. the CHOSEN M orientation's) and the packing
        # transforms. The contraction is unchanged: `*` commutes on `T` and
        # `conj` is elementwise, so `sum_k conj?(B[n,k]) * conj?(A[m,k])` is
        # the same sum.
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

# `mr` of the kernel each orientation would run: the as-is one at `(Qm, Qn)`
# and the swapped one at `(Qn, Qm)`. A caller-named kernel runs either way;
# an automatic one is `_default_shape`'s pick, whose two candidates differ
# only when the small-M demotion applies to one orientation. The swapped
# candidate is resolved for a real `T` only, the only `T` that can swap.
@inline _candidate_mrs(::Type{T}, kernel, Qm::Int, Qn::Int) where {T} = (mr(kernel), mr(kernel))
@inline function _candidate_mrs(::Type{T}, ::Nothing, Qm::Int, Qn::Int) where {T}
    mr_asis = _default_shape(T, Qm, Qn)[1][1]
    mr_swapped = T <: Real ? _default_shape(T, Qn, Qm)[1][1] : mr_asis
    return mr_asis, mr_swapped
end

# Kernel resolution, the last step before the barrier. A caller-named kernel
# goes straight through, never demoted. An automatic one is chosen as a
# `(shape, method)` value (`_default_shape`) and built INSIDE
# `_with_menu_kernel`'s unrolled ladder, so that `_plan_demoted` -- and
# through it `_demote_for_run` and `_plan_contract` -- always runs on a
# concrete kernel type and every call below this line is static.
#
# Why not simply `_plan_contract(_default_kernel(T, Qm, Qn), ...)`: that
# value's type is the Union of `T`'s whole menu, ten members for ComplexF64,
# past the four Julia union-splits, so the barrier call became a
# `jl_apply_generic` that boxed the request and -- for reasons this file does
# not claim to understand -- missed the method cache more often than not.
# Measured on ccqlin038 (Julia 1.12.7, 8x8x8 ComplexF64, pooled workspace):
# `plan_contract` 0.78 us and 704 B/call with the Union-typed call; the same
# tail with a concrete kernel 0.09 us and 0 B. Float64 was already static
# (a two-member Union) and is unchanged by this.
@inline _plan_with_kernel(kernel, atransform, btransform, req::_PlanRequest) =
    _plan_contract(kernel, atransform, btransform, req)
@inline function _plan_with_kernel(::Nothing, atransform, btransform, req::_PlanRequest{T}) where {T}
    shape, method = _default_shape(T, axis_length(req.mgroup), axis_length(req.ngroup))
    return _with_menu_kernel(_plan_demoted, shape, T, method, atransform, btransform, req)
end

# Run-length demotion (`_demote_for_run`) of an automatically chosen kernel,
# keyed on the chosen M orientation's own run (`req.run`), then the barrier.
@inline function _plan_demoted(kernel, atransform, btransform, req::_PlanRequest{T}) where {T}
    kernel_final = _demote_for_run(
        T, kernel, req.run, axis_length(req.mgroup), axis_length(req.kgroup)
    )
    return _plan_contract(kernel_final, atransform, btransform, req)
end

# Function barrier: the `conj`/`identity` transform Unions die here (and, on
# the real path, `_demote_for_run`'s small kernel Union), so `ContractPlan`'s
# `Kern`, `TA` and `TB` are concrete and `execute!` sees no abstract type.
# `TA`/`TB` each get their own bound parameter for the same reason `K` does.
# The continuation `req.f` runs in here rather than on the returned plan:
# `f(plan)` sees a concrete plan type, so an executor passed as `f` needs no
# further dispatch and no boxed plan.
function _plan_contract(
        kernel::K, atransform::TA, btransform::TB, req::_PlanRequest{T}
    ) where {K, TA, TB, T}
    scalartype(kernel) === T ||
        throw(ArgumentError("kernel scalar type $(scalartype(kernel)) does not match eltype(C) = $T"))

    Qm = axis_length(req.mgroup)
    Qn = axis_length(req.ngroup)
    Qk = axis_length(req.kgroup)

    defaults = default_blocking(kernel)
    # Blocking's own constructor validates all three >= 1.
    mc, kc, nc = req.mc, req.kc, req.nc
    requested = Blocking(
        mc === nothing ? defaults.mc : mc,
        kc === nothing ? defaults.kc : kc,
        nc === nothing ? defaults.nc : nc
    )

    MRk = mr(kernel)
    NRk = nr(kernel)

    mc_rounded = _roundup(requested.mc, MRk)
    nc_rounded = _roundup(requested.nc, NRk)

    # On an empty extent both drivers short-circuit before reading these, so
    # the floor here only keeps Blocking's >=1 invariant and the buffers
    # well-formed.
    mc_eff = Qm == 0 ? MRk : min(mc_rounded, _roundup(Qm, MRk))
    nc_eff = Qn == 0 ? NRk : min(nc_rounded, _roundup(Qn, NRk))
    kc_eff = Qk == 0 ? 1 : min(requested.kc, Qk)

    blocking = Blocking(mc_eff, kc_eff, nc_eff)

    # Buffers are `undef`-initialized, not zeroed; see `ContractWorkspace`.
    ws = _resolve_workspace(T, req.workspace, kernel, blocking, req.oracle, req.allocator)

    plan = ContractPlan(
        kernel, req.mgroup, req.ngroup, req.kgroup, blocking,
        req.Astorage, req.Abase, req.Bstorage, req.Bbase, req.Cstorage, req.Cbase,
        atransform, btransform, ws
    )
    return req.f(plan)
end
