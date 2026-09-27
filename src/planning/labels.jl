# Label planning: classify every label into M/N/K, order the free labels
# by their stride in C, and decide the M/N operand orientation.
#
# Every label list here is a statically sized `NTuple{D,Int}`, never a
# `Vector{Int}`: the three composite RANKS are fixed by the label tuples'
# lengths alone (`_group_ranks`), so the lists can be built, sorted and handed
# to `_build_pair_group` without one heap allocation, and each `AxisGroup{D,2}`
# comes out concretely typed. Measured (ccqlin038, Julia 1.12.7, 8x8x8 real
# GEMM through the TensorOperations adapter): the previous `Vector`-based
# planner allocated five vectors (~370 B) per call, and the value-dependent
# rank made the `_plan_contract` barrier a dynamic dispatch that boxed every
# non-isbits argument (~800 B more) -- together roughly half of a 1.1 us call.

# Membership test against a statically-sized label tuple, used instead of a
# `Set`: the label tuples have a compile-time-known LENGTH (the `NA`/`NB`/`NC`
# parameters, one specialization per arity), so this unrolls into a chain of
# integer compares and allocates nothing, where each `Set` would cost a
# `Dict`'s slot/key arrays on every plan. The tuples are `allunique` by the
# checks at the top of `_classify_labels`, so a linear scan is also the whole
# of the membership question.
@inline _label_in(lbl::Int, t::NTuple{N, Int}) where {N} = any(==(lbl), t)

# The M/N/K composite ranks, from the label tuples' lengths alone. Once
# `_classify_labels`' validation has passed, the labels of A, B and C are each
# partitioned as A = M ∪ K, B = N ∪ K, C = M ∪ N (every label is in exactly two
# operands), so
#
#     NA = |M| + |K|,  NB = |N| + |K|,  NC = |M| + |N|
#
# and the three counts follow by solving. Compile-time constants: `NA`/`NB`/
# `NC` are type parameters, so the label tuples below get literal lengths and
# every `AxisGroup` a concrete rank. On an input the validation REJECTS these
# formulas may be meaningless (even negative); they are only ever evaluated
# after validation has passed, and `_classify_labels` still asserts them
# against the counts it actually found.
@inline _group_ranks(NA::Int, NB::Int, NC::Int) =
    ((NA + NC - NB) ÷ 2, (NB + NC - NA) ÷ 2, (NA + NB - NC) ÷ 2)

@noinline _throw_rank_mismatch(which::Symbol, found::Int, derived::Int) = throw(
    ArgumentError(
        "internal error: $which label count $found does not match the rank $derived " *
            "derived from the label tuple lengths"
    )
)

# Append `x` as element `n` of the statically sized `out` (1-based), leaving
# it unchanged when `n` is past the end: `_classify_labels` fills the lists as
# it validates, and the length guard keeps an input that is about to be
# rejected from indexing past a literal-length tuple before the rejection
# fires.
@inline function _push_label(out::NTuple{D, Int}, n::Int, x::Int) where {D}
    return n <= D ? _tupleset(out, n, x) : out
end

# Classify every label in indA ∪ indB ∪ indC into M/N/K. Returns
# (mlabels, nlabels, klabels) as `NTuple`s in indA/indB appearance order, with
# lengths fixed by `_group_ranks`. Per (inA,inB,inC):
#   (T,F,T)->M  (F,T,T)->N  (T,T,F)->K  everything else -> ArgumentError
# (labeled in C only, present in all three, or dangling in just A or B).
function _classify_labels(
        indA::NTuple{NA, Int}, indB::NTuple{NB, Int},
        indC::NTuple{NC, Int}
    ) where {NA, NB, NC}
    allunique(indA) ||
        throw(ArgumentError("indA has a repeated label (diagonal), not supported: $indA"))
    allunique(indB) ||
        throw(ArgumentError("indB has a repeated label (diagonal), not supported: $indB"))
    allunique(indC) ||
        throw(ArgumentError("indC has a repeated label (diagonal), not supported: $indC"))

    DM, DN, DK = _group_ranks(NA, NB, NC)
    mlabels = ntuple(_ -> 0, Val(max(DM, 0)))
    nlabels = ntuple(_ -> 0, Val(max(DN, 0)))
    klabels = ntuple(_ -> 0, Val(max(DK, 0)))
    nm = 0
    nk = 0
    for lbl in indA
        inB = _label_in(lbl, indB)
        inC = _label_in(lbl, indC)
        if inB && inC
            throw(
                ArgumentError(
                    "label $lbl appears in indA, indB, and indC: labels present in all " *
                        "three operands (batch-like) are not supported"
                )
            )
        elseif inB && !inC
            nk += 1
            klabels = _push_label(klabels, nk, lbl)
        elseif !inB && inC
            nm += 1
            mlabels = _push_label(mlabels, nm, lbl)
        else
            throw(
                ArgumentError(
                    "label $lbl appears only in indA (not in indB or indC): not a valid " *
                        "free (M) or contracted (K) label"
                )
            )
        end
    end

    nn = 0
    for lbl in indB
        inA = _label_in(lbl, indA)
        inC = _label_in(lbl, indC)
        if inA && inC
            continue  # already rejected while scanning indA, above.
        elseif inA && !inC
            continue  # already classified as K, above.
        elseif !inA && inC
            nn += 1
            nlabels = _push_label(nlabels, nn, lbl)
        else
            throw(
                ArgumentError(
                    "label $lbl appears only in indB (not in indA or indC): not a valid " *
                        "free (N) or contracted (K) label"
                )
            )
        end
    end

    for lbl in indC
        inA = _label_in(lbl, indA)
        inB = _label_in(lbl, indB)
        (inA || inB) ||
            throw(ArgumentError("label $lbl appears in indC but not in indA or indB"))
    end

    # Unreachable once the three loops above have passed (see `_group_ranks`);
    # kept so a wrong rank can never silently truncate a label list.
    nm == DM || _throw_rank_mismatch(:M, nm, DM)
    nn == DN || _throw_rank_mismatch(:N, nn, DN)
    nk == DK || _throw_rank_mismatch(:K, nk, DK)
    return mlabels, nlabels, klabels
end

@noinline _throw_label_length(lbl::Int, l1::Int, l2::Int) = throw(
    DimensionMismatch("label $lbl has mismatched axis length: $l1 vs $l2")
)

# ----------------------------------------------------------------------------
# Free-label order and M/N orientation. `_classify_labels` lists free labels
# in A's/B's own axis order,
# which is incidental to C: `fill_offsets!` enumerates a composite with its
# FIRST label fastest, so that order fixes the store loop's walk through C.
# Both helpers below are pure planning-time functions of (labels, indC, C).
# ----------------------------------------------------------------------------

# Stable sort of `labels` by `abs(stride)` of each label's axis in C,
# ascending; ties keep input order, so a single label or an already-sorted list
# comes back unchanged. Every label must occur in `indC` (the M/N lists from
# `_classify_labels` do by construction; K labels never come here).
# Insertion sort rather than `sortperm` + permuted copy: the latter allocates
# the key vector, the permutation and the result, and these lists have at most
# `ndims(C)` entries, so an O(n^2) sort with n <= 6 is not a cost. Strict `>`
# in the shift test keeps it STABLE, which is the contract (ties keep input
# order). Works on the tuple by value through `_tupleset`, so the caller's
# `labels` is untouched and nothing is allocated.
function _order_free_labels(
        labels::NTuple{D, Int}, indC::NTuple{NC, Int}, C::StridedView
    ) where {D, NC}
    st = Base.strides(C)
    key(l::Int) = abs(st[findfirst(==(l), indC)::Int])
    out = labels
    @inbounds for i in 2:D
        x = out[i]
        kx = key(x)
        j = i - 1
        while j >= 1 && key(out[j]) > kx
            out = _tupleset(out, j + 1, out[j])
            j -= 1
        end
        out = _tupleset(out, j + 1, x)
    end
    return out
end

# ----------------------------------------------------------------------------
# Contracted-label order. `_classify_labels` lists the K labels in `indA`
# order, which is incidental to the memory walk: `_pack_panel!` steps through
# K in the composite's enumeration order (its OUTER loop, one K step per
# `mr`/`nr`-element sliver row), so the K order decides how BOTH packs walk
# their operand. When the K axes sit in a different relative order in A and B
# -- `C[a,e] = A[a,b,c,d] * B[d,c,b,e]`, the upstream suite's
# `contract_scrambled` layout -- no single order suits both, and `indA` order
# is the one that suits neither: B's unit-stride axis `d` becomes the SLOWEST K
# coordinate, so `pack_b!` jumps `96^2` elements per K step and touches a new
# cache line (and page) for every element it reads. Measured (dim 96, Float64,
# Cascade Lake, benchmark/probes/probe_stage_breakdown.jl): 49 ns per B
# element, 92% of a 4.6 s contraction, against 1.3 ns per A element of the
# same case.
# ----------------------------------------------------------------------------

# Stable sort of `labels` by `abs(stride)` of each label's axis in `v`: the
# same allocation-free insertion sort as `_order_free_labels`, over a
# different operand, so it is that function under a name that does not say
# "free" -- the result is an `NTuple{D,Int}` of the input's static length.
@inline _sort_labels_by_stride(
    labels::NTuple{D, Int}, ind::NTuple{N, Int}, v::StridedView
) where {D, N} = _order_free_labels(labels, ind, v)

# The K order is chosen among three candidates -- `indA` order (the historical
# one), sorted by A's strides, sorted by B's strides -- by a cost model of the
# two packs, `_k_order_cost`. Ties keep `indA` order, so any layout the model
# is indifferent about (every gemm_ready/a_permuted/b_permuted layout of the
# upstream suite, and every single-K-label contraction) plans exactly what it
# did before the rule existed. The order is a property of the pair (A, B) and
# is the same whether or not the M/N orientation is later swapped.
#
# The model, per operand (`n` elements, `sizeof(T)` bytes each):
#
#     cost = n * walk * amplification
#
#   * `walk` is 1 when the fastest K axis of the order steps the operand by at
#     most a page (`_K_WALK_FAR_BYTES`) and `_K_WALK_FAR_PENALTY` beyond that:
#     a pack whose consecutive K steps land pages apart is a chain of demand
#     misses no prefetcher follows (measured 3.3 ns per element against
#     0.9-1.3 for the same sliver shape at a 768 B step, dim96_1_3_1).
#   * `amplification` is the number of times a cache line of the operand is
#     fetched. It is 1 when the operand's slivers are whole lines (its free
#     group's leading axis is unit-stride and at least a line long), or when
#     the operand's smallest-stride axis `u` is a K axis that consecutive K
#     steps advance (it is the fastest K axis), or when the lines touched
#     before `u` advances -- (product of the faster K extents) x (free extent)
#     lines -- still fit the core's L2. Otherwise every line is fetched once
#     per element it holds along `u`, i.e. `min(L_u, line_bytes / stride_u)`
#     times (8 for a unit-stride Float64 axis).
#
# Why not "sort by the operand with the unit-stride K axis": that operand may
# be small enough to live in L2 whatever the order (TRG's rank-3 factor at
# chi = 24 is 110 KB) while the other loses a 192 B walk for a 2.6 MB one --
# measured 1.71x slower on that contraction. The model keeps `indA` order
# there (A: 8M elements x walk 1 vs x walk 3) and still flips
# `contract_scrambled` (B: 85M elements x walk 3 x amplification 8 before).

# Fastest-K byte stride beyond which the pack is latency-bound, and its cost
# relative to a prefetchable walk.
const _K_WALK_FAR_BYTES = 4096
const _K_WALK_FAR_PENALTY = 3

# Cache-line size the model assumes; every measured host has 64 B lines.
const _K_LINE_BYTES = 64

# The core's private L2 share (see `_modelled_blocking`, src/planning/blocking.jl),
# or 1 MB when undetected -- between the 512 KB (Zen 2) and 1.25 MB (Ice Lake)
# of the machines measured, so an unknown host errs neither way. Read from the
# detected cache profile (`target_profile()`, filled by `_init_target!` at
# load). The planner does not call this per plan: the value is resolved once
# per (profile, eltype) into `ResolvedDefaults.l2_core` (src/planning/
# defaults.jl) and read from there (`_l2_core_bytes(T)`).
function _l2_core_bytes(profile::TargetProfile)
    l2 = profile.l2
    l2.bytes > 0 || return 1 << 20
    smt = max(1, profile.l1d.sharing)
    return l2.bytes ÷ max(1, l2.sharing ÷ smt)
end
_l2_core_bytes() = _l2_core_bytes(target_profile())

# Cached form for element type `T`: a field of the per-(profile, eltype)
# defaults, so no division is redone per plan; an element type without a
# defaults slot (no kernel menu, it fails later in planning) reads the
# profile directly.
@inline _l2_core_bytes(::Type{T}) where {T} =
    _defaults_slot(T) === nothing ? _l2_core_bytes() : _resolved_defaults(T).l2_core

# What the cost model reads of one operand, gathered ONCE per plan in
# `klabels` order (`_k_operand`): each K label's extent `len[i]` and
# |stride| `st[i]` in the operand, the element count `n`, the element size
# `S`, and whether its register slivers are whole lines (`wholeline`: the
# free composite's leading -- C-fastest, first non-singleton of `free` --
# axis is unit-stride here and at least a line long). An order is then a
# permutation `perm` of `1:DK` (`klabels[perm[j]]` is its j-th label), and
# costing a candidate is plain tuple indexing: no `findfirst` over the
# operand's labels per axis visited, as a label-keyed evaluation would do six
# times over (three candidates, two operands).
struct _KOperand{DK}
    len::NTuple{DK, Int}
    st::NTuple{DK, Int}
    n::Int
    S::Int
    wholeline::Bool
end

@inline function _k_operand(
        klabels::NTuple{DK, Int}, ind::NTuple{N, Int}, v::StridedView,
        free::NTuple{DF, Int}
    ) where {DK, N, DF}
    st = Base.strides(v)
    sz = size(v)
    S = sizeof(eltype(v))
    pos(l::Int) = findfirst(==(l), ind)::Int
    ps = map(pos, klabels)
    len = map(p -> sz[p], ps)
    kst = map(p -> abs(st[p]), ps)
    line_elems = max(1, _K_LINE_BYTES ÷ S)
    wholeline = false
    for l in free
        p = pos(l)
        sz[p] == 1 && continue
        wholeline = st[p] == 1 && sz[p] >= line_elems
        break
    end
    return _KOperand{DK}(len, kst, length(v), S, wholeline)
end

# Stable insertion sort of the permutation `perm` by `key[perm[j]]`,
# ascending: with `perm = 1:DK` this is `_sort_labels_by_stride` on
# `klabels` expressed as positions (same comparisons, same tie rule), so
# `map(i -> klabels[i], _sort_perm(1:DK, op.st))` is the operand-sorted order.
@inline function _sort_perm(perm::NTuple{D, Int}, key::NTuple{D, Int}) where {D}
    out = perm
    @inbounds for i in 2:D
        x = out[i]
        kx = key[x]
        j = i - 1
        while j >= 1 && key[out[j]] > kx
            out = _tupleset(out, j + 1, out[j])
            j -= 1
        end
        out = _tupleset(out, j + 1, x)
    end
    return out
end

# Cost of packing one operand under the K order `perm` (positions into the
# `klabels` the operand facts were gathered in; see `_KOperand`). `Qfree` is
# the operand's free extent.
function _k_order_cost(
        perm::NTuple{DK, Int}, op::_KOperand{DK}, Qfree::Int, l2bytes::Int
    ) where {DK}
    len, st, S = op.len, op.st, op.S
    # Fastest non-singleton K axis of this order.
    kfast = 0
    @inbounds for i in perm
        if len[i] > 1
            kfast = i
            break
        end
    end
    kfast == 0 && return 0  # K is all singletons here: nothing walks.
    walk = @inbounds(st[kfast]) * S > _K_WALK_FAR_BYTES ? _K_WALK_FAR_PENALTY : 1

    # Whole-line slivers: every line is fetched once.
    op.wholeline && return op.n * walk

    # The operand's smallest-stride K axis `u` (first in this order among
    # equal strides), and how many of its elements share a line.
    u = 0
    su = typemax(Int)
    @inbounds for i in perm
        len[i] == 1 && continue
        if st[i] < su
            su = st[i]
            u = i
        end
    end
    # share = su*S >= line ? 1 : min(len[u], line ÷ (su*S)). As `u` is not a
    # singleton (len[u] >= 2; an empty axis makes n = 0 and every return 0),
    # share == 1 exactly when su*S > line/2, which is tested first so the
    # runtime division is only paid on the one return that uses its value.
    suS = su * S
    (2 * suS > _K_LINE_BYTES || u == kfast) && return op.n * walk

    # Lines touched before `u` advances: one per element (scattered slivers)
    # over the faster K extents and the whole free extent.
    faster = 1
    @inbounds for i in perm
        i == u && break
        faster *= len[i]
    end
    footprint = Int128(faster) * Int128(Qfree) * Int128(_K_LINE_BYTES)
    footprint <= l2bytes && return op.n * walk
    share = min(@inbounds(len[u]), _K_LINE_BYTES ÷ suS)
    return op.n * walk * share
end

# Label-keyed form: the cost of packing the operand `v` (labels `ind`) under
# the K order `korder`, where `free` is the operand's free-label list in the
# order the M/N composite enumerates it (C-stride order, `_order_free_labels`)
# -- its FIRST non-singleton label decides whether a register sliver is whole
# lines -- and `Qfree` the free extent. For probes and tests; the planner
# costs its candidates through the positional form above.
function _k_order_cost(
        korder::NTuple{DK, Int}, ind::NTuple{N, Int}, v::StridedView,
        free::NTuple{DF, Int}, Qfree::Int, l2bytes::Int
    ) where {DK, N, DF}
    return _k_order_cost(ntuple(identity, Val(DK)), _k_operand(korder, ind, v, free), Qfree, l2bytes)
end

# Entry point `plan_contract` calls; the decision itself is `_choose_k_order`
# (kept apart so a probe can wrap the entry point and log/toggle decisions
# without duplicating the model -- benchmark/probes/probe_network_korder.jl).
# Without an explicit `l2bytes` the core's L2 share (`_l2_core_bytes`) is
# used (`_choose_k_order` also takes `nothing` for it, resolved lazily).
#
# Tuple port (the label lists are `NTuple`s, see the top of this file): the
# model takes `klabels`, `morder` and `norder` as statically sized tuples,
# represents every candidate order as a permutation tuple of `1:DK`, and
# returns an `NTuple{DK,Int}` of the input's static length, so the K group's
# rank stays a compile-time constant and nothing allocates. Zero or one K
# label returns at a compile-time branch, before the model or the L2 lookup,
# so a plain GEMM pays nothing for it.
@inline function _order_contract_labels(
        klabels::NTuple{DK, Int},
        indA::NTuple{NA, Int}, A::StridedView, morder::NTuple{DM, Int},
        indB::NTuple{NB, Int}, B::StridedView, norder::NTuple{DN, Int},
        Qm::Int, Qn::Int
    ) where {DK, NA, NB, DM, DN}
    DK <= 1 && return klabels
    return _choose_k_order(klabels, indA, A, morder, indB, B, norder, Qm, Qn, nothing)
end
function _order_contract_labels(
        klabels::NTuple{DK, Int},
        indA::NTuple{NA, Int}, A::StridedView, morder::NTuple{DM, Int},
        indB::NTuple{NB, Int}, B::StridedView, norder::NTuple{DN, Int},
        Qm::Int, Qn::Int, l2bytes::Int
    ) where {DK, NA, NB, DM, DN}
    return _choose_k_order(klabels, indA, A, morder, indB, B, norder, Qm, Qn, l2bytes)
end

function _choose_k_order(
        klabels::NTuple{DK, Int},
        indA::NTuple{NA, Int}, A::StridedView, morder::NTuple{DM, Int},
        indB::NTuple{NB, Int}, B::StridedView, norder::NTuple{DN, Int},
        Qm::Int, Qn::Int, l2bytes::Union{Int, Nothing}
    ) where {DK, NA, NB, DM, DN}
    DK <= 1 && return klabels
    opA = _k_operand(klabels, indA, A, morder)
    opB = _k_operand(klabels, indB, B, norder)
    id = ntuple(identity, Val(DK))
    permA = _sort_perm(id, opA.st)   # == _sort_labels_by_stride(klabels, indA, A)
    permB = _sort_perm(id, opB.st)   # == _sort_labels_by_stride(klabels, indB, B)
    # Both sorts already give indA order (every gemm_ready/a_permuted/
    # b_permuted layout): the loop below would skip both candidates as equal
    # to `best`, so the answer is `klabels` without evaluating a cost.
    permA == id && permB == id && return klabels
    # `nothing`: the core's L2 share, looked up only now that a cost is needed
    # (a field of the cached defaults, `_l2_core_bytes(T)`).
    l2 = l2bytes === nothing ? _l2_core_bytes(eltype(A)) : l2bytes
    cost(perm) = _k_order_cost(perm, opA, Qm, l2) + _k_order_cost(perm, opB, Qn, l2)
    best = id
    bestcost = cost(id)
    for cand in (permA, permB)
        cand == best && continue
        c = cost(cand)
        if c < bestcost
            best = cand
            bestcost = c
        end
    end
    return map(i -> @inbounds(klabels[i]), best)
end

# Element count of the leading unit-stride run when `labels` (already ordered
# by `_order_free_labels`) is enumerated first-label-fastest into C: the first
# non-singleton label must have C-stride exactly +1 (`_unit_stride_rows` is
# `stride == 1`, a descending run does not qualify), and each following label
# extends the run only if its stride equals the run so far. Singleton axes are
# skipped (their coordinate never advances, whatever their stride says).
# Returns 1 when no run starts, 0 if an empty axis is met first.
function _leading_unit_run(
        labels::NTuple{D, Int}, indC::NTuple{NC, Int}, C::StridedView
    ) where {D, NC}
    st = Base.strides(C)
    run = 1
    for l in labels
        p = findfirst(==(l), indC)::Int
        L = size(C, p)
        L == 1 && continue
        L == 0 && return 0
        st[p] == run || break
        run *= L
    end
    return run
end

# Whether to swap the operand roles (B feeds M, A feeds N), given the two
# composites' OWN leading unit-stride run lengths (`_leading_unit_run` above;
# `plan_contract` computes `run_m`/`run_n` once and reuses them here and at
# both `_demote_for_run` call sites).
# The vectorized store (`_vector_store_eligible`) needs a register sliver --
# `mr(kernel)` consecutive M coordinates -- to be unit-stride in C, so a
# leading run shorter than `mr` buys nothing (measured: swapping onto a
# 16-wide run under a 32-wide kernel is a ~1.2x REGRESSION). Swap only when the
# as-is orientation misses that bar and the swapped one clears it. The two `mr`
# arguments are the widths of the kernel each orientation would actually run
# (they differ only when the default kernel's small-Qm demotion applies to one
# side).
#
# Callers additionally restrict this to real dtypes. That restriction was
# measured (a ~2-4% regression on `ccsd_t_3`, dim=16, both complex dtypes:
# the swap loses the as-is orientation's N-side locality for no store-side
# gain) only against a scattered/scalar complex store; the planar vectorized
# store (`_store_tile_planar_vector!`, `src/microkernels/planar.jl`) gives the
# swap something to win on the complex path too. The `T <: Real` guard is
# therefore an UNMEASURED, DELIBERATELY DEFERRED question, not a settled case:
# lifting it needs its own before/after measurement. See the `T <: Real` guard
# at the call site.
function _prefer_swap(run_m::Int, run_n::Int, mr_asis::Int, mr_swapped::Int = mr_asis)
    return run_m < mr_asis && run_n >= mr_swapped
end

# Label-list form, for callers that have `morder`/`norder` but not their run
# lengths; derives the same two run lengths `plan_contract` computes once.
function _prefer_swap(
        morder::NTuple{DM, Int}, norder::NTuple{DN, Int}, indC::NTuple{NC, Int},
        C::StridedView, mr_asis::Int, mr_swapped::Int = mr_asis
    ) where {DM, DN, NC}
    return _prefer_swap(
        _leading_unit_run(morder, indC, C), _leading_unit_run(norder, indC, C),
        mr_asis, mr_swapped
    )
end
