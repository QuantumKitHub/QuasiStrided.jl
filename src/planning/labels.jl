# Label planning: classify every label into M/N/K, order the free labels
# by their stride in C, and decide the M/N operand orientation.

# Membership test against a statically-sized label tuple, used instead of a
# `Set`: the label tuples have a compile-time-known LENGTH (the `NA`/`NB`/`NC`
# parameters, one specialization per arity), so this unrolls into a chain of
# integer compares and allocates nothing, where each `Set` would cost a
# `Dict`'s slot/key arrays on every plan. The tuples are `allunique` by the
# checks at the top of `_classify_labels`, so a linear scan is also the whole
# of the membership question.
@inline _label_in(lbl::Int, t::NTuple{N, Int}) where {N} = any(==(lbl), t)

# Classify every label in indA ∪ indB ∪ indC into M/N/K. Returns
# (mlabels, nlabels, klabels) in indA/indB appearance order. Per (inA,inB,inC):
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

    # Sized once to their worst case and trimmed at the end, rather than grown
    # by `push!`: `NA`/`NB` are compile-time bounds on the M+K and N counts.
    mlabels = Vector{Int}(undef, NA)
    klabels = Vector{Int}(undef, NA)
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
            @inbounds klabels[nk] = lbl
        elseif !inB && inC
            nm += 1
            @inbounds mlabels[nm] = lbl
        else
            throw(
                ArgumentError(
                    "label $lbl appears only in indA (not in indB or indC): not a valid " *
                        "free (M) or contracted (K) label"
                )
            )
        end
    end

    nlabels = Vector{Int}(undef, NB)
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
            @inbounds nlabels[nn] = lbl
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

    resize!(mlabels, nm)
    resize!(nlabels, nn)
    resize!(klabels, nk)
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
# the key vector, the permutation and the result (three `Vector`s where one is
# needed), and these lists have at most `ndims(C)` entries, so an O(n^2) sort
# with n <= 6 is not a cost. Strict `>` in the shift test keeps it STABLE,
# which is the contract (ties keep input order). A fresh vector is returned:
# sorting `labels` in place would mutate `_classify_labels`'s output, which
# callers (and test/planning/test_plan_contract.jl's label-order pinning) read afterwards.
function _order_free_labels(
        labels::Vector{Int}, indC::NTuple{NC, Int}, C::StridedView
    ) where {NC}
    st = Base.strides(C)
    key(l::Int) = abs(st[findfirst(==(l), indC)::Int])
    out = copy(labels)
    @inbounds for i in 2:length(out)
        x = out[i]
        kx = key(x)
        j = i - 1
        while j >= 1 && key(out[j]) > kx
            out[j + 1] = out[j]
            j -= 1
        end
        out[j + 1] = x
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

# Stable sort of `labels` by `abs(stride)` of each label's axis in `v` (the
# same insertion sort as `_order_free_labels`, over a different operand).
function _sort_labels_by_stride(
        labels::Vector{Int}, ind::NTuple{N, Int}, v::StridedView
    ) where {N}
    st = Base.strides(v)
    key(l::Int) = abs(st[findfirst(==(l), ind)::Int])
    out = copy(labels)
    @inbounds for i in 2:length(out)
        x = out[i]
        kx = key(x)
        j = i - 1
        while j >= 1 && key(out[j]) > kx
            out[j + 1] = out[j]
            j -= 1
        end
        out[j + 1] = x
    end
    return out
end

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
# of the machines measured, so an unknown host errs neither way.
function _l2_core_bytes()
    profile = target_profile()
    l2 = profile.l2
    l2.bytes > 0 || return 1 << 20
    smt = max(1, profile.l1d.sharing)
    return l2.bytes ÷ max(1, l2.sharing ÷ smt)
end

# Cost of packing one operand under K order `korder`. `free` is the operand's
# free-label list in the order the M/N composite enumerates it (C-stride
# order, `_order_free_labels`), whose FIRST label decides whether a register
# sliver is whole lines. `Qfree` is the free extent.
function _k_order_cost(
        korder::Vector{Int}, ind::NTuple{N, Int}, v::StridedView,
        free::Vector{Int}, Qfree::Int, l2bytes::Int
    ) where {N}
    st = Base.strides(v)
    S = sizeof(eltype(v))
    line_elems = max(1, _K_LINE_BYTES ÷ S)
    pos(l::Int) = findfirst(==(l), ind)::Int
    n = length(v)

    # Fastest non-singleton K axis of this order.
    kfast = 0
    for l in korder
        if size(v, pos(l)) > 1
            kfast = l
            break
        end
    end
    kfast == 0 && return 0  # K is all singletons here: nothing walks.
    walk = abs(st[pos(kfast)]) * S > _K_WALK_FAR_BYTES ? _K_WALK_FAR_PENALTY : 1

    # Whole-line slivers: the free composite's leading (C-fastest) axis is
    # unit-stride here and at least a line long.
    wholeline = false
    for l in free
        p = pos(l)
        size(v, p) == 1 && continue
        wholeline = st[p] == 1 && size(v, p) >= line_elems
        break
    end
    wholeline && return n * walk

    # The operand's smallest-stride K axis `u`, and how many of its elements
    # share a line.
    u = 0
    su = typemax(Int)
    for l in korder
        p = pos(l)
        size(v, p) == 1 && continue
        if abs(st[p]) < su
            su = abs(st[p])
            u = l
        end
    end
    share = su * S >= _K_LINE_BYTES ? 1 : min(size(v, pos(u)), _K_LINE_BYTES ÷ (su * S))
    (share == 1 || u == kfast) && return n * walk

    # Lines touched before `u` advances: one per element (scattered slivers)
    # over the faster K extents and the whole free extent.
    faster = 1
    for l in korder
        l == u && break
        faster *= size(v, pos(l))
    end
    footprint = Int128(faster) * Int128(Qfree) * Int128(_K_LINE_BYTES)
    return footprint <= l2bytes ? n * walk : n * walk * share
end

# Entry point `plan_contract` calls; the decision itself is `_choose_k_order`
# (kept apart so a probe can wrap the entry point and log/toggle decisions
# without duplicating the model -- benchmark/probes/probe_network_korder.jl).
function _order_contract_labels(
        klabels::Vector{Int},
        indA::NTuple{NA, Int}, A::StridedView, morder::Vector{Int},
        indB::NTuple{NB, Int}, B::StridedView, norder::Vector{Int},
        Qm::Int, Qn::Int, l2bytes::Int = _l2_core_bytes()
    ) where {NA, NB}
    return _choose_k_order(klabels, indA, A, morder, indB, B, norder, Qm, Qn, l2bytes)
end

function _choose_k_order(
        klabels::Vector{Int},
        indA::NTuple{NA, Int}, A::StridedView, morder::Vector{Int},
        indB::NTuple{NB, Int}, B::StridedView, norder::Vector{Int},
        Qm::Int, Qn::Int, l2bytes::Int
    ) where {NA, NB}
    length(klabels) <= 1 && return klabels
    cost(order) = _k_order_cost(order, indA, A, morder, Qm, l2bytes) +
        _k_order_cost(order, indB, B, norder, Qn, l2bytes)
    best = klabels
    bestcost = cost(klabels)
    for cand in (_sort_labels_by_stride(klabels, indA, A), _sort_labels_by_stride(klabels, indB, B))
        cand == best && continue
        c = cost(cand)
        if c < bestcost
            best = cand
            bestcost = c
        end
    end
    return best
end

# Element count of the leading unit-stride run when `labels` (already ordered
# by `_order_free_labels`) is enumerated first-label-fastest into C: the first
# non-singleton label must have C-stride exactly +1 (`_unit_stride_rows` is
# `stride == 1`, a descending run does not qualify), and each following label
# extends the run only if its stride equals the run so far. Singleton axes are
# skipped (their coordinate never advances, whatever their stride says).
# Returns 1 when no run starts, 0 if an empty axis is met first.
function _leading_unit_run(
        labels::Vector{Int}, indC::NTuple{NC, Int}, C::StridedView
    ) where {NC}
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
        morder::Vector{Int}, norder::Vector{Int}, indC::NTuple{NC, Int}, C::StridedView,
        mr_asis::Int, mr_swapped::Int = mr_asis
    ) where {NC}
    return _prefer_swap(
        _leading_unit_run(morder, indC, C), _leading_unit_run(norder, indC, C),
        mr_asis, mr_swapped
    )
end
