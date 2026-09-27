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
