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

# Smallest `abs(stride)` among the K labels' non-singleton axes in one operand
# (`typemax(Int)` when every K axis is a singleton, i.e. K does not walk that
# operand at all).
function _min_label_stride(
        labels::Vector{Int}, ind::NTuple{N, Int}, v::StridedView
    ) where {N}
    st = Base.strides(v)
    best = typemax(Int)
    for l in labels
        p = findfirst(==(l), ind)::Int
        size(v, p) == 1 && continue
        best = min(best, abs(st[p]))
    end
    return best
end

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

# The K order: sort by `abs(stride)` in the operand in which K is the more
# "inner" group, i.e. has the smaller minimum K stride. That operand's pack has
# no contiguous free axis to hide behind (its sliver rows are far apart in
# memory), so its line locality has to come from consecutive K steps landing
# in the same line; the other operand's free axis supplies the locality
# instead (an `mr`-row unit-stride sliver per K step is already whole cache
# lines, whatever the K stride). Ties -- K inner in both, `A[k..,m] * B[k..,n]`
# with the K axes in different orders -- go to the operand with more elements
# to read (the larger free extent), then to A. Stable, so labels with equal
# strides keep `indA` order, and any layout whose K strides are already
# ascending in the chosen operand (every gemm_ready/a_permuted/b_permuted
# layout of the upstream suite) gets exactly the order it had before this rule
# existed. The order is a property of the pair (A, B) and is the same whether
# or not the M/N orientation is later swapped.
function _order_contract_labels(
        klabels::Vector{Int},
        indA::NTuple{NA, Int}, A::StridedView,
        indB::NTuple{NB, Int}, B::StridedView,
        Qm::Int, Qn::Int
    ) where {NA, NB}
    length(klabels) <= 1 && return klabels
    minA = _min_label_stride(klabels, indA, A)
    minB = _min_label_stride(klabels, indB, B)
    by_b = minB < minA || (minB == minA && Qn > Qm)
    return by_b ? _sort_labels_by_stride(klabels, indB, B) :
        _sort_labels_by_stride(klabels, indA, A)
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
