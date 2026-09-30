# Label planning: classify every label into M/N/K, order the labels within each
# composite, and decide the M/N operand orientation. Every label list is an
# `NTuple` whose length follows from the label tuples' lengths alone, so
# planning allocates nothing and every `AxisGroup` is concretely typed.

@inline _label_in(lbl::Int, t::NTuple{N, Int}) where {N} = any(==(lbl), t)

# The M/N/K ranks: with A = M ∪ K, B = N ∪ K, C = M ∪ N, solve
# NA = |M| + |K|, NB = |N| + |K|, NC = |M| + |N|. Meaningful only for input that
# `_classify_labels` accepts, which asserts them against the counts it finds.
@inline _group_ranks(NA::Int, NB::Int, NC::Int) =
    ((NA + NC - NB) ÷ 2, (NB + NC - NA) ÷ 2, (NA + NB - NC) ÷ 2)

@noinline _throw_rank_mismatch(which::Symbol, found::Int, derived::Int) = throw(
    ArgumentError(
        "internal error: $which label count $found does not match the rank $derived " *
            "derived from the label tuple lengths"
    )
)

# Set element `n` of `out`, ignoring an `n` past the end (input about to be rejected).
@inline function _push_label(out::NTuple{D, Int}, n::Int, x::Int) where {D}
    return n <= D ? _tupleset(out, n, x) : out
end

# (mlabels, nlabels, klabels) in indA/indB order. Per (inA, inB, inC):
# (T,F,T) -> M, (F,T,T) -> N, (T,T,F) -> K, anything else is an ArgumentError.
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
        if inA
            continue  # K, or already rejected while scanning indA
        elseif inC
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

    # Unreachable after the checks above; guards against a silently truncated list.
    nm == DM || _throw_rank_mismatch(:M, nm, DM)
    nn == DN || _throw_rank_mismatch(:N, nn, DN)
    nk == DK || _throw_rank_mismatch(:K, nk, DK)
    return mlabels, nlabels, klabels
end

@noinline _throw_label_length(lbl::Int, l1::Int, l2::Int) = throw(
    DimensionMismatch("label $lbl has mismatched axis length: $l1 vs $l2")
)

# Stable insertion sort of the permutation `perm` by `key[perm[j]]`, ascending
# (n <= ndims, and unlike `sortperm` it allocates nothing).
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

# `labels` stably sorted by `abs(stride)` of their axis in `C` (every label must
# be in `indC`): a composite enumerates its first label fastest, so this walks
# C's fastest axis fastest.
function _order_free_labels(
        labels::NTuple{D, Int}, indC::NTuple{NC, Int}, C::StridedView
    ) where {D, NC}
    st = Base.strides(C)
    key = map(l -> abs(st[findfirst(==(l), indC)::Int]), labels)
    return map(i -> @inbounds(labels[i]), _sort_perm(ntuple(identity, Val(D)), key))
end

# ----------------------------------------------------------------------------
# Contracted-label order. Both packs step through K in the K composite's order,
# so when the K axes sit in a different relative order in A and B no single
# order suits both, and indA order can make one pack jump a page per element.
# The order is chosen among indA order, sorted by A's strides and sorted by B's
# strides by a cost model of the two packs; ties keep indA order. Per operand
# (`n` elements):
#
#     cost = n * walk * amplification
#
#   * `walk` is `_K_WALK_FAR_PENALTY` when the fastest K axis steps more than a
#     page (a chain of demand misses no prefetcher follows), else 1.
#   * `amplification` is how often a cache line is fetched: 1 when the slivers
#     are whole lines, when the operand's smallest-stride axis `u` is the
#     fastest K axis, or when the lines touched before `u` advances (faster K
#     extents x free extent) fit the core's L2; otherwise
#     `min(L_u, line_bytes / stride_u)`.
# ----------------------------------------------------------------------------

const _K_WALK_FAR_BYTES = 4096
const _K_WALK_FAR_PENALTY = 3
const _K_LINE_BYTES = 64

# The core's private L2 share, or 1 MB when undetected.
function _l2_core_bytes(profile::TargetProfile)
    profile.l2.bytes > 0 || return 1 << 20
    return core_bytes(profile, profile.l2)
end

# Through the per-eltype cache; an eltype without a slot fails later in planning.
@inline _l2_core_bytes(::Type{T}) where {T} =
    _defaults_slot(T) === nothing ? _l2_core_bytes(target_profile()) : _resolved_defaults(T).l2_core

# What the cost model reads of one operand, gathered once per plan in `klabels`
# order: each K label's extent and |stride|, the element count and size, and
# whether its register slivers are whole lines (the free composite's first
# non-singleton axis is unit-stride and at least a line long). A candidate
# order is then a permutation of `1:DK`.
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

# Cost of packing one operand under the K order `perm`; `Qfree` is its free extent.
function _k_order_cost(
        perm::NTuple{DK, Int}, op::_KOperand{DK}, Qfree::Int, l2bytes::Int
    ) where {DK}
    len, st, S = op.len, op.st, op.S
    kfast = 0
    @inbounds for i in perm
        if len[i] > 1
            kfast = i
            break
        end
    end
    kfast == 0 && return 0  # all-singleton K: nothing walks
    walk = @inbounds(st[kfast]) * S > _K_WALK_FAR_BYTES ? _K_WALK_FAR_PENALTY : 1
    op.wholeline && return op.n * walk

    # The smallest-stride K axis `u` (first in this order among equal strides).
    u = 0
    su = typemax(Int)
    @inbounds for i in perm
        len[i] == 1 && continue
        if st[i] < su
            su = st[i]
            u = i
        end
    end
    # One element of `u` per line exactly when su*S > line/2 (len[u] >= 2);
    # tested first so the division is only paid where its value is used.
    suS = su * S
    (2 * suS > _K_LINE_BYTES || u == kfast) && return op.n * walk

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

# Zero or one K label returns at a compile-time branch, before the model.
@inline function _order_contract_labels(
        klabels::NTuple{DK, Int},
        indA::NTuple{NA, Int}, A::StridedView, morder::NTuple{DM, Int},
        indB::NTuple{NB, Int}, B::StridedView, norder::NTuple{DN, Int},
        m_length::Int, n_length::Int
    ) where {DK, NA, NB, DM, DN}
    DK <= 1 && return klabels
    return _choose_k_order(klabels, indA, A, morder, indB, B, norder, m_length, n_length, nothing)
end

# `l2bytes = nothing`: the core's L2 share, looked up only once a cost is needed.
function _choose_k_order(
        klabels::NTuple{DK, Int},
        indA::NTuple{NA, Int}, A::StridedView, morder::NTuple{DM, Int},
        indB::NTuple{NB, Int}, B::StridedView, norder::NTuple{DN, Int},
        m_length::Int, n_length::Int, l2bytes::Union{Int, Nothing}
    ) where {DK, NA, NB, DM, DN}
    DK <= 1 && return klabels
    opA = _k_operand(klabels, indA, A, morder)
    opB = _k_operand(klabels, indB, B, norder)
    id = ntuple(identity, Val(DK))
    permA = _sort_perm(id, opA.st)
    permB = _sort_perm(id, opB.st)
    permA == id && permB == id && return klabels
    l2 = l2bytes === nothing ? _l2_core_bytes(eltype(A)) : l2bytes
    cost(perm) = _k_order_cost(perm, opA, m_length, l2) + _k_order_cost(perm, opB, n_length, l2)
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

# Length of the leading unit-stride run in C when `labels` is enumerated
# first-label-fastest: the first non-singleton label needs C-stride exactly +1,
# and each following one extends the run only if its stride equals the run so
# far. 1 when no run starts, 0 if an empty axis is met first.
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

# Swap the operand roles (B feeds M) when the as-is M run cannot fill a register
# sliver of the kernel it would run and the swapped one can: the vectorized
# store needs `m_tile` consecutive M coordinates unit-stride in C, so a shorter
# run buys nothing.
_prefer_swap(run_m::Int, run_n::Int, m_tile_asis::Int, m_tile_swapped::Int = m_tile_asis) =
    run_m < m_tile_asis && run_n >= m_tile_swapped
