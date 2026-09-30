# Packing by cache lines. When a register sliver reads one element per line and
# the line's other elements belong to coordinates a whole sweep of the free
# composite later, lines rarely survive until reused; at power-of-two extents
# they alias in a few cache sets and the pack costs a memory latency per
# element. A split plan enumerates the free composite so that a macro block
# holds whole lines, and packs each block line by line
# (`_pack_block_transposed!`); the packed format and the K order are unchanged.

"""
    PackSplit(q, dj, L, kinner, Eq, E)

How one operand's free group is enumerated and packed under a split: axis `q`
in chunks of `Eq`, the unit-stride axis `dj > q` in chunks of `L` (a line's
worth) placed right after the `Eq` chunk. A group is then `E x L`
coordinates, `E` = (lengths before `q`) * `Eq`, and the block walk reads its
lines K step by K step, or with K innermost when `kinner`. `L == 0`: no split.
"""
struct PackSplit
    # Narrow fields: the plan carries two of these through every call.
    q::UInt8
    dj::UInt8
    L::UInt8
    kinner::Bool
    Eq::Int32
    E::Int32
end

const _NO_SPLIT = PackSplit(0, 0, 0, false, 0, 0)

_is_split(s::PackSplit) = s.L != 0

# The split for the operand behind map 1 of `g` (A for M, B for N) and the block
# extent it needs, or `(eff, _NO_SPLIT)`. `eff` is the plan's extent, `rounded`
# the requested one rounded to `R`; a split block may take the budget the
# blocking reserved for `k_block_requested` when the K extent clamps `k_block`.
# `l2bytes = nothing`: `_split_capacity`, doubled for complex, whose block
# walk costs more per element.
@inline function _pack_split(
        g::AxisGroup{D}, kg::AxisGroup, kmap::Int, R::Int, S::Int, complex::Bool, k_block::Int,
        eff::Int, rounded::Int, k_block_requested::Int, l2bytes::Union{Int, Nothing} = nothing
    ) where {D}
    D < 256 || return (eff, _NO_SPLIT)
    line = _K_LINE_BYTES
    d1 = 0
    for d in 1:D
        g.lengths[d] > 1 && (d1 = d; break)
    end
    (d1 > 0 && abs(g.strides[1][d1]) * S >= line) || return (eff, _NO_SPLIT)
    dj = 0
    for d in (d1 + 1):D
        g.lengths[d] > 1 && abs(g.strides[1][d]) == 1 && (dj = d; break)
    end
    dj == 0 && return (eff, _NO_SPLIT)
    return _pack_split_window(g, kg, kmap, d1, dj, R, S, complex, k_block, eff, rounded, k_block_requested, l2bytes)
end

# Out of line: shared by every eltype, and most plans never get here.
@noinline function _pack_split_window(
        g::AxisGroup, kg::AxisGroup{DK}, kmap::Int, d1::Int, dj::Int, R::Int, S::Int, complex::Bool,
        k_block::Int, eff::Int, rounded::Int, k_block_requested::Int, l2bytes::Union{Int, Nothing}
    ) where {DK}
    line = _K_LINE_BYTES
    L = _largest_divisor_upto(g.lengths[dj], max(1, line ÷ S))
    L >= 2 || return (eff, _NO_SPLIT)

    ks = 0
    for d in 1:DK
        kg.lengths[d] > 1 && (ks = abs(kg.strides[kmap][d]); break)
    end
    # K steps within a page make each sliver element's lines a stream the
    # prefetchers follow; a split then only saves refetching those lines, which
    # does not pay for the block walk's costlier complex scatter.
    kinner = ks * S < _K_WALK_FAR_BYTES
    kinner && complex && return (eff, _NO_SPLIT)

    # A line of `dj` is reused `psi` coordinates later; while the lines touched
    # meanwhile fit the cache, the per-sliver walk is already cache-friendly.
    psi = 1
    for d in d1:(dj - 1)
        psi *= g.lengths[d]
    end
    klines = ks * S >= line ? k_block : cld(k_block * ks * S, line)
    aliased = abs(g.strides[1][d1]) * S % _K_WALK_FAR_BYTES == 0
    cap = something(l2bytes, _split_capacity(target_profile(), aliased) << complex)
    widemul(psi, max(1, klines) * line) > cap || return (eff, _NO_SPLIT)
    return _pack_split_dynamic(g.lengths, d1, dj, L, psi, R, eff, rounded, k_block, k_block_requested, kinner)
end

# The cache that can hold the per-sliver walk's reuse window: the core's L2
# share, plus its L3 share unless the sliver steps a whole number of pages,
# which folds the lines it gathers into a few sets of every level.
function _split_capacity(profile::TargetProfile, aliased::Bool)
    l2 = _l2_core_bytes(profile)
    l3 = profile.l3
    (aliased || l3.bytes <= 0) && return l2
    return l2 + l3.bytes ÷ max(1, l3.sharing ÷ max(1, profile.l1d.sharing))
end

# Past the tests above the plan is for a large contraction; a dynamic call keeps
# the search from being compiled for every other plan.
@noinline _pack_split_dynamic(args...) = Base.inferencebarrier(_pack_split_blocks)(args...)::Tuple{Int, PackSplit}

function _pack_split_blocks(
        lengths::NTuple{D, Int}, d1::Int, dj::Int, L::Int, psi::Int, R::Int,
        eff::Int, rounded::Int, k_block::Int, k_block_requested::Int, kinner::Bool
    ) where {D}
    budget = max(rounded, Int(min(widemul(rounded, k_block_requested) ÷ k_block, typemax(Int32))) ÷ R * R)
    # Prefixes of `d1:dj-1`: whole axes, then a divisor chunk of the next one.
    # `E` is also the run of consecutive C coordinates a block stores, so take
    # the largest whose whole groups fit the budget, preferring `R | E`. A
    # prefix shorter than a sliver would make slivers straddle chunks.
    best = anyc = (0, 0, 0)
    pre = 1
    for q in d1:(dj - 1)
        len = lengths[q]
        limit = budget ÷ (pre * L)
        limit >= 1 || break
        for Eq in min(len, limit):-1:1
            len % Eq == 0 || continue
            E = pre * Eq
            ((E >= R || E == psi) && lcm(R, E * L) <= budget) || continue
            E > anyc[3] && (anyc = (q, Eq, E))
            E % R == 0 && E > best[3] && (best = (q, Eq, E))
        end
        len <= limit || break
        pre *= len
    end
    (q, Eq, E) = best[3] > 0 ? best : anyc
    E == 0 && return (eff, _NO_SPLIT)
    blk = lcm(R, E * L)
    neweff = min(budget ÷ blk * blk, _roundup(_unchecked_axis_length(lengths), R))
    return (neweff, PackSplit(q, dj, L, kinner, Eq, E))
end

@inline function _largest_divisor_upto(n::Int, cap::Int)
    for L in min(n, cap):-1:1
        n % L == 0 && return L
    end
    return 1
end

# `g` in the enumeration `s` describes: `q` and `dj` each replaced by an (inner,
# outer) pair of axes, the inner chunk of `dj` moved right after that of `q`.
function _split_group(g::AxisGroup{D, P}, s::PackSplit) where {D, P}
    ax(k, p) = _split_axis(g, s, k, p)
    return AxisGroup(
        ntuple(k -> ax(k, 0), Val(D + 2)), ntuple(p -> ntuple(k -> ax(k, p), Val(D + 2)), Val(P))
    )
end

# Length (`p == 0`) or map-`p` stride of axis `k` of `_split_group(g, s)`.
@inline function _split_axis(g::AxisGroup, s::PackSplit, k::Int, p::Int)
    len(d) = g.lengths[d]
    str(d) = g.strides[p][d]
    q, dj, Eq, L = Int(s.q), Int(s.dj), Int(s.Eq), Int(s.L)
    k < q && return p == 0 ? len(k) : str(k)
    k == q && return p == 0 ? Eq : str(q)
    k == q + 1 && return p == 0 ? L : str(dj)
    k == q + 2 && return p == 0 ? len(q) ÷ Eq : Eq * str(q)
    k <= dj + 1 && return p == 0 ? len(k - 2) : str(k - 2)
    k == dj + 2 && return p == 0 ? len(dj) ÷ L : L * str(dj)
    return p == 0 ? len(k - 2) : str(k - 2)
end
