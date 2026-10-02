"""
    Blocking(m_block::Int, k_block::Int, n_block::Int)

Cache-blocking factors for the macro-blocking driver: `m_block` (M block
extent), `k_block` (K block/panel depth) and `n_block` (N block extent).
All three fields must be `>= 1`; the constructor throws `ArgumentError`
otherwise.
"""
struct Blocking
    m_block::Int
    k_block::Int
    n_block::Int

    function Blocking(m_block::Int, k_block::Int, n_block::Int)
        m_block >= 1 || throw(ArgumentError("Blocking requires m_block >= 1, got m_block = $m_block"))
        k_block >= 1 || throw(ArgumentError("Blocking requires k_block >= 1, got k_block = $k_block"))
        n_block >= 1 || throw(ArgumentError("Blocking requires n_block >= 1, got n_block = $n_block"))
        return new(m_block, k_block, n_block)
    end
end

"""
    default_blocking(kernel, profile = target_profile()) -> Blocking

Cache-blocking factors for `kernel` on `profile`: an analytical model of the
cache geometry, or fixed constants where L1d or L2 is undetected.
`plan_contract` rounds `m_block`/`n_block` to multiples of the tile size and
clamps all three to the contraction's extents.
"""
default_blocking(kernel, profile::TargetProfile = target_profile()) =
    kernel_blocking(profile, scalartype(kernel), typeof(kernel), tile_size(kernel)...)

# The default blocking of a kernel of type `K` for `T` with an `MR x NR` tile.
# The fallback rows are divided by the packed reals per element of each
# operand, so every kernel type gets the same packed byte budget.
@inline function kernel_blocking(profile::TargetProfile, ::Type{T}, ::Type{K}, MR::Int, NR::Int) where {T, K}
    R = real(T)
    a_reals, b_reals = map(reals_per_element, pack_formats(K))
    modelled = modelled_blocking(profile, MR, NR, a_reals * sizeof(R), b_reals * sizeof(R))
    modelled === nothing || return modelled
    base = fallback_blocking(R)
    return Blocking(max(1, base.m_block ÷ a_reals), base.k_block, max(1, base.n_block ÷ b_reals))
end

# For a host whose L1d or L2 size is undetected.
fallback_blocking(::Type{Float64}) = Blocking(128, 256, 768)
fallback_blocking(::Type{Float32}) = Blocking(96, 768, 1152)

# Analytical blocking from the cache geometry, with `SA`/`SB` the packed bytes
# per element of A and B. Per (N, K) block the packed B panel is
# `n_block*k_block` elements, per (N, K, M) block the A block
# `m_block*k_block`, and one `NR x k_block` B sliver is reused by every A
# sliver of the block:
#
#     NR*k_block*SB       <= L1/2              the reused B sliver
#     m_block*k_block*SA  <= L2core/2          the A block
#     n_block*k_block*SB  <= L2core + L3core   the B panel
#
# rounded down to MR/NR multiples, shared levels divided by the cores sharing
# them. `n_block` takes the whole per-core capacity, not half the L3: an
# oversized B panel only re-streams B, an undersized one repacks A once per N
# block.
# `nothing` when L1d or L2 is undetected.
@inline function modelled_blocking(profile::TargetProfile, MR::Int, NR::Int, SA::Int, SB::Int)
    l1 = profile.l1d
    (l1.bytes > 0 && profile.l2.bytes > 0) || return nothing
    l2core, l3core = profile.l2_share, profile.l3_share
    k_block = max(1, (l1.bytes ÷ 2) ÷ (NR * SB))
    m_block = max(MR, ((l2core ÷ 2) ÷ (k_block * SA)) ÷ MR * MR)
    n_block = max(NR, ((l2core + l3core) ÷ (k_block * SB)) ÷ NR * NR)
    return Blocking(m_block, k_block, n_block)
end

# Smallest multiple of `n` that is >= `x` (`x >= 0`, `n >= 1`).
@inline roundup(x::Int, n::Int) = cld(x, n) * n
