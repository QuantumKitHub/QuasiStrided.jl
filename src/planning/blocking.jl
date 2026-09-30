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

# Complex blocking is the real row divided by the packed reals per element of
# each operand, so every method gets the same packed BYTE budget (1m's
# `m_block` is half planar's).
@inline _scale_blocking(base::Blocking, m::ComplexMethod) =
    Blocking(max(1, base.m_block ÷ a_reals(m)), base.k_block, max(1, base.n_block ÷ b_reals(m)))

# For a host whose L1d or L2 size is undetected.
_fallback_blocking(::Type{Float64}) = Blocking(128, 256, 768)
_fallback_blocking(::Type{Float32}) = Blocking(96, 768, 1152)

_real_blocking_row(profile::TargetProfile, ::Type{T}) where {T <: Union{Float32, Float64}} =
    something(_modelled_blocking(profile, T), _fallback_blocking(T))

# Analytical real row from the cache geometry. Per (N, K) block the packed B
# panel is `n_block*k_block` elements, per (N, K, M) block the A block
# `m_block*k_block`, and one `NR x k_block` B sliver is reused by every A
# sliver of the block:
#
#     NR*k_block*S       <= L1/2              the reused B sliver
#     m_block*k_block*S  <= L2core/2          the A block
#     n_block*k_block*S  <= L2core + L3core   the B panel
#
# rounded down to MR/NR multiples, shared levels divided by the cores sharing
# them. `n_block` takes the whole per-core capacity, not half the L3: an
# oversized B panel only re-streams B, an undersized one repacks A once per N
# block.
# `nothing` when L1d or L2 is undetected.
function _modelled_blocking(profile::TargetProfile, ::Type{T}, MR::Int, NR::Int) where {T}
    l1, l2, l3 = profile.l1d, profile.l2, profile.l3
    (l1.bytes > 0 && l2.bytes > 0) || return nothing
    smt = max(1, l1.sharing)
    core_share(c) = c.bytes ÷ max(1, c.sharing ÷ smt)
    l2core = core_share(l2)
    l3core = l3.bytes > 0 ? core_share(l3) : 0
    S = sizeof(T)
    k_block = max(1, (l1.bytes ÷ 2) ÷ (NR * S))
    m_block = max(MR, ((l2core ÷ 2) ÷ (k_block * S)) ÷ MR * MR)
    n_block = max(NR, ((l2core + l3core) ÷ (k_block * S)) ÷ NR * NR)
    return Blocking(m_block, k_block, n_block)
end

function _modelled_blocking(profile::TargetProfile, ::Type{T}) where {T <: Real}
    MR, NR, _ = _derived_shape(profile, T)
    return _modelled_blocking(profile, T, MR, NR)
end

# For a bare scalar type: the fallback row, independent of the host.
default_blocking(::Type{Float64}) = _fallback_blocking(Float64)
default_blocking(::Type{Float32}) = _fallback_blocking(Float32)

# Smallest multiple of `n` that is >= `x` (`x >= 0`, `n >= 1`).
@inline _roundup(x::Int, n::Int) = cld(x, n) * n
