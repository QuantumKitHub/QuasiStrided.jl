"""
    Blocking(mc::Int, kc::Int, nc::Int)

Cache-blocking factors for the macro-blocking driver: `mc` (M block extent,
loop 3), `kc` (K block/panel depth, loop 4), `nc` (N block extent, loop 5).
All three fields must be `>= 1`; the constructor throws `ArgumentError`
otherwise.
"""
struct Blocking
    mc::Int
    kc::Int
    nc::Int

    function Blocking(mc::Int, kc::Int, nc::Int)
        mc >= 1 || throw(ArgumentError("Blocking requires mc >= 1, got mc = $mc"))
        kc >= 1 || throw(ArgumentError("Blocking requires kc >= 1, got kc = $kc"))
        nc >= 1 || throw(ArgumentError("Blocking requires nc >= 1, got nc = $nc"))
        return new(mc, kc, nc)
    end
end

# Complex blocking is the real row divided by the packed reals per element of
# each operand, so every method gets the same packed BYTE budget (1m's `mc` is
# half planar's).
@inline _scale_blocking(base::Blocking, m::ComplexMethod) =
    Blocking(max(1, base.mc ÷ a_reals(m)), base.kc, max(1, base.nc ÷ b_reals(m)))

# For a host whose L1d or L2 size is undetected.
_fallback_blocking(::Type{Float64}) = Blocking(128, 256, 768)
_fallback_blocking(::Type{Float32}) = Blocking(96, 768, 1152)

_real_blocking_row(profile::TargetProfile, ::Type{T}) where {T <: Union{Float32, Float64}} =
    something(_modelled_blocking(profile, T), _fallback_blocking(T))

# Analytical real row from the cache geometry. Per (jc, pc) the packed B panel
# is `nc*kc` elements, per (jc, pc, ic) the A block `mc*kc`, and one `NR x kc`
# B sliver is reused by every A sliver of the block:
#
#     NR*kc*S  <= L1/2              the reused B sliver
#     mc*kc*S  <= L2core/2          the A block
#     nc*kc*S  <= L2core + L3core   the B panel
#
# rounded down to MR/NR multiples, shared levels divided by the cores sharing
# them. `nc` takes the whole per-core capacity, not half the L3: an oversized B
# panel only re-streams B, an undersized one repacks A once per `jc` block.
# `nothing` when L1d or L2 is undetected.
function _modelled_blocking(profile::TargetProfile, ::Type{T}, MR::Int, NR::Int) where {T}
    l1, l2, l3 = profile.l1d, profile.l2, profile.l3
    (l1.bytes > 0 && l2.bytes > 0) || return nothing
    smt = max(1, l1.sharing)
    core_share(c) = c.bytes ÷ max(1, c.sharing ÷ smt)
    l2core = core_share(l2)
    l3core = l3.bytes > 0 ? core_share(l3) : 0
    S = sizeof(T)
    kc = max(1, (l1.bytes ÷ 2) ÷ (NR * S))
    mc = max(MR, ((l2core ÷ 2) ÷ (kc * S)) ÷ MR * MR)
    nc = max(NR, ((l2core + l3core) ÷ (kc * S)) ÷ NR * NR)
    return Blocking(mc, kc, nc)
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
