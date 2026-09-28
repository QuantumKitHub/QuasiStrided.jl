# Runtime hardware detection. It runs once per process in `__init__`, never at
# precompile time, and every failure path resolves to `:unknown`, which selects
# the fixed fallback shapes.

"""
    CacheLevel(bytes, ways, line, sharing)

One detected cache level; any field may be `0` for "not detected". `sharing` is
the number of logical CPUs sharing this level.
"""
struct CacheLevel
    bytes::Int
    ways::Int
    line::Int
    sharing::Int
end
CacheLevel() = CacheLevel(0, 0, 0, 0)

"""
    TargetProfile

What was detected about the host CPU. `isa` is one of `:avx512`, `:avx2`,
`:neon` or `:unknown`.
"""
struct TargetProfile
    isa::Symbol
    arch::Symbol
    cpu_name::String
    vector_bytes::Int
    nregisters::Int
    l1d::CacheLevel
    l2::CacheLevel
    l3::CacheLevel
end

unknown_target() = TargetProfile(
    :unknown, Sys.ARCH, "", 0, 0, CacheLevel(), CacheLevel(), CacheLevel()
)

# LLVM CPU name -> vector ISA. Unlisted names fall through to the CPUID probe.
const _UARCH_ISA = Dict{String, Symbol}(
    n => :avx512 for n in (
            "skylake-avx512", "cascadelake", "cooperlake", "cannonlake",
            "icelake-client", "icelake-server", "tigerlake", "rocketlake",
            "sapphirerapids", "emeraldrapids", "graniterapids", "knl", "knm",
            "znver4", "znver5",
        )
)
for n in (
        "haswell", "broadwell", "skylake", "alderlake", "raptorlake",
        "meteorlake", "sierraforest", "grandridge", "tremont", "goldmont",
        "goldmont-plus", "znver1", "znver2", "znver3", "bdver4",
    )
    _UARCH_ISA[n] = :avx2
end

# `Base.BinaryPlatforms.CPUID` is undocumented, so a failure degrades to `:unknown`.
function _isa_from_cpuid()
    return try
        C = Base.BinaryPlatforms.CPUID
        C.test_cpu_feature(C.JL_X86_avx512f) ? :avx512 :
            C.test_cpu_feature(C.JL_X86_avx2) ? :avx2 : :unknown
    catch
        :unknown
    end
end

_isa_rank(k::Symbol) = k === :avx512 ? 2 : k === :avx2 ? 1 : 0

function _detect_isa()
    if Sys.ARCH === :x86_64 || Sys.ARCH === :i686
        key = get(_UARCH_ISA, Sys.CPU_NAME, :miss)
        key === :miss && return _isa_from_cpuid()
        # A hypervisor can mask CPUID features without changing the CPU name;
        # AVX-512 code would then SIGILL, so a narrower live probe wins.
        probe = _isa_from_cpuid()
        probe === :unknown && return key
        return _isa_rank(probe) < _isa_rank(key) ? probe : key
    elseif Sys.ARCH === :aarch64
        # SVE is not detected: its runtime vector length cannot be a fixed kernel shape.
        return :neon
    end
    return :unknown
end

# The register count is the ISA's, not the lane width's: AVX512VL gives 32 ymm registers.
_isa_vector_bytes(::Val{K}) where {K} = K === :avx512 ? 64 : K === :avx2 ? 32 : K === :neon ? 16 : 0
_isa_nregisters(::Val{K}) where {K} = K === :avx512 ? 32 : K === :avx2 ? 16 : K === :neon ? 32 : 0

# "32K" / "1024K" / "2M" as Linux sysfs writes them.
function _parse_size(s::AbstractString)
    s = strip(s)
    isempty(s) && return 0
    mult = get(Dict('K' => 1024, 'M' => 1024^2, 'G' => 1024^3), uppercase(s[end]), 1)
    mult == 1 || (s = s[1:(end - 1)])
    return something(tryparse(Int, strip(s)), 0) * mult
end

# "0,16" -> 2; "0-7,16-23" -> 16; "" -> 0.
function _count_cpu_list(s::AbstractString)
    n = 0
    for part in split(strip(s), ',')
        isempty(part) && continue
        lohi = split(part, '-')
        lo = tryparse(Int, lohi[1])
        lo === nothing && continue
        hi = length(lohi) == 2 ? tryparse(Int, lohi[2]) : lo
        hi === nothing || (n += max(0, hi - lo + 1))
    end
    return n
end

_read_or(path, default = "") = try
    isfile(path) ? chomp(read(path, String)) : default
catch
    default
end
_int_or(path) = something(tryparse(Int, _read_or(path)), 0)

function _cache_topology_linux()
    base = "/sys/devices/system/cpu/cpu0/cache"
    isdir(base) || return nothing
    levels = Dict{Symbol, CacheLevel}()
    for entry in readdir(base)
        startswith(entry, "index") || continue
        d = joinpath(base, entry)
        level = tryparse(Int, _read_or(joinpath(d, "level")))
        level === nothing && continue
        kind = _read_or(joinpath(d, "type"))
        key = level == 1 ? (kind == "Data" ? :l1d : :skip) :
            level == 2 ? :l2 : level == 3 ? :l3 : :skip
        key === :skip && continue
        levels[key] = CacheLevel(
            _parse_size(_read_or(joinpath(d, "size"))),
            _int_or(joinpath(d, "ways_of_associativity")),
            _int_or(joinpath(d, "coherency_line_size")),
            _count_cpu_list(_read_or(joinpath(d, "shared_cpu_list"))),
        )
    end
    return (
        l1d = get(levels, :l1d, CacheLevel()), l2 = get(levels, :l2, CacheLevel()),
        l3 = get(levels, :l3, CacheLevel()),
    )
end

_sysctl_int(name) = try
    something(tryparse(Int, chomp(read(`sysctl -n $name`, String))), 0)
catch
    0
end

# macOS exposes no associativity; `cpusperl2` gives the L2 sharing.
function _cache_topology_darwin()
    line = _sysctl_int("hw.cachelinesize")
    pick(a, b) = (v = _sysctl_int(a); v == 0 ? _sysctl_int(b) : v)
    return (
        l1d = CacheLevel(pick("hw.perflevel0.l1dcachesize", "hw.l1dcachesize"), 0, line, 1),
        l2 = CacheLevel(
            pick("hw.perflevel0.l2cachesize", "hw.l2cachesize"), 0, line,
            _sysctl_int("hw.perflevel0.cpusperl2")
        ),
        l3 = CacheLevel(
            _sysctl_int("hw.l3cachesize"), 0, line,
            _sysctl_int("hw.perflevel0.logicalcpu")
        ),
    )
end

"""
    cache_topology() -> NamedTuple or nothing

Detected cache hierarchy as `(; l1d, l2, l3)` of [`CacheLevel`](@ref), read from
Linux sysfs or macOS `sysctl`, or `nothing` if unavailable.
"""
function cache_topology()
    return try
        Sys.islinux() ? _cache_topology_linux() :
            Sys.isapple() ? _cache_topology_darwin() : nothing
    catch
        nothing
    end
end

function _detect_target()
    key = _detect_isa()
    topo = cache_topology()
    l1d, l2, l3 = topo === nothing ?
        (CacheLevel(), CacheLevel(), CacheLevel()) : (topo.l1d, topo.l2, topo.l3)
    return TargetProfile(
        key, Sys.ARCH, Sys.CPU_NAME,
        _isa_vector_bytes(Val(key)), _isa_nregisters(Val(key)), l1d, l2, l3
    )
end

const _TARGET = Ref{TargetProfile}(unknown_target())

"""
    target_profile() -> TargetProfile

The [`TargetProfile`](@ref) detected for this process (all `:unknown` if
detection failed).
"""
target_profile() = _TARGET[]

function _init_target!()
    _TARGET[] = try
        _detect_target()
    catch
        unknown_target()
    end
    return nothing
end

# Shared by the deinterleaving complex packer and the planar store: both pay
# off only with 512-bit vector registers.
@inline _complex_fastpath_isa_eligible(profile::TargetProfile) =
    profile.vector_bytes == _isa_vector_bytes(Val(:avx512))
@inline _complex_fastpath_isa_eligible() = _complex_fastpath_isa_eligible(target_profile())
