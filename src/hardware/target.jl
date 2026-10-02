# Runtime hardware detection. It runs once per process in `__init__`, never at
# precompile time, and every failure path resolves to `:unknown`, which selects
# the fixed fallback shapes.

"""
    CacheLevel(bytes, line, sharing)

One detected cache level; any field may be `0` for "not detected". `sharing` is
the number of logical CPUs sharing this level.
"""
struct CacheLevel
    bytes::Int
    line::Int
    sharing::Int
end
CacheLevel() = CacheLevel(0, 0, 0)

"""
    TargetProfile(isa, cpu_name, l1d, l2, l3)

What was detected about the host CPU. `isa` is one of `:avx512`, `:avx2`,
`:neon` or `:unknown`, and fixes `vector_bytes` and `nregisters`. The core's
share of L2 and L3 (`l2_share`, `l3_share`, in bytes) and whether 512-bit FMAs
are double-pumped (`double_pumped`) are derived as well.
"""
struct TargetProfile
    isa::Symbol
    cpu_name::String
    vector_bytes::Int
    nregisters::Int
    l1d::CacheLevel
    l2::CacheLevel
    l3::CacheLevel
    # Derived here: planning reads them on every call.
    l2_share::Int
    l3_share::Int
    double_pumped::Bool

    function TargetProfile(isa::Symbol, cpu_name::AbstractString, l1d::CacheLevel, l2::CacheLevel, l3::CacheLevel)
        isa in (:avx512, :avx2, :neon, :unknown) ||
            throw(ArgumentError("TargetProfile: unknown ISA $(repr(isa))"))
        return new(
            isa, cpu_name, isa_vector_bytes(isa), isa_nregisters(isa), l1d, l2, l3,
            core_bytes(l1d, l2), core_bytes(l1d, l3), cpu_name in _DOUBLE_PUMPED_CPUS
        )
    end
end

# This core's share of `level`: the CPUs sharing it over the SMT threads per
# core, which are the CPUs sharing L1d.
core_bytes(l1d::CacheLevel, level::CacheLevel) = level.bytes ÷ max(1, level.sharing ÷ max(1, l1d.sharing))

# AMD's AVX-512 cores, which double-pump 512-bit FMAs.
const _DOUBLE_PUMPED_CPUS = ("znver4", "znver5")

unknown_target() = TargetProfile(:unknown, "", CacheLevel(), CacheLevel(), CacheLevel())

# The register count is the ISA's, not the lane width's: AVX512VL gives 32 ymm registers.
isa_vector_bytes(isa::Symbol) = isa === :avx512 ? 64 : isa === :avx2 ? 32 : isa === :neon ? 16 : 0
isa_nregisters(isa::Symbol) = isa === :avx512 ? 32 : isa === :avx2 ? 16 : isa === :neon ? 32 : 0

# `Base.BinaryPlatforms.CPUID` is undocumented, so a failure degrades to `:unknown`.
function isa_from_cpuid()
    return try
        C = Base.BinaryPlatforms.CPUID
        C.test_cpu_feature(C.JL_X86_avx512f) ? :avx512 :
            C.test_cpu_feature(C.JL_X86_avx2) ? :avx2 : :unknown
    catch
        :unknown
    end
end

function detect_isa()
    (Sys.ARCH === :x86_64 || Sys.ARCH === :i686) && return isa_from_cpuid()
    # SVE is not detected: its runtime vector length cannot be a fixed kernel shape.
    Sys.ARCH === :aarch64 && return :neon
    return :unknown
end

# "32K" / "1024K" / "2M" as Linux sysfs writes them.
function parse_size(s::AbstractString)
    s = strip(s)
    isempty(s) && return 0
    unit = uppercase(s[end])
    mult = unit == 'K' ? 1024 : unit == 'M' ? 1024^2 : unit == 'G' ? 1024^3 : 1
    mult == 1 || (s = s[1:(end - 1)])
    return something(tryparse(Int, strip(s)), 0) * mult
end

# "0,16" -> 2; "0-7,16-23" -> 16; "" -> 0.
function count_cpu_list(s::AbstractString)
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

read_or(path, default = "") = try
    isfile(path) ? chomp(read(path, String)) : default
catch
    default
end
int_or(path) = something(tryparse(Int, read_or(path)), 0)

function cache_topology_linux()
    base = "/sys/devices/system/cpu/cpu0/cache"
    isdir(base) || return nothing
    levels = Dict{Symbol, CacheLevel}()
    for entry in readdir(base)
        startswith(entry, "index") || continue
        d = joinpath(base, entry)
        level = tryparse(Int, read_or(joinpath(d, "level")))
        level === nothing && continue
        kind = read_or(joinpath(d, "type"))
        key = level == 1 ? (kind == "Data" ? :l1d : :skip) :
            level == 2 ? :l2 : level == 3 ? :l3 : :skip
        key === :skip && continue
        levels[key] = CacheLevel(
            parse_size(read_or(joinpath(d, "size"))),
            int_or(joinpath(d, "coherency_line_size")),
            count_cpu_list(read_or(joinpath(d, "shared_cpu_list"))),
        )
    end
    return (
        l1d = get(levels, :l1d, CacheLevel()), l2 = get(levels, :l2, CacheLevel()),
        l3 = get(levels, :l3, CacheLevel()),
    )
end

sysctl_int(name) = try
    something(tryparse(Int, chomp(read(`sysctl -n $name`, String))), 0)
catch
    0
end

# macOS exposes no associativity; `cpusperl2` gives the L2 sharing.
function cache_topology_darwin()
    line = sysctl_int("hw.cachelinesize")
    pick(a, b) = (v = sysctl_int(a); v == 0 ? sysctl_int(b) : v)
    return (
        l1d = CacheLevel(pick("hw.perflevel0.l1dcachesize", "hw.l1dcachesize"), line, 1),
        l2 = CacheLevel(
            pick("hw.perflevel0.l2cachesize", "hw.l2cachesize"), line,
            sysctl_int("hw.perflevel0.cpusperl2")
        ),
        l3 = CacheLevel(
            sysctl_int("hw.l3cachesize"), line,
            sysctl_int("hw.perflevel0.logicalcpu")
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
        Sys.islinux() ? cache_topology_linux() :
            Sys.isapple() ? cache_topology_darwin() : nothing
    catch
        nothing
    end
end

function detect_target()
    topo = cache_topology()
    l1d, l2, l3 = topo === nothing ?
        (CacheLevel(), CacheLevel(), CacheLevel()) : (topo.l1d, topo.l2, topo.l3)
    return TargetProfile(detect_isa(), Sys.CPU_NAME, l1d, l2, l3)
end

const TARGET = Ref{TargetProfile}(unknown_target())

"""
    target_profile() -> TargetProfile

The [`TargetProfile`](@ref) detected for this process (all `:unknown` if
detection failed).
"""
target_profile() = TARGET[]

init_target!() = (TARGET[] = detect_target(); nothing)

# The core's private L2 share, or 1 MB when undetected.
l2_core_bytes(profile::TargetProfile) = profile.l2_share > 0 ? profile.l2_share : 1 << 20

line_bytes(profile::TargetProfile) = profile.l1d.line > 0 ? profile.l1d.line : 64

# A K step at least a page apart: a chain of demand misses no prefetcher
# follows, which the K-order model charges this factor.
const _K_WALK_FAR_BYTES = 4096
const _K_WALK_FAR_PENALTY = 3
