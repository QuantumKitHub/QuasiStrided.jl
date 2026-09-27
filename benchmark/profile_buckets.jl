# Bucketed cost-attribution helper for benchmark/profile_to_suite.jl.
# `include`d, not a module, consistent with benchmark/harness.jl.
#
# Parses `Profile.fetch()`'s raw backtrace data directly (rather than relying
# on `Profile.print`'s text output) so we can compute, per frame, which named
# cost bucket it belongs to and tally sample counts ourselves.

using Profile
using Printf

# Bucket definitions per backend "family". Each entry is
# `(bucketname, [substring, ...])`; a frame matches a bucket if ANY of its
# substrings appears in either the frame's function name or file path
# (case-sensitive, as the source identifiers themselves are).
#
# Two tiers, checked in two separate passes over a whole sample's stack (see
# `_classify_backtrace`), NOT a single first-match-wins pass over one bucket
# list: `SPECIFIC_QS_BUCKETS` matches on function names (unambiguous: a
# frame named `pack_a!` is packing, wherever it lives), `FALLBACK_QS_BUCKETS`
# matches on file-path substrings (the `src/` stage folders: `"microkernels/"`,
# `"planning/"`, ...),
# used only if NO frame anywhere in the sample's stack matched a specific
# name. This two-pass split matters: a stack's leaf is very often a generic
# or third-party frame (an inlined `SIMD.jl` intrinsic, a `macro expansion`
# thunk, `Base.range`'s `iterate`) that itself matches nothing specific, and
# that leaf's *file* can be misleading -- e.g. a `macro expansion` frame
# inside `_store_tile_vector!`'s generated body lives in the same file
# (`src/microkernels/simd.jl`) as the FMA microkernel, so checking file
# substrings before walking further up the stack to the actual
# `_store_tile_vector!` frame would misclassify the whole sample as
# "microkernel" instead of "store". Running the specific-name pass across
# the ENTIRE stack first, before ever consulting a file-level fallback,
# avoids that: the walk keeps going past unmatched generic/third-party
# frames until it either finds a named function anywhere in the sample, or
# exhausts the stack and falls back to file-level matching.
const SPECIFIC_QS_BUCKETS = [
    ("gc/alloc", ["jl_gc", "gc_pool_alloc", "tensoralloc"]),
    ("adapter/prepare", ["_qs_prepare", "_qs_eligible", "_qs_throw", "argcheck", "dimcheck", "mightalias"]),
    (
        "planning",
        [
            "plan_contract", "_planned", "_plan_contract", "_plan_with_kernel",
            "_plan_resolved", "_menu_val", "_classify_labels", "_order_free_labels",
            "_prefer_swap", "_default_kernel", "_default_shape", "_resolved_defaults",
            "_kernel_from_shape", "default_blocking",
            # These would land in "planning" through the ancestor walk anyway
            # (`plan_contract` is their only caller); naming them makes the
            # attribution explicit and cannot change any classification.
            "_build_pair_group", "_candidate_mrs",
        ],
    ),
    ("packing", ["pack_a!", "pack_b!", "_pack_panel!", "_pack_a_contiguous!", "_pack_sliver!", "_pack_emit", "_check_pack_a", "_check_pack_b"]),
    # "store_tile" (no trailing "!") deliberately catches all three store
    # entry points -- `store_tile!`, `_store_tile_scattered!`, and
    # `_store_tile_vector!` -- the last of which does NOT contain the
    # substring "store_tile!" (the "!" lands after "vector", not "tile").
    ("store", ["store_tile", "scale_tile!", "_axpby_tile!", "_axpby_at!", "_store_prologue!"]),
    ("microkernel", ["accumulate", "_accumulate_step", "_execute_micro_tile!", "execute_tile!", "zero_accumulator"]),
    # The BLIS macro-blocking five/six-loop nest and its per-block
    # bookkeeping -- distinct from "planning" (the one-time, per-call
    # `plan_contract` construction): this is the *executed* loop nest.
    (
        "driver_loop",
        [
            "_execute_nest!", "execute!", "_axis_of", "_classify_slivers!",
            "_sliver_panel", "_scale_micro_tile!", "_scale_all_of_C!",
            "fill_offsets!", "describe_block", "block_descriptors!", "checked_tile_storage_bounds",
            # The closed-form affine block path and the hoisted once-per-macro-block bounds check.
            # Same note as under "planning" -- `_execute_nest!` is the only
            # caller of each, so these names change no classification; the
            # `unsafe_pack_*!`/`unsafe_execute_*!` entry points need no entry
            # at all, since they already contain the `pack_a!`/`execute_tile!`
            # substrings their buckets match on.
            "affine_ramp", "_ramp_slivers!", "_ramp_descriptor", "_ramp_offset_range",
            "descriptor_offset_range", "checked_span_bounds",
        ],
    ),
]
const FALLBACK_QS_BUCKETS = [
    ("gc/alloc", ["gc"]),
    ("adapter/prepare", ["integrations/", "StridedView"]),
    ("packing", ["packing/"]),
    ("microkernel", ["microkernels/"]),
    ("driver_loop", ["execution/", "layout/"]),
    ("planning", ["planning/"]),
]

const SPECIFIC_BLAS_BUCKETS = [
    ("gc/alloc", ["jl_gc", "gc_pool_alloc", "tensoralloc"]),
    ("blas", ["dgemm", "sgemm", "gemm", "openblas", "cblas"]),
    ("permute/copy", ["_mapreduce", "permutedims"]),
]
const FALLBACK_BLAS_BUCKETS = [
    ("gc/alloc", ["gc"]),
    ("permute/copy", ["Strided", "copy"]),
]

const BUCKET_ORDER_QS = ["adapter/prepare", "planning", "driver_loop", "packing", "microkernel", "store", "gc/alloc", "TO overhead", "other"]
const BUCKET_ORDER_BLAS = ["blas", "permute/copy", "gc/alloc", "TO overhead", "other"]

# Cache StackTraces lookups: the same instruction pointer recurs across many
# backtraces/samples, and `StackTraces.lookup` is not free.
const _LOOKUP_CACHE = Dict{UInt, Vector{Base.StackTraces.StackFrame}}()

function _lookup_cached(ip)
    key = UInt(ip)
    return get!(_LOOKUP_CACHE, key) do
        Base.StackTraces.lookup(convert(Ptr{Cvoid}, ip))
    end
end

_frame_haystack(frame) = string(frame.func) * "|" * string(frame.file)

# `occursin("promote", ...)` alone would match ordinary `Base.promote`/
# `promote_type` frames that can appear inlined deep inside the kernel;
# restrict to TensorOperations' own promotion entry point.
function _is_tensoroperations_path(frame)
    f = string(frame.file)
    fn = string(frame.func)
    return occursin("TensorOperations", f) || occursin("tensorcontract", fn) ||
        occursin("ncon", fn) || occursin("promote_contract", fn)
end

function _match_bucket(hay::AbstractString, buckets)
    for (name, substrs) in buckets
        for s in substrs
            occursin(s, hay) && return name
        end
    end
    return nothing
end

# Two-pass, single-frame classification: specific (function-name) match
# first, generic (file-path) fallback second. Kept separate from the
# per-sample walk in `_classify_backtrace` (which must run the *specific*
# pass across the WHOLE stack before considering any *fallback* match) --
# this function alone would still be wrong to call leaf-only for the same
# reason leaf-only self-time was wrong (see module docstring above).
function _classify_frame_or_nothing(frame, backendname::AbstractString; fallback::Bool)
    hay = _frame_haystack(frame)
    specific, generic = backendname == "StridedBLAS" ?
        (SPECIFIC_BLAS_BUCKETS, FALLBACK_BLAS_BUCKETS) : (SPECIFIC_QS_BUCKETS, FALLBACK_QS_BUCKETS)
    name = _match_bucket(hay, specific)
    name === nothing || return name
    fallback || return nothing
    name = _match_bucket(hay, generic)
    name === nothing || return name
    # "TO overhead" fallback: anything in the TensorOperations package path
    # (or the generic entry-point function names) not already caught above.
    _is_tensoroperations_path(frame) && return "TO overhead"
    return nothing
end

# Walk one sample's IPs leaf-to-root (as `Profile.fetch` stores them),
# checking every inlined frame at each IP innermost-first. Pass 1 checks
# ONLY specific (function-name) matches across the entire stack; only if
# that whole-stack pass finds nothing does pass 2 re-walk the same stack
# allowing file-level fallback matches. This ordering is what makes
# `_store_tile_vector!`'s generated `macro expansion` leaf classify
# correctly: that leaf frame itself matches nothing specific, but walking
# further up the SAME sample's stack (still within pass 1) reaches the
# `_store_tile_vector!` frame itself, which does. A single-pass "first
# frame with any match, specific-or-fallback" walk would instead stop at
# the leaf's own file-level "kernels/" match and call it "microkernel".
function _classify_backtrace(ips, backendname::AbstractString)
    for fallback in (false, true)
        for ip in ips
            for frame in _lookup_cached(ip)
                name = _classify_frame_or_nothing(frame, backendname; fallback)
                name === nothing || return name
            end
        end
    end
    return "other"
end

"""
    compute_buckets(data, backendname) -> Dict{String,Int}

`data` must be the raw backtrace stream from `Profile.fetch(include_meta =
false)` -- metadata (threadid/taskid/etc, included by `fetch`'s default
`include_meta = true`) is NOT instruction-pointer data and must be stripped
first, or `Base.StackTraces.lookup` will resolve garbage frames from it.

Backtraces are stored leaf-first (the actively-executing frame comes first)
and each per-sample backtrace is terminated by a `0` sentinel. This function
splits on those sentinels and classifies each sample via `_classify_backtrace`
-- "nearest named ancestor, specific names preferred over file-level
fallbacks", not strict leaf-only self-time, and not inclusive/"Count"-column
semantics either: each sample still contributes to exactly one bucket, so
generic runtime/dispatch frames sitting between named layers cannot
spuriously dominate every bucket the way naive inclusive counting would.

Returns raw sample counts per bucket name (NOT yet normalized to
percentages); callers should divide by the sum of all bucket counts (not by
`length(data)`, which also counts sentinels) to get percentages that sum to
100%.
"""
function compute_buckets(data, backendname::AbstractString)
    order = backendname == "StridedBLAS" ? BUCKET_ORDER_BLAS : BUCKET_ORDER_QS
    counts = Dict{String, Int}(name => 0 for name in order)
    sample_ips = UInt[]
    for ip in data
        if iszero(ip)
            # Sentinel: end of one sample's backtrace.
            if !isempty(sample_ips)
                name = _classify_backtrace(sample_ips, backendname)
                counts[name] = get(counts, name, 0) + 1
                empty!(sample_ips)
            end
            continue
        end
        push!(sample_ips, UInt(ip))
    end
    if !isempty(sample_ips)  # tolerate a stream not ending on a sentinel
        name = _classify_backtrace(sample_ips, backendname)
        counts[name] = get(counts, name, 0) + 1
    end
    return counts
end

"""
    print_bucket_table(io, caseid, backendname, buckets, total_samples, allocated_bytes, reps)

Print a percentage-of-samples table for one (case, backend) pair. Percentages
are computed against the sum of all bucket counts (not the raw `data`
length, which also includes sentinel/zero entries), so they sum to ~100%
regardless of how many samples were sentinels.
"""
function print_bucket_table(io, caseid, backendname, buckets, total_samples, allocated_bytes, reps)
    denom = sum(values(buckets))
    println(io, "="^72)
    @printf(
        io, "case=%s backend=%s reps=%d total_raw_samples=%d classified_samples=%d\n",
        caseid, backendname, reps, total_samples, denom
    )
    @printf(io, "allocated (representative single call): %d bytes\n", allocated_bytes)
    println(io, "-"^72)
    order = backendname == "StridedBLAS" ? BUCKET_ORDER_BLAS : BUCKET_ORDER_QS
    pct_sum = 0.0
    for name in order
        c = get(buckets, name, 0)
        pct = denom == 0 ? 0.0 : 100.0 * c / denom
        pct_sum += pct
        @printf(io, "  %-18s %8d samples  %6.2f%%\n", name, c, pct)
    end
    @printf(io, "  %-18s %8s          %6.2f%% (sanity check, should be ~100.00%%)\n", "TOTAL", "", pct_sum)
    println(io)
    return nothing
end
