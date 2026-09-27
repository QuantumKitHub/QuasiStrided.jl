# Pack-A walk in isolation: gather every A element of a contraction into
# `mr`-row sliver panels, mc rows x kc K steps at a time, exactly as
# `_execute_nest!` visits them, with the leaf loop in two orders:
#
#   sliver  for r in slivers, for p in 0:kc-1, for t in 0:mr-1   (shipped:
#           one sliver's whole panel, then the next sliver)
#   kouter  for p in 0:kc-1, for r in slivers, for t in 0:mr-1   (each K step
#           across all slivers of the block before the next K step)
#
# When the sliver rows' and the K steps' byte strides in A are multiples of
# 4 KB (intensli_7 at dims 16/24/32: A strides dim^3, dim^4 elements), every
# load of one sliver's panel lands in the same L1 set; `kouter` interleaves
# slivers from other C-order coordinates (other sets). Plain scalar loads
# through precomputed offsets -- no QuasiStrided packer -- so only the walk
# differs.
#
#   julia -t 1 --project=benchmark benchmark/probes/probe_pack_order.jl \
#       --case intensli_7 --dim 24 [--shape 8,6,4] [--pad 1] [--reps 7]

using QuasiStrided
using QuasiStrided: plan_contract, mr, axis_length, _kernel_from_shape
using StridedViews
using Statistics: median
using Printf

include(joinpath(@__DIR__, "..", "harness.jl"))

const KNOWN = Dict("intensli_6" => "abcde bf>dcfea", "intensli_7" => "abcde df>ecbfa", "intensli_8" => "abcde fb>dfcea")
const CASE = argopt("case", "intensli_7")
const EXPR = something(argval("expr"), get(KNOWN, CASE, nothing))
const DIM = argopt("dim", 16)
const PAD = argopt("pad", 0)
const REPS = argopt("reps", 7)
const SHAPE = Tuple(parse.(Int, split(argopt("shape", "8,6,4"), ',')))::NTuple{3, Int}
const T = Float64

lhs, rhs = split(EXPR, '>')
la, lb = split(strip(lhs), ' ')
letters = unique(vcat(collect(la), collect(lb), collect(rhs)))
lab(s) = Tuple(findfirst(==(c), letters) for c in s)
indA, indB, indC = lab(la), lab(lb), lab(rhs)

# `--pad P` pads EVERY axis of A's parent to DIM + P, so no stride is a
# multiple of 4 KB (padding only the first axis leaves dim^3 * (dim+P) * 8
# a 4 KB multiple at dim 32).
Apar = randn(T, ntuple(_ -> DIM + PAD, length(la))...)
A = view(Apar, ntuple(_ -> 1:DIM, length(la))...)
B = randn(T, ntuple(_ -> DIM, length(lb))...)
C = zeros(T, ntuple(_ -> DIM, length(rhs))...)
plan = plan_contract(
    StridedView(C), StridedView(A), indA, StridedView(B), indB, indC;
    oracle = false, kernel = _kernel_from_shape(SHAPE, T)
)
pointer(plan.Astorage) == pointer(Apar) || error("swapped plan: A does not feed M")
const Ast = plan.Astorage  # the flat storage the engine indexes (0-based offsets)

# Offsets of every M row and K step into A's parent (0-based), in the plan's
# enumeration order (first axis fastest).
function offsets(lengths, strides)
    out = Int[0]
    for (L, s) in zip(lengths, strides)
        out = vec([o + i * s for o in out, i in 0:(L - 1)])
    end
    return out
end
mo = offsets(plan.mgroup.lengths, plan.mgroup.strides[1])
ko = offsets(plan.kgroup.lengths, plan.kgroup.strides[1])
base = plan.Abase
MR = mr(plan.kernel)
mc, kc = plan.blocking.mc, plan.blocking.kc
println("case = $CASE  dim = $DIM  pad = $PAD  MR = $MR  mc = $mc  kc = $kc")
println("A strides (bytes) = ", strides(A) .* sizeof(T), "  M strides(A) = ", plan.mgroup.strides[1], "  K strides(A) = ", plan.kgroup.strides[1])

function pack_all!(buf, Apar, mo, ko, base, MR, mc, kc, ::Val{ORDER}) where {ORDER}
    Qm, Qk = length(mo), length(ko)
    acc = 0.0
    @inbounds for pc in 0:kc:(Qk - 1)
        kb = min(kc, Qk - pc)
        for ic in 0:mc:(Qm - 1)
            mb = min(mc, Qm - ic)
            ns = cld(mb, MR)
            if ORDER === :sliver
                for r in 0:(ns - 1), p in 0:(kb - 1), t in 0:(MR - 1)
                    i = r * MR + t
                    i < mb || continue
                    buf[r * MR * kb + p * MR + t + 1] = Apar[base + mo[ic + i + 1] + ko[pc + p + 1]]
                end
            else
                for p in 0:(kb - 1), r in 0:(ns - 1), t in 0:(MR - 1)
                    i = r * MR + t
                    i < mb || continue
                    buf[r * MR * kb + p * MR + t + 1] = Apar[base + mo[ic + i + 1] + ko[pc + p + 1]]
                end
            end
            acc += buf[1]  # keep the block's pack live
        end
    end
    return acc
end

buf = zeros(T, (mc + MR) * kc)
n = length(mo) * length(ko)
res = Dict{Symbol, Vector{Float64}}(:sliver => Float64[], :kouter => Float64[])
for o in (:sliver, :kouter)
    pack_all!(buf, Ast, mo, ko, base + 1, MR, mc, kc, Val(o))  # warm-up/compile
end
# `base` (plan.Abase) is 0-based into the parent; Julia indexing is 1-based.
for rep in 1:REPS, o in (:sliver, :kouter)  # interleaved
    t0 = time_ns()
    pack_all!(buf, Ast, mo, ko, base + 1, MR, mc, kc, Val(o))
    push!(res[o], (time_ns() - t0) / n)
end
for o in (:sliver, :kouter)
    v = res[o]
    @printf("  %-7s  %6.2f ns/elem  (min %.2f, max %.2f)\n", o, median(v), minimum(v), maximum(v))
end
