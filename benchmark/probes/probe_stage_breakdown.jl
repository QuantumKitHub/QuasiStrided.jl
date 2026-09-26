# Stage breakdown of one contraction under `execute!`: an instrumented copy of
# `_execute_nest!` (src/execution/execute.jl) with `time_ns` brackets around
# the B pack, the A pack and the micro-kernel+store loop, so that a slow case
# can be attributed to the stage that eats the time. Also prints the plan's
# resolved axis groups (each label's stride in both operands) so the memory
# walk each pack performs can be read off directly.
#
#   julia -t 1 --project=benchmark benchmark/probes/probe_stage_breakdown.jl \
#       --case scrambled_1_3_1 --dim 48 --dtype Float64 --reps 5
#
# Cases are einsum-style letter strings `A B>C` (TensorOperationsBenchmarks'
# `expr` column of bench_to_suite.csv); `--expr "abcd dcbe>ae"` accepts any.
# Every axis gets extent `--dim`. Times are medians over `--reps` calls after
# one warm-up; the stage times come from the instrumented nest, the "execute!"
# row from the shipped driver (they should agree to within the timer overhead).

using QuasiStrided
using QuasiStrided: plan_contract, execute!, mr, nr, axis_length, affine_ramp,
    fill_offsets!, describe_block, descriptor_offset_range, checked_span_bounds,
    unsafe_pack_a!, unsafe_pack_b!, packed_a_per_k, packed_b_per_k,
    _ramp_slivers!, _ramp_descriptor, _ramp_offset_range, _classify_slivers!,
    _axis_of, _sliver_panel, _pack_sliver!, unsafe_execute_micro_tile!
using StridedViews
using Random
using Statistics: median
using Printf
using LinearAlgebra

include(joinpath(@__DIR__, "..", "harness.jl"))

const KNOWN = Dict(
    "scrambled_1_3_1" => "abcd dcbe>ae",
    "gemm_ready_1_3_1" => "abcd bcde>ae",
    "b_permuted_1_3_1" => "abcd becd>ae",
    "a_permuted_2_2_2" => "abcd bdef>acef",
    "b_permuted_2_2_2" => "abcd cedf>abef",
    "gemm_ready_2_2_2" => "abcd cdef>abef",
    "intensli_6" => "abcde bf>dcfea",
    "intensli_7" => "abcde df>ecbfa",
    "intensli_8" => "abcde fb>dfcea",
    "ao2mo_1" => "ab acde>bcde",
    "ccsd_t_3" => "abcd cefg>fgdabe",
)

const CASE = argopt("case", "scrambled_1_3_1")
const EXPR = something(argval("expr"), get(KNOWN, CASE, nothing))
EXPR === nothing && error("unknown --case $CASE and no --expr given")
const DIM = argopt("dim", 48)
const T = parse_dtypes(argopt("dtype", "Float64"))[1]
const REPS = argopt("reps", 5)

function parse_expr(expr::AbstractString)
    lhs, rhs = split(expr, '>')
    a, b = split(strip(lhs), ' ')
    return collect(a), collect(b), collect(strip(rhs))
end

const LA, LB, LC = parse_expr(EXPR)
const LETTERS = unique(vcat(LA, LB, LC))
label(c) = findfirst(==(c), LETTERS)
const indA = Tuple(label.(LA))
const indB = Tuple(label.(LB))
const indC = Tuple(label.(LC))

rng = MersenneTwister(1234)
A = randn(rng, T, ntuple(_ -> DIM, length(LA))...)
B = randn(rng, T, ntuple(_ -> DIM, length(LB))...)
C = zeros(T, ntuple(_ -> DIM, length(LC))...)
Av, Bv, Cv = StridedView(A), StridedView(B), StridedView(C)

plan = plan_contract(Cv, Av, indA, Bv, indB, indC; oracle = false)

# Optional hand-made M split (see probe_msplit_prototype.jl): `--msplit-pos P
# --msplit-li L` splits A's fastest M axis into (L, rest) and moves the inner
# part to position P of the M enumeration.
const MSPLIT_POS = argopt("msplit-pos", 0)
const MSPLIT_LI = argopt("msplit-li", 8)
if MSPLIT_POS > 0
    using QuasiStrided: AxisGroup, ContractPlan
    g = plan.mgroup
    D = length(g.lengths)
    du = argmin([g.lengths[d] > 1 ? abs(g.strides[1][d]) : typemax(Int) for d in 1:D])
    dims = [(g.lengths[d], ntuple(p -> g.strides[p][d], 2)) for d in 1:D]
    inner = (MSPLIT_LI, dims[du][2])
    outer = (g.lengths[du] ÷ MSPLIT_LI, MSPLIT_LI .* dims[du][2])
    dims[du] = outer
    insert!(dims, MSPLIT_POS, inner)
    mg = AxisGroup(ntuple(i -> dims[i][1], D + 1), ntuple(p -> ntuple(i -> dims[i][2][p], D + 1), 2))
    global plan = ContractPlan(
        plan.kernel, mg, plan.ngroup, plan.kgroup, plan.blocking,
        plan.Astorage, plan.Abase, plan.Bstorage, plan.Bbase, plan.Cstorage, plan.Cbase,
        plan.atransform, plan.btransform, plan.workspace
    )
end

# ---------------------------------------------------------------------------
# Plan report
# ---------------------------------------------------------------------------
println("case = $CASE  expr = $EXPR  dim = $DIM  T = $T")
println("A labels ", join(LA), " strides ", strides(Av))
println("B labels ", join(LB), " strides ", strides(Bv))
println("C labels ", join(LC), " strides ", strides(Cv))
println("kernel = ", typeof(plan.kernel).name.name, " MR=", mr(plan.kernel), " NR=", nr(plan.kernel))
println("blocking = ", plan.blocking)
println("swapped = ", plan.Astorage !== parent(Av))
for (name, g, ops) in (("M", plan.mgroup, "(A,C)"), ("N", plan.ngroup, "(B,C)"), ("K", plan.kgroup, "(A,B)"))
    println("$name group $ops: lengths=", g.lengths, " strides=", g.strides, "  Q=", axis_length(g), "  ramp=", affine_ramp(g)[1])
end
flops = 2.0 * axis_length(plan.mgroup) * axis_length(plan.ngroup) * axis_length(plan.kgroup)
println("flops = ", flops, "  bytes(A+B+C) = ", sizeof(A) + sizeof(B) + sizeof(C))

# ---------------------------------------------------------------------------
# Instrumented nest (copy of `_execute_nest!`, timers added, nothing else)
# ---------------------------------------------------------------------------
mutable struct StageTimes
    packb::Int
    packa::Int
    kern::Int
    total::Int
end

function timed_nest!(plan, alphaT::T, betaT::T, st::StageTimes) where {T}
    ws = plan.workspace
    kernel = plan.kernel
    MRk = mr(kernel)
    NRk = nr(kernel)
    Qm = axis_length(plan.mgroup)
    Qn = axis_length(plan.ngroup)
    Qk = axis_length(plan.kgroup)
    mc_eff = plan.blocking.mc
    kc_eff = plan.blocking.kc
    nc_eff = plan.blocking.nc
    MRp = packed_a_per_k(kernel)
    NRp = packed_b_per_k(kernel)
    atransform = plan.atransform
    btransform = plan.btransform
    (m_ramp, m_step) = affine_ramp(plan.mgroup)
    (n_ramp, n_step) = affine_ramp(plan.ngroup)
    (k_ramp, k_step) = affine_ramp(plan.kgroup)
    lenA = length(plan.Astorage)
    lenB = length(plan.Bstorage)
    lenC = length(plan.Cstorage)
    t_start = time_ns()
    GC.@preserve ws begin
        jc = 0
        while jc < Qn
            nblock = min(nc_eff, Qn - jc)
            n_slivers = cld(nblock, NRk)
            (rng_nB, rng_nC) = if n_ramp
                _ramp_slivers!(ws.n_desc_B, ws.n_desc_C, n_step[1], n_step[2], jc, nblock, NRk, n_slivers)
            else
                fill_offsets!((ws.n_buf_B, ws.n_buf_C), plan.ngroup, jc, nblock)
                _classify_slivers!(ws.n_desc_B, ws.n_desc_C, ws.n_buf_B, ws.n_buf_C, nblock, NRk, n_slivers)
            end
            pc = 0
            firstpanel = true
            while pc < Qk
                kblock = min(kc_eff, Qk - pc)
                dK_A, dK_B, rng_kA, rng_kB = if k_ramp
                    (
                        _ramp_descriptor(k_step[1], pc, kblock), _ramp_descriptor(k_step[2], pc, kblock),
                        _ramp_offset_range(k_step[1], pc, kblock), _ramp_offset_range(k_step[2], pc, kblock),
                    )
                else
                    fill_offsets!((ws.k_buf_A, ws.k_buf_B), plan.kgroup, pc, kblock)
                    dA = describe_block(ws.k_buf_A, 0, kblock)
                    dB = describe_block(ws.k_buf_B, 0, kblock)
                    (dA, dB, descriptor_offset_range(dA, ws.k_buf_A, 0), descriptor_offset_range(dB, ws.k_buf_B, 0))
                end
                colsA_k = _axis_of(dK_A, ws.k_buf_A, 0)
                rowsB_k = _axis_of(dK_B, ws.k_buf_B, 0)
                checked_span_bounds(plan.Bbase, rng_kB, rng_nB, lenB)
                beta_eff = firstpanel ? betaT : one(T)

                t0 = time_ns()
                for s in 0:(n_slivers - 1)
                    sfirst = s * NRk
                    colsB = _axis_of(ws.n_desc_B[s + 1], ws.n_buf_B, sfirst)
                    bpanel = _sliver_panel(ws.packed_b, NRp, kblock, s)
                    _pack_sliver!(unsafe_pack_b!, bpanel, plan.Bstorage, plan.Bbase, rowsB_k, colsB, kernel, btransform)
                end
                st.packb += time_ns() - t0

                ic = 0
                while ic < Qm
                    mblock = min(mc_eff, Qm - ic)
                    m_slivers = cld(mblock, MRk)
                    (rng_mA, rng_mC) = if m_ramp
                        _ramp_slivers!(ws.m_desc_A, ws.m_desc_C, m_step[1], m_step[2], ic, mblock, MRk, m_slivers)
                    else
                        fill_offsets!((ws.m_buf_A, ws.m_buf_C), plan.mgroup, ic, mblock)
                        _classify_slivers!(ws.m_desc_A, ws.m_desc_C, ws.m_buf_A, ws.m_buf_C, mblock, MRk, m_slivers)
                    end
                    checked_span_bounds(plan.Abase, rng_mA, rng_kA, lenA)
                    checked_span_bounds(plan.Cbase, rng_mC, rng_nC, lenC)

                    t0 = time_ns()
                    for r in 0:(m_slivers - 1)
                        rfirst = r * MRk
                        rowsA = _axis_of(ws.m_desc_A[r + 1], ws.m_buf_A, rfirst)
                        apanel = _sliver_panel(ws.packed_a, MRp, kblock, r)
                        _pack_sliver!(unsafe_pack_a!, apanel, plan.Astorage, plan.Abase, rowsA, colsA_k, kernel, atransform)
                    end
                    t1 = time_ns()
                    st.packa += t1 - t0

                    for s in 0:(n_slivers - 1)
                        sfirst = s * NRk
                        colsC = _axis_of(ws.n_desc_C[s + 1], ws.n_buf_C, sfirst)
                        bpanel = _sliver_panel(ws.packed_b, NRp, kblock, s)
                        for r in 0:(m_slivers - 1)
                            rfirst = r * MRk
                            rowsC = _axis_of(ws.m_desc_C[r + 1], ws.m_buf_C, rfirst)
                            apanel = _sliver_panel(ws.packed_a, MRp, kblock, r)
                            unsafe_execute_micro_tile!(kernel, plan.Cstorage, plan.Cbase, rowsC, colsC, apanel, bpanel, kblock, alphaT, beta_eff)
                        end
                    end
                    st.kern += time_ns() - t1
                    ic += mblock
                end
                firstpanel = false
                pc += kblock
            end
            jc += nblock
        end
    end
    st.total += time_ns() - t_start
    return nothing
end

# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
execute!(plan, one(T), zero(T))  # warm-up
Cref = copy(C)
t_exec = median_time_s(() -> execute!(plan, one(T), zero(T)); reps = REPS)

st = StageTimes(0, 0, 0, 0)
timed_nest!(plan, one(T), zero(T), st)  # warm-up
@assert C ≈ Cref
rows = StageTimes[]
for r in 1:REPS
    s = StageTimes(0, 0, 0, 0)
    timed_nest!(plan, one(T), zero(T), s)
    push!(rows, s)
end
med(f) = median([f(s) for s in rows]) / 1.0e9

t_total = med(s -> s.total)
t_packb = med(s -> s.packb)
t_packa = med(s -> s.packa)
t_kern = med(s -> s.kern)
println()
@printf("execute!  median  %9.4f s   %7.2f GF/s\n", t_exec, flops / t_exec / 1.0e9)
@printf("nest      total   %9.4f s   %7.2f GF/s\n", t_total, flops / t_total / 1.0e9)
@printf("  pack B          %9.4f s  (%5.1f%%)   %7.2f ns/elem of B read\n", t_packb, 100 * t_packb / t_total, 1.0e9 * t_packb / (length(B) * cld(axis_length(plan.mgroup), plan.blocking.mc) / cld(axis_length(plan.mgroup), plan.blocking.mc)))
@printf("  pack A          %9.4f s  (%5.1f%%)   %7.2f ns/elem of A read (x%d N-blocks)\n", t_packa, 100 * t_packa / t_total, 1.0e9 * t_packa / (length(A) * cld(axis_length(plan.ngroup), plan.blocking.nc)), cld(axis_length(plan.ngroup), plan.blocking.nc))
@printf("  kernel+store    %9.4f s  (%5.1f%%)   %7.2f GF/s kernel-only\n", t_kern, 100 * t_kern / t_total, flops / t_kern / 1.0e9)
@printf("  other           %9.4f s  (%5.1f%%)\n", t_total - t_packb - t_packa - t_kern, 100 * (t_total - t_packb - t_packa - t_kern) / t_total)
