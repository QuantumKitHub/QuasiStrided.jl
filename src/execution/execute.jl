"""
    execute!(plan::ContractPlan, alpha::Number, beta::Number)

Compute `C = alpha * A * B + beta * C` for the contraction `plan` describes,
and return `plan.Cstorage`. Runs a BLIS five-loop nest over `plan.blocking`,
or a dedicated dot/outer-product path for degenerate extents. `beta` applies
exactly once per element and `beta == 0` never reads `C`; empty K or
`alpha == 0` never reads `A`/`B`. Operand bounds are checked once per macro
block. Allocation-free.
"""
function execute!(plan::ContractPlan{T}, alpha::Number, beta::Number) where {T}
    alphaT = convert(T, alpha)
    betaT = convert(T, beta)
    if iszero(alphaT)
        scale_all_of_C!(plan, betaT)
    else
        execute_path!(plan, alphaT, betaT, plan.path)
    end
    return plan.Cstorage
end

execute_path!(plan::ContractPlan, alphaT, betaT, ::EmptyPath) = nothing

execute_path!(plan::ContractPlan, alphaT, betaT, ::ScalePath) = scale_all_of_C!(plan, betaT)

execute_path!(plan::ContractPlan, alphaT, betaT, ::DotPath{MATB, W}) where {MATB, W} =
    (_execute_dot!(plan, alphaT, betaT, MATB, Val(W)); nothing)

execute_path!(plan::ContractPlan, alphaT, betaT, ::OuterPath{W}) where {W} = (
    _execute_outer!(
        plan, alphaT, betaT, axis_length(plan.mgroup), axis_length(plan.ngroup), Val(W)
    ); nothing
)

function execute_path!(plan::ContractPlan{T}, alphaT::T, betaT::T, path::NestPath) where {T}
    ws = plan.workspace
    GC.@preserve ws nest!(plan, alphaT, betaT, path, 0:(axis_length(plan.ngroup) - 1))
    return nothing
end

# `C *= beta` tile by tile, never reading A or B. Borrows the first tile of the
# M/N offset buffers.
function scale_all_of_C!(plan::ContractPlan{T}, betaT::T) where {T}
    ws = plan.workspace
    m_tile, n_tile = tile_size(plan.kernel)
    m_length = axis_length(plan.mgroup)
    n_length = axis_length(plan.ngroup)
    m_bufs = ws.m.offsets
    n_bufs = ws.n.offsets
    m_tile_start = 0
    while m_tile_start < m_length
        m_tile_length = min(m_tile, m_length - m_tile_start)
        (_, dM_C) = block_descriptors!(m_bufs, plan.mgroup, m_tile_start, m_tile_length)
        n_tile_start = 0
        while n_tile_start < n_length
            n_tile_length = min(n_tile, n_length - n_tile_start)
            (_, dN_C) = block_descriptors!(n_bufs, plan.ngroup, n_tile_start, n_tile_length)
            rowsC = axis_of(dM_C, m_bufs[2], 0)
            colsC = axis_of(dN_C, n_bufs[2], 0)
            GC.@preserve ws scale_micro_tile!(plan.Cstorage, plan.Cbase, rowsC, colsC, betaT)
            n_tile_start += n_tile_length
        end
        m_tile_start += m_tile_length
    end
    return nothing
end

@inline function scale_micro_tile!(
        storage::S, base::Int, rows::R, cols::C, beta
    ) where {S, R <: AbstractVector{Int}, C <: AbstractVector{Int}}
    destination = Tile(storage, base, rows, cols)
    scale_tile!(destination, beta)
    return nothing
end

"""
    contract!(C::StridedView, alpha::Number,
              A::StridedView, indA::NTuple{NA,Int},
              B::StridedView, indB::NTuple{NB,Int},
              beta::Number,
              indC::NTuple{NC,Int}; accumulator = nothing) where {NA,NB,NC}

Compute `C[indC] = alpha * sum_K A[indA] * B[indB] + beta * C[indC]`, with one
`Int` label per axis: a label in `indA` and `indB` but not `indC` is contracted
(K), and a label in `indC` and exactly one of `indA`/`indB` is free (M or N).
Any other label pattern, or a label repeated within one tuple, throws an
`ArgumentError`; matched labels of unequal axis length throw a
`DimensionMismatch`. Eltypes and `accumulator` are as in
[`plan_contract`](@ref). Equivalent to
`execute!(plan_contract(C, A, indA, B, indB, indC; accumulator), alpha, beta)`
— use those directly to reuse a plan across calls. Returns `C`.
"""
function contract!(
        C::StridedView, alpha::Number,
        A::StridedView, indA::NTuple{NA, Int},
        B::StridedView, indB::NTuple{NB, Int},
        beta::Number,
        indC::NTuple{NC, Int};
        accumulator::Union{Nothing, Type{Float32}, Type{Float64}} = nothing
    ) where {NA, NB, NC}
    plan = plan_contract(C, A, indA, B, indB, indC; accumulator)
    execute!(plan, alpha, beta)
    return C
end
