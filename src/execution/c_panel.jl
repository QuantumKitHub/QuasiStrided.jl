# The panel of C: for an eltype of C narrower than the compute type, partial
# sums across K blocks live in a compute-type panel in the workspace, one N
# block at a time.

# Per N block: load the block of C into the panel (unless `beta == 0`), run the
# nest on a plan whose C is the panel holding just that block, with dense M/N
# maps, and round the block back into C.
function execute_path!(
        plan::ContractPlan{T}, alphaT::T, betaT::T, ::PanelPath{P}
    ) where {T, P}
    ws = plan.workspace
    m_length = axis_length(plan.mgroup)
    n_length = axis_length(plan.ngroup)
    n_block = plan.blocking.n_block
    panel_plan = ContractPlan(
        plan; mgroup = dense_second_map(plan.mgroup, 1), ngroup = dense_second_map(plan.ngroup, m_length),
        Cstorage = ws.c_panel
    )
    n_block_start = 0
    while n_block_start < n_length
        n_block_length = min(n_block, n_length - n_block_start)
        n_range = n_block_start:(n_block_start + n_block_length - 1)
        iszero(betaT) || panel_copy!(plan, n_range, true)
        block_plan = ContractPlan(panel_plan; Cbase = -n_block_start * m_length)
        GC.@preserve ws nest!(block_plan, alphaT, betaT, P(), n_range)
        panel_copy!(plan, n_range, false)
        n_block_start += n_block_length
    end
    return nothing
end

# `g` with its second map replaced by the column-major one scaled by `step`.
@inline function dense_second_map(g::AxisGroup{D, 2}, step::Int) where {D}
    dense = ntuple(d -> step * prod(ntuple(i -> i < d ? g.lengths[i] : 1, Val(D))), Val(D))
    return AxisGroup(g.lengths, (g.strides[1], dense))
end

# Columns `n_range` of `plan`'s C into (`load`) or out of the panel, converting
# to the destination's eltype. Borrows the M/N offset buffers, which the nest
# refills before reading them again.
function panel_copy!(plan::ContractPlan, n_range::UnitRange{Int}, load::Bool)
    ws = plan.workspace
    panel = ws.c_panel
    C = plan.Cstorage
    m_length = axis_length(plan.mgroup)
    m_block = plan.blocking.m_block
    n_bufC = ws.n.offsets[2]
    fill_offsets!(ws.n.offsets, plan.ngroup, first(n_range), length(n_range))
    m_block_start = 0
    while m_block_start < m_length
        m_block_length = min(m_block, m_length - m_block_start)
        m_bufC = ws.m.offsets[2]
        fill_offsets!(ws.m.offsets, plan.mgroup, m_block_start, m_block_length)
        for j in 1:length(n_range), i in 1:m_block_length
            c = plan.Cbase + m_bufC[i] + n_bufC[j] + 1
            q = m_block_start + i + (j - 1) * m_length
            if load
                panel[q] = C[c]
            else
                C[c] = panel[q]
            end
        end
        m_block_start += m_block_length
    end
    return nothing
end
