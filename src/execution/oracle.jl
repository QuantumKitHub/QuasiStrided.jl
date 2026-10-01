# The tile-by-tile correctness oracle for `execute!`: per register tile, K in
# `k_block` panels, pack one sliver each and call the checked `execute_tile!`.
# Uses only its own `tw_*` buffers, so it shares no mutable state with
# `execute!`. Needs a plan built with `oracle = true`. Unlike `execute!`, rounds
# a C narrower than the compute type once per K block.
function execute_tilewise!(plan::ContractPlan{T}, alpha::Number, beta::Number) where {T}
    ws = plan.workspace
    # `tw_packed_a` is never empty for a legal blocking unless `oracle = false`.
    isempty(ws.tw_packed_a) && throw(
        ArgumentError(
            "execute_tilewise! needs the oracle buffers, which this plan was built " *
                "without; re-plan with `oracle = true`"
        )
    )

    alphaT = convert(T, alpha)
    betaT = convert(T, beta)

    kernel = plan.kernel
    m_tile, n_tile = tile_size(kernel)

    m_length = axis_length(plan.mgroup)
    n_length = axis_length(plan.ngroup)
    k_length = axis_length(plan.kgroup)

    (m_length == 0 || n_length == 0) && return plan.Cstorage

    if k_length == 0 || iszero(alphaT)
        _scale_all_of_C!(plan, betaT, m_tile, n_tile, m_length, n_length)
        return plan.Cstorage
    end

    m_bufs = (ws.tile_m_buf_A, ws.tile_m_buf_C)
    n_bufs = (ws.tile_n_buf_B, ws.tile_n_buf_C)
    k_bufs = (ws.tw_k_buf_A, ws.tw_k_buf_B)
    k_block = plan.blocking.k_block

    # Scatter axes borrow pointers into `ws`.
    GC.@preserve ws begin
        m_tile_start = 0
        while m_tile_start < m_length
            m_tile_length = min(m_tile, m_length - m_tile_start)
            (dM_A, dM_C) = block_descriptors!(m_bufs, plan.mgroup, m_tile_start, m_tile_length)
            rowsA = _axis_of(dM_A, ws.tile_m_buf_A, 0)
            rowsC = _axis_of(dM_C, ws.tile_m_buf_C, 0)

            n_tile_start = 0
            while n_tile_start < n_length
                n_tile_length = min(n_tile, n_length - n_tile_start)
                (dN_B, dN_C) = block_descriptors!(n_bufs, plan.ngroup, n_tile_start, n_tile_length)
                colsB = _axis_of(dN_B, ws.tile_n_buf_B, 0)
                colsC = _axis_of(dN_C, ws.tile_n_buf_C, 0)

                k_block_start = 0
                firstpanel = true
                while k_block_start < k_length
                    k_block_length = min(k_block, k_length - k_block_start)
                    (dK_A, dK_B) = block_descriptors!(k_bufs, plan.kgroup, k_block_start, k_block_length)
                    colsK_A = _axis_of(dK_A, ws.tw_k_buf_A, 0)
                    rowsK_B = _axis_of(dK_B, ws.tw_k_buf_B, 0)

                    # Whole single-sliver buffers, so no `_sliver_panel`.
                    _pack_sliver!(
                        ws.tw_packed_a, plan.Astorage, plan.Abase, rowsA, colsK_A,
                        sliver_spec(kernel, 1), plan.atransform
                    )
                    _pack_sliver!(
                        ws.tw_packed_b, plan.Bstorage, plan.Bbase, colsB, rowsK_B,
                        sliver_spec(kernel, 2), plan.btransform
                    )

                    beta_eff = firstpanel ? betaT : one(T)
                    _execute_micro_tile!(
                        kernel, plan.Cstorage, plan.Cbase, rowsC, colsC,
                        ws.tw_packed_a, ws.tw_packed_b, k_block_length, alphaT, beta_eff
                    )

                    firstpanel = false
                    k_block_start += k_block_length
                end

                n_tile_start += n_tile_length
            end
            m_tile_start += m_tile_length
        end
    end

    return plan.Cstorage
end
