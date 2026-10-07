include("../helpers.jl")

# Label order within M/N and the M/N orientation swap.
const _lo_order = QuasiStrided.order_free_labels
const _lo_run = QuasiStrided.leading_unit_run
_lo_swap(morder, norder, indC, C, m_tile_asis, m_tile_swapped = m_tile_asis) = QuasiStrided.prefer_swap(
    _lo_run(morder, indC, C), _lo_run(norder, indC, C), m_tile_asis, m_tile_swapped
)

# A view with these strides over dummy data (only strides/size are read).
_lo_view(sz::NTuple{N, Int}, st::NTuple{N, Int}) where {N} =
    StridedView(zeros(Float64, 4096), sz, st, 2048, identity)

# The four TCCG `ccsd_t_*` shapes, C stored in (a,b,c,i,j,k) order, labelled
# as the TensorOperations adapter labels them.
const LO_IC = (:a, :b, :c, :i, :j, :k)
const LO_CASES = (
    ("ccsd_t_1", (:i, :j, :m, :a), (:m, :k, :b, :c)),
    ("ccsd_t_2", (:i, :j, :m, :b), (:m, :k, :a, :c)),
    ("ccsd_t_3", (:i, :j, :m, :c), (:m, :k, :a, :b)),
    ("ccsd_t_4", (:i, :k, :m, :b), (:m, :j, :a, :c)),
)
function _lo_labels(IA, IB)
    pA, pB, pAB = TO.contract_indices(IA, IB, LO_IC)
    return QuasiStrided.contraction_labels(pA, pB, pAB), (pA, pB, pAB)
end
