using TensorOperations, QuasiStrided, TensorOperationsBenchmarks, Statistics
const TOB = TensorOperationsBenchmarks
LinearAlgebra = TensorOperations.LinearAlgebra; LinearAlgebra.BLAS.set_num_threads(1)
cases = filter(c -> c.id in ("ao2mo_2_dim16", "ao2mo_2_dim24", "ccsd_t_2_dim16", "ccsd_t_4_dim16", "ccsd_t_2_dim24", "ccsd_1_dim16"), TOB._tccg_cases((16, 24)))
append!(cases, filter(c -> c.id in ("mps_1site_D64", "mps_2site_D64"), TOB._mps_cases((64,))))
for c in cases
    s = c.spec
    if s isa TOB.ContractSpec
        dims(I) = ntuple(i -> s.dims[I[i]], length(I))
        A = randn(dims(s.IA)); B = randn(dims(s.IB)); C = zeros(dims(s.IC))
        pA, pB, pAB = TensorOperations.contract_indices(s.IA, s.IB, s.IC)
        f = () -> TensorOperations.tensorcontract!(C, A, pA, false, B, pB, false, pAB, 1.0, 0.0, QuasiStridedBackend())
    else
        ts = [randn(ntuple(i -> s.dims[abs(il[i])], length(il))...) for il in s.indexlists]
        f = () -> ncon(ts, s.indexlists, s.conjlist; order = s.order, output = s.output, backend = QuasiStridedBackend())
    end
    f(); ts_ = [(@elapsed f()) for _ in 1:15]
    println(rpad(c.id, 18), " ", round(median(ts_) * 1.0e6, digits = 1), " us")
end
