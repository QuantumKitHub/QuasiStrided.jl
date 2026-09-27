t0 = time(); using TensorOperations, QuasiStrided; tload = time() - t0
function first(T)
    A = randn(T, 20, 30, 10); B = randn(T, 30, 15); C = zeros(T, 20, 10, 15)
    t = @elapsed @tensor backend = QuasiStridedBackend() C[a, b, n] = A[a, k, b] * B[k, n]
    return t
end
ts = [(T, first(T)) for T in (Float64, ComplexF64)]
t2 = @elapsed @tensor backend = QuasiStridedBackend() D[a, b] := randn(8, 1)[a, k] * randn(1, 9)[k, b]
println("load ", round(tload, digits = 1), " s; first call ", join(["$T $(round(t, digits = 1)) s" for (T, t) in ts], ", "), "; first K=1 outer ", round(t2, digits = 1), " s")
