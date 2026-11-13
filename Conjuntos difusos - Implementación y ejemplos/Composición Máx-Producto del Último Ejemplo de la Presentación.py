import numpy as np

def composicion_max_producto(R, S):
    rows_R, cols_R = R.shape
    rows_S, cols_S = S.shape
    T = np.zeros((rows_R, cols_S))
    for i in range(rows_R):
        for j in range(cols_S):
            T[i, j] = np.max(R[i, :] * S[:, j])
    return T

C = np.array([
    [0.9, 0.4, 0.1],
    [0.1, 0.4, 0.9]
])

MB = np.array([
    [0.0, 0.0],
    [0.6, 0.0],
    [1.0, 0.7]
])

print("Relacion C(u, v):\n", C)
print("\nRelacion MB(v, w):\n", MB)
resultado_prod = composicion_max_producto(C, MB)
print("\nComposicion Max-Producto (C o MB):\n", resultado_prod)