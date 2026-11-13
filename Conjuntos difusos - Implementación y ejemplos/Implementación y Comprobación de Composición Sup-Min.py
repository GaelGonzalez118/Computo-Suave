import numpy as np

def composicion_sup_min(R, S):
    rows_R, cols_R = R.shape
    rows_S, cols_S = S.shape
    assert cols_R == rows_S, "Las dimensiones internas de las matrices deben coincidir."
    
    T = np.zeros((rows_R, cols_S))
    for i in range(rows_R):
        for j in range(cols_S):
            T[i, j] = np.max(np.minimum(R[i, :], S[:, j]))
    return T

# Comprobacion:
R = np.array([[1, 0], 
              [0, 1]])

S = np.array([[1, 0, 0], 
              [1, 0, 1]])

resultado_RS = composicion_sup_min(R, S)
print("Resultado R o S (Sup-Min):\n", resultado_RS)