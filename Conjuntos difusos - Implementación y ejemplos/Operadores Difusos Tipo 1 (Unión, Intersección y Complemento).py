import numpy as np

def union_difusa(A, B):
    """Union difusa utilizando el operador Maximo (S-Norma)."""
    return np.maximum(A, B)

def interseccion_difusa(A, B):
    """Interseccion difusa utilizando el operador Minimo (T-Norma)."""
    return np.minimum(A, B)

def complemento_difuso(A):
    """Complemento difuso (1 - mu_A)."""
    return 1.0 - A

# Ejemplo:
A = np.array([0.1, 0.5, 0.8])
B = np.array([0.4, 0.3, 0.9])

print("Union (Max):", union_difusa(A, B))
print("Interseccion (Min):", interseccion_difusa(A, B))
print("Complemento de A:", complemento_difuso(A))