"""Entrenamiento de la red optimizada mediante algoritmo genético.

La búsqueda de hiperparámetros y la comparación contra la red base se
implementan en ``optimizar_cancer.py``. Este archivo es el punto de entrada
correspondiente al modelo 2 del dataset de cáncer.
"""

from optimizar_cancer import ejecutar_optimizacion_cancer


if __name__ == "__main__":
    ejecutar_optimizacion_cancer()
