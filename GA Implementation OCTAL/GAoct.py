import random
import copy
from typing import Dict, List
from CromosomaOCTAL import CromosomaOctal
from interfaz import AptitudReinas, SeleccionTorneo, ReproduccionUnCruce, MutacionOctal

class GA(object):
    def __init__(self, num_inv: int, num_genes: int, hyperparams: Dict):
        self.poblacion: List[CromosomaOctal] = [CromosomaOctal() for _ in range(num_inv)]
        self.num_inv: int = num_inv
        self.hyperparams: Dict = hyperparams
        self.evaluar = hyperparams["Aptitud"]
        self.seleccion = hyperparams["Seleccion"]
        self.reproducion = hyperparams["Reproduccion"]
        self.mutacion = hyperparams["Mutacion"]
        self.num_remplazo = int(hyperparams["por-remplazo"] * num_inv)

        # Inicializacion aleatoria
        for i in range(num_inv):
            self.poblacion[i].llenar_random_data()

    def buscar_solucion(self, var: Dict = None):
        generacion: int = 1
        
        while True:
            # 1. Evaluar poblacion
            self.evaluar.evaluar_poblacion(self.poblacion, var)
            
            # Ordenar por aptitud (de mayor a menor)
            self.poblacion = sorted(self.poblacion, key=lambda c: c.score, reverse=True)
            
            mejor_score = self.poblacion[0].score
            sum_scores = sum(c.score for c in self.poblacion) / self.num_inv
            
            if generacion % 10 == 0 or generacion == 1 or mejor_score == 28:
                print(f"Generacion {generacion:04d} | Mejor Score: {mejor_score}/28 | Avg Score: {sum_scores:.2f}")

            # Criterio de parada: Se encontro la solucion sin conflictos (Score = 28)
            if mejor_score == 28:
                print(f"\nSolucion optima en la generacion {generacion}!")
                break

            # 2. Seleccion
            mating_pool = self.seleccion.seleccion_poblacion(self.poblacion)

            # 3. Elitismo
            generacion_nueva = copy.deepcopy(self.poblacion[:self.num_remplazo])

            # 4. Reproduccion y Mutacion
            self.reproducion.reproduccion_poblacion(generacion_nueva, mating_pool, self.num_inv)
            self.mutacion.mutacion_poblacion(generacion_nueva)

            generacion += 1
            self.poblacion = copy.deepcopy(generacion_nueva)


def imprimir_tablero(cromosoma: CromosomaOctal):
    posiciones = [cromosoma.get_gen(col) for col in range(8)]
    print("\nVector de solucion [Columna -> Fila]:", posiciones)
    print("\nRepresentacion del Tablero (Q = Reina, . = Vacio):\n")
    for fila in range(8):
        linea = ""
        for col in range(8):
            if posiciones[col] == fila:
                linea += " Q "
            else:
                linea += " . "
        print(linea)


if __name__ == "__main__":
    num_genes = 8
    cant_ind = 100

    gaparams = {
        "Aptitud": AptitudReinas(),
        "Seleccion": SeleccionTorneo(participantes=3),
        "Reproduccion": ReproduccionUnCruce(num_genes),
        "Mutacion": MutacionOctal(probabilidad=0.05),
        "por-remplazo": 0.2
    }

    algoritmoGA = GA(cant_ind, num_genes, gaparams)
    algoritmoGA.buscar_solucion()

    solucion = sorted(algoritmoGA.poblacion, key=lambda c: c.score, reverse=True)[0]
    print("\n================== SOLUCION FINAL ==================")
    print(f"Score Maximo: {solucion.score}/28")
    print(f"Conflictos Totales: {28 - solucion.score}")
    imprimir_tablero(solucion)