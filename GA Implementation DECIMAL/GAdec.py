import random
import copy
from typing import Dict, List
from CromosomaDECIMAL import CromosomaDecimal
from interfaz import AptitudProcesadores, SeleccionRuleta, ReproduccionUnCruce, MutacionDecimal

class GA(object):
    def __init__(self, num_inv: int, num_genes: int, hyperparams: Dict):
        self.poblacion: List[CromosomaDecimal] = [CromosomaDecimal() for _ in range(num_inv)]
        self.num_inv: int = num_inv
        self.hyperparams: Dict = hyperparams
        self.evaluar = hyperparams["Aptitud"]
        self.seleccion = hyperparams["Seleccion"]
        self.reproducion = hyperparams["Reproduccion"]
        self.mutacion = hyperparams["Mutacion"]
        self.num_remplazo = int(hyperparams["por-remplazo"] * num_inv)

        for i in range(num_inv):
            self.poblacion[i].llenar_random_data()

    def buscar_solucion(self, max_generaciones: int = 200):
        generacion: int = 1
        
        while generacion <= max_generaciones:
            # 1. Evaluar población
            self.evaluar.evaluar_poblacion(self.poblacion)
            
            # Ordenar de mayor a menor score
            self.poblacion = sorted(self.poblacion, key=lambda c: c.score, reverse=True)
            
            mejor_score = self.poblacion[0].score
            max_carga_actual = round((1.0 / mejor_score) - 1)
            sum_scores = sum(c.score for c in self.poblacion) / self.num_inv
            
            if generacion % 10 == 0 or generacion == 1:
                print(f"Gen {generacion:03d} | M(x) Minimo: {max_carga_actual} | Mejor Fitness: {mejor_score:.5f} | Avg Fitness: {sum_scores:.5f}")

            # 2. Selección por Ruleta
            mating_pool = self.seleccion.seleccion_poblacion(self.poblacion)

            # 3. Elitismo
            generacion_nueva = copy.deepcopy(self.poblacion[:self.num_remplazo])

            # 4. Reproducción y Mutación
            self.reproducion.reproduccion_poblacion(generacion_nueva, mating_pool, self.num_inv)
            self.mutacion.mutacion_poblacion(generacion_nueva)

            generacion += 1
            self.poblacion = copy.deepcopy(generacion_nueva)


def imprimir_desglose(cromosoma: CromosomaDecimal, tiempos: List[int]):
    cargas = {1: [], 2: [], 3: []}
    tiempos_proc = {1: 0, 2: 0, 3: 0}
    
    for tarea in range(cromosoma.n_genes):
        proc = cromosoma.get_gen(tarea)
        cargas[proc].append(f"T{tarea+1}")
        tiempos_proc[proc] += tiempos[tarea]

    print("\n================== DETALLE DE ASIGNACION ==================")
    print(f"Cromosoma Optimo: {cromosoma.a_cadena()}")
    print("-" * 50)
    print(f"{'Procesador':<12} | {'Tareas Asignadas':<20} | {'Tiempo Total'}")
    print("-" * 50)
    for p in range(1, 4):
        tareas_str = ", ".join(cargas[p]) if cargas[p] else "Ninguna"
        print(f"P{p:<11} | {tareas_str:<20} | {tiempos_proc[p]}")
    print("-" * 50)
    max_carga = max(tiempos_proc.values())
    print(f"Tiempo total maximo M(x): {max_carga}")
    print(f"Fitness f(x) = 1 / (1 + {max_carga}): {cromosoma.score:.5f}")


if __name__ == "__main__":
    num_genes = 6
    cant_ind = 50
    tiempos_tareas = [8, 5, 7, 3, 6, 4]

    gaparams = {
        "Aptitud": AptitudProcesadores(tiempos_tareas),
        "Seleccion": SeleccionRuleta(),
        "Reproduccion": ReproduccionUnCruce(num_genes),
        "Mutacion": MutacionDecimal(probabilidad=0.05),
        "por-remplazo": 0.2
    }

    algoritmoGA = GA(cant_ind, num_genes, gaparams)
    algoritmoGA.buscar_solucion(max_generaciones=100)

    solucion = sorted(algoritmoGA.poblacion, key=lambda c: c.score, reverse=True)[0]
    imprimir_desglose(solucion, tiempos_tareas)