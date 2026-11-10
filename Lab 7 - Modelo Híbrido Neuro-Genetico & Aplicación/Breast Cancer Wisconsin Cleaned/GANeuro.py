import random
import copy
from typing import Dict, List
from CromosomaNeuro import CromosomaNeuro

class GANeuro(object):
    def __init__(self, num_inv: int, num_genes: int, hyperparams: Dict):
        self.num_inv: int = num_inv
        self.num_genes: int = num_genes
        self.hyperparams: Dict = hyperparams
        self.evaluar = hyperparams["Aptitud"]
        self.seleccion = hyperparams["Seleccion"]
        self.reproducion = hyperparams["Reproduccion"]
        self.mutacion = hyperparams["Mutacion"]
        self.num_remplazo = max(1, int(hyperparams.get("por-remplazo", 0.2) * num_inv))
        
        self.poblacion: List[CromosomaNeuro] = [CromosomaNeuro() for _ in range(num_inv)]
        for i in range(num_inv):
            self.poblacion[i].llenar_random_data()

        self.historial_mejor_score = []
        self.historial_avg_score = []
        self.mejor_individuo_historico = None

    def buscar_solucion(self, max_generaciones: int = 10):
        generacion: int = 1
        
        while generacion <= max_generaciones:
            self.evaluar.evaluar_poblacion(self.poblacion)
            self.poblacion = sorted(self.poblacion, key=lambda c: c.score, reverse=True)
            
            mejor_actual = self.poblacion[0]
            mejor_score = mejor_actual.score
            sum_scores = sum(c.score for c in self.poblacion) / self.num_inv
            
            self.historial_mejor_score.append(mejor_score)
            self.historial_avg_score.append(sum_scores)

            if self.mejor_individuo_historico is None or mejor_score > self.mejor_individuo_historico.score:
                self.mejor_individuo_historico = copy.deepcopy(mejor_actual)

            print(f"Gen {generacion:02d}/{max_generaciones:02d} | "
                  f"Mejor Fitness: {mejor_score:.4f} (MSE: {mejor_actual.mse_val:.4f}, R2: {mejor_actual.r2_val:.4f}, Acc: {mejor_actual.acc_val*100:.1f}%) | "
                  f"Avg: {sum_scores:.4f} | "
                  f"Mejor config: {mejor_actual.a_cadena()}")

            if generacion == max_generaciones:
                break

            mating_pool = self.seleccion.seleccion_poblacion(self.poblacion)
            generacion_nueva = copy.deepcopy(self.poblacion[:self.num_remplazo])
            self.reproducion.reproduccion_poblacion(generacion_nueva, mating_pool, self.num_inv)
            self.mutacion.mutacion_poblacion(generacion_nueva[self.num_remplazo:])

            generacion += 1
            self.poblacion = copy.deepcopy(generacion_nueva)

        return self.mejor_individuo_historico
