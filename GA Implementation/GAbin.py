import random
import copy
from typing import Dict, List
from CromosomaBin import CromosomaBinario
from interfaz import AptitudMochila, SeleccionTorneo, ReproduccionUnCruce, MutacionFlip

class GA(object):
    def __init__(self, num_inv:int, num_genes: int, hyperparams: Dict):
        self.poblacion: List[CromosomaBinario] = [CromosomaBinario(num_genes) for i in range(num_inv)]
        self.num_inv: int = num_inv
        self.hyperparams: Dict = hyperparams
        self.evaluar = hyperparams["Aptitud"]
        self.seleccion = hyperparams["Seleccion"]
        self.reproducion = hyperparams["Reproduccion"]
        self.mutacion = hyperparams["Mutacion"]
        self.num_remplazo = int(hyperparams["por-remplazo"]*num_inv)
        
        #inicializar
        for i in range(num_inv):
            self.poblacion[i].llenar_random_data()

    def buscar_solucion(self, tolerancia: float, var: Dict):
        avg_tol: float = 99999999.99
        sum_score_prev: float = 0
        sum_score: float = 1
        generacion: int = 1
        mating_pool = None
        generacion_inicial: List = copy.deepcopy(self.poblacion)
        generacion_nueva: List = None
        fail: bool = False
        fail_count: int = 0
        fail_score: float = 0.0
        
        while (not fail):
            self.evaluar.evaluar_poblacion(self.poblacion, var)
            sum_scores: float = sum([cromosoma.score for cromosoma in self.poblacion])
            sum_scores /= self.num_inv
            
            if sum_score != 0:
                avg_tol = abs(sum_scores - sum_score_prev) / sum_score
                
            sum_score_prev = sum_scores
            
            if (generacion+1) % 100 == 0:
                print(f"generacion {generacion:04d} avg_err {avg_tol:.5f} avg_score {sum_scores:.5f}")
                
            if fail_score == avg_tol:
                fail_count += 1
            if fail_count >= 10:
                fail = True
                break
            if avg_tol < tolerancia:
                break
                
            fail_score = avg_tol
            
            self.poblacion = sorted(self.poblacion, key=lambda c: c.score, reverse=True)
            mating_pool = self.seleccion.seleccion_poblacion(self.poblacion)
            generacion_nueva = copy.deepcopy(self.poblacion[:self.num_remplazo])
            
            self.reproducion.reproduccion_poblacion(generacion_nueva, mating_pool, self.num_inv)
            self.mutacion.mutacion_poblacion(generacion_nueva)
            
            generacion += 1
            self.poblacion = copy.deepcopy(generacion_nueva)

def sum_pesos_test(pesos, cromosoma):
    s = 0
    for i in range(cromosoma.n_genes):
        if cromosoma.get_bits(i+1) == 1:
            s += pesos[i]
    return s

if __name__ == "__main__":
    variables = {}
    variables["pesos"] = [random.randint(1, 10) for _ in range(32)]
    variables["valores"] = [random.randint(10, 100) for _ in range(32)]
    variables["capacidad"] = 80 #kg
    
    num_genes = 32
    cant_ind = 100
    
    gaparams = {}
    gaparams["Aptitud"] = AptitudMochila()
    gaparams["Seleccion"] = SeleccionTorneo(3)
    gaparams["Reproduccion"] = ReproduccionUnCruce(num_genes)
    gaparams["Mutacion"] = MutacionFlip(0.05)
    gaparams["por-remplazo"] = 0.5 #%
    
    algoritmoGA = GA(cant_ind, num_genes, gaparams)
    algoritmoGA.buscar_solucion(1e-4, variables)
    
    print(f"Cromosoma")
    solucion = sorted(algoritmoGA.poblacion, key=lambda c: c.score, reverse=True)
    for i, cromosoma in enumerate(solucion):
        print(f"{i:4d}|| {cromosoma.bits:032b} {cromosoma.score} {sum_pesos_test(variables['pesos'], cromosoma):4d} kg")