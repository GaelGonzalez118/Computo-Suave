from abc import ABC, abstractmethod
from typing import List, Dict
from CromosomaDECIMAL import CromosomaDecimal

class Aptitud(ABC):
    @abstractmethod
    def evaluar_poblacion(self, poblacion: List[CromosomaDecimal], var: Dict = None):
        """Interfaz para asignar score a la poblacion"""
        pass

class AptitudProcesadores(Aptitud):
    def __init__(self, tiempos_tareas: List[int] = None):
        # Tiempos de ejecucion: T1=8, T2=5, T3=7, T4=3, T5=6, T6=4
        if tiempos_tareas is None:
            self.tiempos = [8, 5, 7, 3, 6, 4]
        else:
            self.tiempos = tiempos_tareas

    def evaluar_poblacion(self, poblacion: List[CromosomaDecimal], var: Dict = None):
        for cromosoma in poblacion:
            cargas = [0, 0, 0]  # Cargas de los procesadores P1, P2 y P3
            
            for tarea in range(cromosoma.n_genes):
                proc = cromosoma.get_gen(tarea)  # Retorna 1, 2 o 3
                cargas[proc - 1] += self.tiempos[tarea]
            
            max_carga = max(cargas)
            # Funcion de aptitud: f(x) = 1 / (1 + M(x))
            cromosoma.score = 1.0 / (1.0 + max_carga)