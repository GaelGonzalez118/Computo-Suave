import random
from abc import ABC, abstractmethod
from typing import List
from CromosomaDECIMAL import CromosomaDecimal

class Seleccion(ABC):
    @abstractmethod
    def seleccion_poblacion(self, poblacion: List[CromosomaDecimal]) -> List[CromosomaDecimal]:
        pass

class SeleccionRuleta(Seleccion):
    def seleccion_poblacion(self, poblacion: List[CromosomaDecimal]) -> List[CromosomaDecimal]:
        ganadores = []
        sum_fitness = sum(c.score for c in poblacion)
        
        if sum_fitness == 0:
            return [random.choice(poblacion) for _ in range(len(poblacion))]

        for _ in range(len(poblacion)):
            pick = random.uniform(0, sum_fitness)
            current = 0.0
            for cromosoma in poblacion:
                current += cromosoma.score
                if current >= pick:
                    ganadores.append(cromosoma)
                    break
                    
        return ganadores