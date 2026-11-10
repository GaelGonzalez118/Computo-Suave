import random
from abc import ABC, abstractmethod
from typing import List
from CromosomaNeuro import CromosomaNeuro

class Seleccion(ABC):
    @abstractmethod
    def seleccion_poblacion(self, poblacion: List[CromosomaNeuro]) -> List[CromosomaNeuro]:
        pass

class SeleccionRuleta(Seleccion):
    def seleccion_poblacion(self, poblacion: List[CromosomaNeuro]) -> List[CromosomaNeuro]:
        ganadores = []
        sum_fitness = sum(c.score for c in poblacion)
        
        if sum_fitness <= 0:
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

class SeleccionTorneo(Seleccion):
    def __init__(self, k: int = 3):
        self.k = k

    def seleccion_poblacion(self, poblacion: List[CromosomaNeuro]) -> List[CromosomaNeuro]:
        ganadores = []
        n = len(poblacion)
        for _ in range(n):
            aspirantes = random.sample(poblacion, min(self.k, n))
            mejor = max(aspirantes, key=lambda c: c.score)
            ganadores.append(mejor)
        return ganadores
