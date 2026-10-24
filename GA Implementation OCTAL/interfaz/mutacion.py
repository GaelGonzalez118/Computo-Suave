from abc import ABC, abstractmethod
import random
from typing import List
from CromosomaOCTAL import CromosomaOctal

class Mutacion(ABC):
    @abstractmethod
    def mutacion_poblacion(self, poblacion: List[CromosomaOctal]):
        pass

class MutacionOctal(Mutacion):
    def __init__(self, probabilidad: float):
        super().__init__()
        self.prob = probabilidad

    def mutacion_poblacion(self, poblacion: List[CromosomaOctal]):
        for crom in poblacion:
            for col in range(crom.n_genes):
                if random.random() < self.prob:
                    nueva_fila = random.randint(0, 7)
                    crom.set_gen(col, nueva_fila)