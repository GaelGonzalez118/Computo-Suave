import random
from abc import ABC, abstractmethod
from typing import List
from CromosomaNeuro import CromosomaNeuro

class Mutacion(ABC):
    @abstractmethod
    def mutacion_poblacion(self, poblacion: List[CromosomaNeuro]):
        pass

class MutacionNeuro(Mutacion):
    def __init__(self, probabilidad: float = 0.15):
        super().__init__()
        self.prob = probabilidad

    def mutacion_poblacion(self, poblacion: List[CromosomaNeuro]):
        for crom in poblacion:
            for gen_idx in range(crom.n_genes):
                if random.random() < self.prob:
                    ancho = crom.gen_specs[gen_idx][1]
                    max_val = (1 << ancho) - 1
                    nuevo_val = random.randint(0, max_val)
                    crom.set_gen(gen_idx, nuevo_val)
