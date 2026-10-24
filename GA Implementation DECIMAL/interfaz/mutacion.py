from abc import ABC, abstractmethod
import random
from typing import List
from CromosomaDECIMAL import CromosomaDecimal

class Mutacion(ABC):
    @abstractmethod
    def mutacion_poblacion(self, poblacion: List[CromosomaDecimal]):
        pass

class MutacionDecimal(Mutacion):
    def __init__(self, probabilidad: float):
        super().__init__()
        self.prob = probabilidad

    def mutacion_poblacion(self, poblacion: List[CromosomaDecimal]):
        for crom in poblacion:
            for task in range(crom.n_genes):
                if random.random() < self.prob:
                    nuevo_proc = random.randint(1, 3)
                    crom.set_gen(task, nuevo_proc)