from abc import ABC, abstractmethod
import random
from typing import List
from CromosomaDECIMAL import CromosomaDecimal

class Reproduccion(ABC):
    @abstractmethod
    def reproduccion_poblacion(self, poblacion: List[CromosomaDecimal], potencial: List[CromosomaDecimal], max_poblacion: int) -> None:
        pass

class ReproduccionUnCruce(Reproduccion):
    def __init__(self, num_genes: int = 6):
        super().__init__()
        self.num_genes = num_genes

    def reproduccion_poblacion(self, poblacion: List[CromosomaDecimal], potencial: List[CromosomaDecimal], max_poblacion: int) -> None:
        curr_po = len(poblacion)
        ord_padres = sorted(potencial, key=lambda c: c.score, reverse=True)
        id_padres: int = 0

        while curr_po < max_poblacion:
            hijoA = CromosomaDecimal()
            hijoB = CromosomaDecimal()
            punto_cruce = random.randint(1, self.num_genes - 1)

            padreA = ord_padres[id_padres % len(ord_padres)]
            padreB = ord_padres[(id_padres + 1) % len(ord_padres)]

            for task in range(self.num_genes):
                if task < punto_cruce:
                    hijoA.set_gen(task, padreA.get_gen(task))
                    hijoB.set_gen(task, padreB.get_gen(task))
                else:
                    hijoA.set_gen(task, padreB.get_gen(task))
                    hijoB.set_gen(task, padreA.get_gen(task))

            if curr_po < max_poblacion:
                poblacion.append(hijoA)
                curr_po += 1
            if curr_po < max_poblacion:
                poblacion.append(hijoB)
                curr_po += 1

            id_padres += 1