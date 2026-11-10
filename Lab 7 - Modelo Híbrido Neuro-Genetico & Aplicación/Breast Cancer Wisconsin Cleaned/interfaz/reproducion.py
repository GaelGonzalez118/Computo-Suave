import random
from abc import ABC, abstractmethod
from typing import List
from CromosomaNeuro import CromosomaNeuro

class Reproduccion(ABC):
    @abstractmethod
    def reproduccion_poblacion(self, poblacion: List[CromosomaNeuro], potencial: List[CromosomaNeuro], max_poblacion: int) -> None:
        pass

class ReproduccionUnCruce(Reproduccion):
    def __init__(self, num_genes: int = 5):
        super().__init__()
        self.num_genes = num_genes

    def reproduccion_poblacion(self, poblacion: List[CromosomaNeuro], potencial: List[CromosomaNeuro], max_poblacion: int) -> None:
        curr_po = len(poblacion)
        ord_padres = sorted(potencial, key=lambda c: c.score, reverse=True)
        id_padres: int = 0

        while curr_po < max_poblacion:
            hijoA = CromosomaNeuro()
            hijoB = CromosomaNeuro()
            punto_cruce = random.randint(1, self.num_genes - 1)

            padreA = ord_padres[id_padres % len(ord_padres)]
            padreB = ord_padres[(id_padres + 1) % len(ord_padres)]

            for gen_idx in range(self.num_genes):
                if gen_idx < punto_cruce:
                    hijoA.set_gen(gen_idx, padreA.get_gen(gen_idx))
                    hijoB.set_gen(gen_idx, padreB.get_gen(gen_idx))
                else:
                    hijoA.set_gen(gen_idx, padreB.get_gen(gen_idx))
                    hijoB.set_gen(gen_idx, padreA.get_gen(gen_idx))

            if curr_po < max_poblacion:
                poblacion.append(hijoA)
                curr_po += 1
            if curr_po < max_poblacion:
                poblacion.append(hijoB)
                curr_po += 1

            id_padres += 1
