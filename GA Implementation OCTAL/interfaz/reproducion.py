from abc import ABC, abstractmethod
import random
from typing import List
from CromosomaOCTAL import CromosomaOctal

class Reproduccion(ABC):
    @abstractmethod
    def reproduccion_poblacion(self, poblacion: List[CromosomaOctal], potencial: List[CromosomaOctal], max_poblacion: int) -> None:
        pass

class ReproduccionUnCruce(Reproduccion):
    def __init__(self, num_genes: int = 8):
        super().__init__()
        self.num_genes = num_genes

    def reproduccion_poblacion(self, poblacion: List[CromosomaOctal], potencial: List[CromosomaOctal], max_poblacion: int) -> None:
        curr_po = len(poblacion)
        ord_padres = sorted(potencial, key=lambda c: c.score, reverse=True)
        id_padres: int = 0

        while curr_po < max_poblacion:
            hijoA = CromosomaOctal()
            hijoB = CromosomaOctal()
            punto_cruce = random.randint(1, self.num_genes - 1)

            padreA = ord_padres[id_padres % len(ord_padres)]
            padreB = ord_padres[(id_padres + 1) % len(ord_padres)]

            # Intercambiar posiciones de filas
            for col in range(self.num_genes):
                if col < punto_cruce:
                    hijoA.set_gen(col, padreA.get_gen(col))
                    hijoB.set_gen(col, padreB.get_gen(col))
                else:
                    hijoA.set_gen(col, padreB.get_gen(col))
                    hijoB.set_gen(col, padreA.get_gen(col))

            if curr_po < max_poblacion:
                poblacion.append(hijoA)
                curr_po += 1
            if curr_po < max_poblacion:
                poblacion.append(hijoB)
                curr_po += 1

            id_padres += 1