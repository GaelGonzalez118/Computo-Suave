import random
from abc import ABC, abstractmethod
from typing import List
from CromosomaOCTAL import CromosomaOctal

class Seleccion(ABC):
    @abstractmethod
    def seleccion_poblacion(self, poblacion: List[CromosomaOctal]) -> List[CromosomaOctal]:
        pass

class SeleccionTorneo(Seleccion):
    def __init__(self, participantes: int = 3) -> None:
        super().__init__()
        self.participantes = participantes

    def seleccion_poblacion(self, poblacion: List[CromosomaOctal]) -> List[CromosomaOctal]:
        ganadores = []
        for _ in range(len(poblacion)):
            competidores = random.sample(poblacion, self.participantes)
            ganador = max(competidores, key=lambda c: c.score)
            ganadores.append(ganador)
        return ganadores