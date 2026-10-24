from abc import ABC, abstractmethod
from typing import List, Dict
from CromosomaOCTAL import CromosomaOctal

class Aptitud(ABC):
    @abstractmethod
    def evaluar_poblacion(self, poblacion: List[CromosomaOctal], var: Dict = None):
        """Interfaz para asignar score a la poblacion"""
        pass

class AptitudReinas(Aptitud):
    def evaluar_poblacion(self, poblacion: List[CromosomaOctal], var: Dict = None):
        for cromosoma in poblacion:
            conflictos = 0
            
            # Comparar cada par de columnas (col1 < col2)
            for col1 in range(cromosoma.n_genes):
                row1 = cromosoma.get_gen(col1)
                for col2 in range(col1 + 1, cromosoma.n_genes):
                    row2 = cromosoma.get_gen(col2)
                    
                    # 1. Mismo renglon
                    if row1 == row2:
                        conflictos += 1
                    # 2. Misma diagonal: |row1 - row2| == |col1 - col2|
                    elif abs(row1 - row2) == abs(col1 - col2):
                        conflictos += 1
            
            # Funcion de aptitud: f(x) = 28 - C(x)
            cromosoma.score = 28 - conflictos