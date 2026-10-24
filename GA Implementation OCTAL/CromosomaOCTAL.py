import random
from typing import Dict, List

class CromosomaOctal(object):
    def __init__(self):
        self.n_genes: int = 8       # 8 columnas (0 a 7)
        self.gen_width: int = 3     # Rango octal (0 a 7)
        self.score: int = -1        # f(x) = 28 - C(x)
        self.bits: int = 0x00000000 # Registro entero de 32 bits
        self.mascaras: Dict = {}
        
        max_value: int = 0b111      # 7 en octal
        
        for col in range(self.n_genes):
            shift = col * self.gen_width
            self.mascaras[f"p{col}"] = shift
            self.mascaras[f"m{col}"] = max_value << shift
            
    def llenar_random_data(self):
        for col in range(self.n_genes):
            fila_aleatoria = random.randint(0, 7)
            self.set_gen(col, fila_aleatoria)
        
    def set_gen(self, columna: int, fila: int) -> None:
        p = self.mascaras[f"p{columna}"]
        m = self.mascaras[f"m{columna}"]
        self.bits = (self.bits & ~m) | ((fila & 0b111) << p)

    def get_gen(self, columna: int) -> int:
        p = self.mascaras[f"p{columna}"]
        m = self.mascaras[f"m{columna}"]
        return (self.bits & m) >> p