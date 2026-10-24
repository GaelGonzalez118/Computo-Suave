import random
from typing import Dict

class CromosomaDecimal(object):
    def __init__(self):
        self.n_genes: int = 6       # 6 tareas (T1 a T6)
        self.gen_width: int = 2     # 2 bits por gen para representar procesadores (1-3)
        self.score: float = -1.0    # f(x) = 1 / (1 + M(x))
        self.bits: int = 0x00000000 # Registro entero
        self.mascaras: Dict = {}
        
        max_value: int = 0b11       # Mascara de 2 bits
        
        for task in range(self.n_genes):
            shift = task * self.gen_width
            self.mascaras[f"p{task}"] = shift
            self.mascaras[f"m{task}"] = max_value << shift
            
    def llenar_random_data(self):
        for task in range(self.n_genes):
            proc_aleatorio = random.randint(1, 3)
            self.set_gen(task, proc_aleatorio)
        
    def set_gen(self, tarea: int, procesador: int) -> None:
        p = self.mascaras[f"p{tarea}"]
        m = self.mascaras[f"m{tarea}"]
        self.bits = (self.bits & ~m) | ((procesador & 0b11) << p)

    def get_gen(self, tarea: int) -> int:
        p = self.mascaras[f"p{tarea}"]
        m = self.mascaras[f"m{tarea}"]
        val = (self.bits & m) >> p
        return val if val in (1, 2, 3) else 1

    def a_cadena(self) -> str:
        return "".join(str(self.get_gen(i)) for i in range(self.n_genes))