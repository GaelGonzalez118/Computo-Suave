import random
from typing import Dict

class CromosomaNeuro(object):
    """
    Cromosoma que codifica hiperparámetros de la Red Neuronal para Breast Cancer
    mediante manipulación de máscaras a nivel de bits.
    """
    OPCIONES_C1 = [8, 12, 16, 24, 32, 48, 64, 80]            # 3 bits
    OPCIONES_C2 = [0, 4, 8, 12, 16, 24, 32, 48]             # 3 bits (0 = 1 sola capa oculta)
    OPCIONES_ACT = ['sigmoid', 'relu']                       # 1 bit
    OPCIONES_LR = [0.1, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]    # 3 bits
    OPCIONES_BIAS = [False, True]                            # 1 bit

    def __init__(self):
        self.n_genes: int = 5
        self.score: float = -1.0
        self.mse_val: float = 999.0
        self.r2_val: float = -999.0
        self.acc_val: float = 0.0
        self.bits: int = 0x00000000

        self.gen_specs = [
            ("c1", 3, 0),
            ("c2", 3, 3),
            ("act", 1, 6),
            ("lr", 3, 7),
            ("bias", 1, 10),
        ]
        self.mascaras: Dict = {}
        for idx, (nombre, ancho, shift) in enumerate(self.gen_specs):
            self.mascaras[f"p{idx}"] = shift
            self.mascaras[f"m{idx}"] = ((1 << ancho) - 1) << shift

    def llenar_random_data(self):
        for idx, (nombre, ancho, shift) in enumerate(self.gen_specs):
            max_val = (1 << ancho) - 1
            val = random.randint(0, max_val)
            self.set_gen(idx, val)

    def set_gen(self, gen_idx: int, val: int) -> None:
        p = self.mascaras[f"p{gen_idx}"]
        m = self.mascaras[f"m{gen_idx}"]
        ancho = self.gen_specs[gen_idx][1]
        mask_base = (1 << ancho) - 1
        self.bits = (self.bits & ~m) | ((val & mask_base) << p)

    def get_gen(self, gen_idx: int) -> int:
        p = self.mascaras[f"p{gen_idx}"]
        m = self.mascaras[f"m{gen_idx}"]
        return (self.bits & m) >> p

    def decodificar(self) -> dict:
        idx_c1 = self.get_gen(0)
        idx_c2 = self.get_gen(1)
        idx_act = self.get_gen(2)
        idx_lr = self.get_gen(3)
        idx_bias = self.get_gen(4)

        return {
            "neuronas_c1": self.OPCIONES_C1[idx_c1 % len(self.OPCIONES_C1)],
            "neuronas_c2": self.OPCIONES_C2[idx_c2 % len(self.OPCIONES_C2)],
            "activacion": self.OPCIONES_ACT[idx_act % len(self.OPCIONES_ACT)],
            "tasa_aprendizaje": self.OPCIONES_LR[idx_lr % len(self.OPCIONES_LR)],
            "usa_bias": self.OPCIONES_BIAS[idx_bias % len(self.OPCIONES_BIAS)],
        }

    def a_cadena(self) -> str:
        d = self.decodificar()
        c2_str = f", Oculta2: {d['neuronas_c2']} neu" if d['neuronas_c2'] > 0 else ", 1 Capa"
        return f"[Oculta1: {d['neuronas_c1']} neu{c2_str} | Act: {d['activacion']} | LR: {d['tasa_aprendizaje']} | Bias: {d['usa_bias']}]"
