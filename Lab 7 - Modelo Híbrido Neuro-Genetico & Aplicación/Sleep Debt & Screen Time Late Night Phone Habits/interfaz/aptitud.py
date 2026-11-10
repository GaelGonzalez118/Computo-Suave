from abc import ABC, abstractmethod
from typing import List, Dict
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from CromosomaNeuro import CromosomaNeuro

def relu(x):
    return np.maximum(0, x)

def relu_derivada(x):
    return (x > 0).astype(float)

def sigmoid(x):
    x_clipped = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x_clipped))

def sigmoid_derivada(x):
    s = sigmoid(x)
    return s * (1.0 - s)

def linear(x):
    return x

def linear_derivada(x):
    return np.ones_like(x)

class CapaModular:
    def __init__(self, n_entradas, n_neuronas, activacion_nombre='relu', usa_bias=True):
        self.n_neuronas = n_neuronas
        self.usa_bias = usa_bias
        self.activacion_nombre = activacion_nombre.lower()
        
        if self.activacion_nombre == 'relu':
            self.activacion = relu
            self.derivada = relu_derivada
        elif self.activacion_nombre == 'sigmoid':
            self.activacion = sigmoid
            self.derivada = sigmoid_derivada
        elif self.activacion_nombre == 'linear':
            self.activacion = linear
            self.derivada = linear_derivada
        else:
            raise ValueError(f"Activacion '{activacion_nombre}' no soportada")

        # Inicialización He / Xavier para estabilidad
        escala = np.sqrt(2.0 / n_entradas) if self.activacion_nombre == 'relu' else 0.1
        if self.usa_bias:
            self.pesos = np.random.randn(n_entradas + 1, n_neuronas) * escala
        else:
            self.pesos = np.random.randn(n_entradas, n_neuronas) * escala

        self.entrada = None
        self.z = None

    def feedforward(self, x):
        if self.usa_bias:
            unos = np.ones((x.shape[0], 1))
            self.entrada = np.hstack((x, unos))
        else:
            self.entrada = x

        self.z = np.dot(self.entrada, self.pesos)
        return self.activacion(self.z)

    def backpropagation(self, error_salida, tasa_aprendizaje=0.01):
        delta = error_salida * self.derivada(self.z)
        gradientes_pesos = np.dot(self.entrada.T, delta)
        error_hacia_atras = np.dot(delta, self.pesos.T)
        
        if self.usa_bias:
            error_hacia_atras = error_hacia_atras[:, :-1]
            
        self.pesos -= tasa_aprendizaje * (gradientes_pesos / self.entrada.shape[0])
        return error_hacia_atras

class RedNeuronalModular:
    def __init__(self):
        self.capas = []
        self.historial_loss = []
        
    def agregar_capa(self, capa):
        self.capas.append(capa)
        
    def predecir(self, x):
        entrada_actual = x
        for capa in self.capas:
            entrada_actual = capa.feedforward(entrada_actual)
        return entrada_actual
        
    def entrenar(self, trainX, trainT, epocas=1000, tol=1e-5, tasa_aprendizaje=0.005, guardar_historial=True):
        self.historial_loss = []
        for epoca in range(epocas):
            prediccion = self.predecir(trainX)
            loss = np.mean(np.square(trainT - prediccion))
            
            if np.isnan(loss) or np.isinf(loss):
                break
                
            if guardar_historial and (epoca % 100 == 0 or epoca == epocas - 1):
                self.historial_loss.append((epoca, loss))
            
            if loss <= tol:
                break
                
            error = prediccion - trainT
            for capa in reversed(self.capas):
                error = capa.backpropagation(error, tasa_aprendizaje)

class Aptitud(ABC):
    @abstractmethod
    def evaluar_poblacion(self, poblacion: List[CromosomaNeuro], var: Dict = None):
        pass

class AptitudNeuro(Aptitud):
    def __init__(self, X_train, y_train, X_val, y_val, epocas_eval=800):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.epocas_eval = epocas_eval
        self.n_entradas = X_train.shape[1]
        self.n_salidas = y_train.shape[1]
        self.cache = {}

    def construir_red(self, params: dict) -> RedNeuronalModular:
        red = RedNeuronalModular()
        c1 = params["neuronas_c1"]
        c2 = params["neuronas_c2"]
        act = params["activacion"]
        bias = params["usa_bias"]

        red.agregar_capa(CapaModular(self.n_entradas, c1, activacion_nombre=act, usa_bias=bias))

        if c2 > 0:
            red.agregar_capa(CapaModular(c1, c2, activacion_nombre=act, usa_bias=bias))
            red.agregar_capa(CapaModular(c2, self.n_salidas, activacion_nombre='linear', usa_bias=bias))
        else:
            red.agregar_capa(CapaModular(c1, self.n_salidas, activacion_nombre='linear', usa_bias=bias))

        return red

    def evaluar_poblacion(self, poblacion: List[CromosomaNeuro], var: Dict = None):
        for cromosoma in poblacion:
            if cromosoma.bits in self.cache:
                cromosoma.score, cromosoma.mse_val, cromosoma.r2_val = self.cache[cromosoma.bits]
                continue

            params = cromosoma.decodificar()
            try:
                np.random.seed(42)
                red = self.construir_red(params)
                # Escalar la tasa de aprendizaje para que sea adecuada a regresión lineal
                lr = params["tasa_aprendizaje"] * 0.005
                red.entrenar(
                    self.X_train, 
                    self.y_train, 
                    epocas=self.epocas_eval, 
                    tasa_aprendizaje=lr,
                    guardar_historial=False
                )

                preds_val = red.predecir(self.X_val)
                if np.isnan(preds_val).any() or np.isinf(preds_val).any():
                    mse = 999.0
                    r2 = -999.0
                else:
                    mse = float(mean_squared_error(self.y_val, preds_val))
                    r2 = float(r2_score(self.y_val, preds_val))
            except Exception:
                mse = 999.0
                r2 = -999.0

            score = 1.0 / (1.0 + max(0.0, mse))
            cromosoma.score = score
            cromosoma.mse_val = mse
            cromosoma.r2_val = r2

            self.cache[cromosoma.bits] = (score, mse, r2)
