import numpy as np

# Funciones de activacion
def relu(x):
    return np.maximum(0, x)

def relu_derivada(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivada(x):
    s = sigmoid(x)
    return s * (1 - s)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def softmax_derivada(x):
    s = softmax(x)
    return s * (1 - s)

# Estructura de la Capa
class Capa:
    def __init__(self, n_entradas, n_neuronas, activacion_nombre='relu', usa_bias=True):
        self.n_neuronas = n_neuronas
        self.usa_bias = usa_bias
        
        # Funcion de activacion y derivada
        activacion_nombre = activacion_nombre.lower()
        if activacion_nombre == 'relu':
            self.activacion = relu
            self.derivada = relu_derivada
        elif activacion_nombre == 'sigmoid':
            self.activacion = sigmoid
            self.derivada = sigmoid_derivada
        elif activacion_nombre == 'softmax':
            self.activacion = softmax
            self.derivada = softmax_derivada
        else:
            raise ValueError("Activacion no soportada")

        # Matriz de pesos (W) con valores aleatorios
        if self.usa_bias:
            # W(entradas+1 x Neuronas)
            self.pesos = np.random.randn(n_entradas + 1, n_neuronas) * 0.1
        else:
            # W(entradas x Neuronas)
            self.pesos = np.random.randn(n_entradas, n_neuronas) * 0.1

        self.entrada = None
        self.z = None

    def feedforward(self, x):
        # Con bias
        if self.usa_bias:
            unos = np.ones((x.shape[0], 1))
            self.entrada = np.hstack((x, unos))
        else:
            self.entrada = x

        # Producto punto: Z = X * W
        self.z = np.dot(self.entrada, self.pesos)
        
        # Aplicar funcion de activacion
        salida = self.activacion(self.z)
        return salida

    def backpropagation(self, error_salida, tasa_aprendizaje=0.01):
        # Calcular el gradiente de la capa
        delta = error_salida * self.derivada(self.z)
        
        # Calcular los gradientes
        gradientes_pesos = np.dot(self.entrada.T, delta)
        
        # Calcular el error para la capa anterior
        error_hacia_atras = np.dot(delta, self.pesos.T)
        
        if self.usa_bias:
            error_hacia_atras = error_hacia_atras[:, :-1]
            
        # Actualizar los pesos de la capa
        self.pesos -= tasa_aprendizaje * gradientes_pesos
        
        return error_hacia_atras

class RedNeuronal:
    def __init__(self):
        self.capas = []
        
    def agregar_capa(self, capa):
        self.capas.append(capa)
        
    def predecir(self, x):
        # FEEDFORWARD
        entrada_actual = x
        for capa in self.capas:
            entrada_actual = capa.feedforward(entrada_actual)
        return entrada_actual
        
    def entrenar(self, trainX, trainT, epocas=1000, tol=1e-5, tasa_aprendizaje=0.01):
        for epoca in range(epocas):
            # YH = feedforward (trainX)
            prediccion = self.predecir(trainX)
            
            # loss = Funcion Costo (T, YH) - Usando MSE
            loss = np.mean(np.square(trainT - prediccion))
            
            if loss <= tol:
                print(f"Se cumplio la tolerancia en la epoca {epoca}")
                break
                
            # Calcular error base para iniciar Backpropagation
            error = prediccion - trainT
            
            # Actualizar pesos con backpropagation
            for capa in reversed(self.capas):
                error = capa.backpropagation(error, tasa_aprendizaje)
