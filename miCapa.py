import numpy as np

# Funciones de activacion
def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

# Estructura de la Capa
class Capa:
    def __init__(self, n_entradas, n_neuronas, activacion_nombre='relu', usa_bias=True):
        self.n_neuronas = n_neuronas
        self.usa_bias = usa_bias
        
        # Funcion de activacion y derivada
        if activacion_nombre.lower() == 'relu':
            self.activacion = relu
        elif activacion_nombre.lower() == 'sigmoid':
            self.activacion = sigmoid
        elif activacion_nombre.lower() == 'softmax':
            self.activacion = softmax
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
