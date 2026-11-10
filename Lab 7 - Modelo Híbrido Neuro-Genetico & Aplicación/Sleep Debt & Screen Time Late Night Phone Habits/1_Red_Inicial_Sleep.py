import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# --- Funciones de Activación ---
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

# --- Estructura de la Capa ---
class Capa:
    def __init__(self, n_entradas, n_neuronas, activacion_nombre='sigmoid', usa_bias=True):
        self.n_neuronas = n_neuronas
        self.usa_bias = usa_bias
        self.activacion_nombre = activacion_nombre.lower()

        if self.activacion_nombre == 'sigmoid':
            self.activacion = sigmoid
            self.derivada = sigmoid_derivada
        elif self.activacion_nombre == 'linear':
            self.activacion = linear
            self.derivada = linear_derivada
        else:
            raise ValueError(f"Activacion '{activacion_nombre}' no soportada")

        if self.usa_bias:
            self.pesos = np.random.randn(n_entradas + 1, n_neuronas) * 0.1
        else:
            self.pesos = np.random.randn(n_entradas, n_neuronas) * 0.1

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

    def backpropagation(self, error_salida, tasa_aprendizaje=0.005):
        delta = error_salida * self.derivada(self.z)
        gradientes_pesos = np.dot(self.entrada.T, delta)
        error_hacia_atras = np.dot(delta, self.pesos.T)
        
        if self.usa_bias:
            error_hacia_atras = error_hacia_atras[:, :-1]
            
        self.pesos -= tasa_aprendizaje * (gradientes_pesos / self.entrada.shape[0])
        return error_hacia_atras

# --- Red Neuronal Inicial ---
class RedNeuronal:
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
        
    def entrenar(self, trainX, trainT, epocas=2000, tol=1e-5, tasa_aprendizaje=0.005, mostrar_cada=200):
        self.historial_loss = []
        for epoca in range(epocas):
            prediccion = self.predecir(trainX)
            loss = np.mean(np.square(trainT - prediccion))
            
            if epoca % mostrar_cada == 0 or epoca == epocas - 1:
                print(f"Época {epoca:04d} | Error (Loss MSE): {loss:.6f}")
                self.historial_loss.append((epoca, loss))
            
            if loss <= tol:
                print(f"Convergencia alcanzada en época {epoca} con Loss: {loss:.6f}")
                break
                
            error = prediccion - trainT
            for capa in reversed(self.capas):
                error = capa.backpropagation(error, tasa_aprendizaje)

# --- EJECUCIÓN: RED NEURONAL INICIAL (PROPUESTA BASE) ---
if __name__ == "__main__":
    print("=" * 70)
    print(" 1. RED NEURONAL INICIAL (PROPUESTA BASE) - SLEEP DEBT")
    print("=" * 70)
    
    df = pd.read_csv('bedtime_screentime_sleep_debt.csv')
    
    # 8 Variables completas (conductuales y fisiológicas del sueño)
    features = [
        'age', 
        'bedtime_phone_minutes', 
        'screen_brightness_pct', 
        'blue_light_filter_active', 
        'caffeine_post_5pm_mg', 
        'physical_activity_min',
        'sleep_latency_min',
        'total_sleep_hours'
    ]
    target = ['next_day_fatigue_score']
    
    df_clean = df[features + target].dropna()
    X = df_clean[features].values
    y = df_clean[target].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Partición 80% Entrenamiento / 20% Prueba
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    print(f"Datos: {len(X_train)} entrenamiento, {len(X_test)} prueba (20%)")
    print("Arquitectura Base Propuesta: 8 entradas -> [12 neuronas sigmoid] -> 1 salida lineal")
    
    np.random.seed(42)
    red_inicial = RedNeuronal()
    red_inicial.agregar_capa(Capa(n_entradas=8, n_neuronas=12, activacion_nombre='sigmoid', usa_bias=True))
    red_inicial.agregar_capa(Capa(n_entradas=12, n_neuronas=1, activacion_nombre='linear', usa_bias=True))
    
    print("\nIniciando entrenamiento...")
    red_inicial.entrenar(X_train, y_train, epocas=2000, tasa_aprendizaje=0.005, mostrar_cada=200)
    
    # Evaluación en datos de prueba
    y_pred = red_inicial.predecir(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    r = np.corrcoef(y_test.flatten(), y_pred.flatten())[0, 1]
    
    print("\n" + "=" * 50)
    print(" RESULTADOS EN DATOS DE PRUEBA (RED INICIAL)")
    print("=" * 50)
    print(f"Pérdida (MSE):                     {mse:.4f}")
    print(f"Raíz del Error Cuadrático (RMSE):   {rmse:.4f}")
    print(f"Error Absoluto Medio (MAE):         {mae:.4f}")
    print(f"Coeficiente de Determinación (R^2): {r2:.4f}")
    print(f"Coeficiente de Correlación (R):     {r:.4f}")
    
    print("\n--- Muestra de Predicciones vs Valores Reales (Primeros 10) ---")
    muestra = pd.DataFrame({
        'Real': y_test.flatten()[:10],
        'Predicción': y_pred.flatten()[:10],
        'Error Absoluto': np.abs(y_test.flatten()[:10] - y_pred.flatten()[:10])
    })
    print(muestra.to_string(index=False))
    
    # Gráfica
    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, y_pred, alpha=0.4, color='coral', edgecolor='k', label=f'R² = {r2:.4f}')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Ideal')
    plt.title('Red Inicial: Valores Reales vs Predicciones (Sleep Debt)')
    plt.xlabel('Fatiga Real')
    plt.ylabel('Fatiga Predicha')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('grafica_red_inicial_sleep.png', dpi=300)
    print("\nGráfica guardada en 'grafica_red_inicial_sleep.png'.")
