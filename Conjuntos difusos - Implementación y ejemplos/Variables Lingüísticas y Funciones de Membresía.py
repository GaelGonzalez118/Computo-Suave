import numpy as np
import matplotlib.pyplot as plt

# Funciones de Membresia (MF)
def trimf(x, params):
    a, b, c = params
    return np.maximum(0, np.minimum((x - a)/(b - a + 1e-9), (c - x)/(c - b + 1e-9)))

def trapmf(x, params):
    a, b, c, d = params
    return np.maximum(0, np.minimum(np.minimum((x - a)/(b - a + 1e-9), 1), (d - x)/(d - c + 1e-9)))

def gaussmf(x, params):
    sigma, mean = params
    return np.exp(-0.5 * ((x - mean) / sigma) ** 2)

# --- 1. Variable: Temperatura ([0, 40] °C) ---
x_temp = np.linspace(0, 40, 400)
temp_fria = trapmf(x_temp, [0, 0, 10, 18])       # Trapezoidal
temp_tempt = trimf(x_temp, [12, 22, 30])         # Triangular
temp_caliente = gaussmf(x_temp, [4.0, 35])        # Gaussiana

# --- 2. Variable: Velocidad del Ventilador ([0, 100] %) ---
x_vel = np.linspace(0, 100, 400)
vel_baja = gaussmf(x_vel, [12.0, 0])             # Gaussiana
vel_media = trimf(x_vel, [25, 50, 75])           # Triangular
vel_alta = trapmf(x_vel, [60, 80, 100, 100])     # Trapezoidal

# --- 3. Variable: Presion ([0, 10] bar) ---
x_pres = np.linspace(0, 10, 400)
pres_baja = trapmf(x_pres, [0, 0, 2, 4])         # Trapezoidal
pres_normal = trimf(x_pres, [3, 5, 7])           # Triangular
pres_alta = gaussmf(x_pres, [1.2, 8.5])          # Gaussiana

# Graficacion
fig, axs = plt.subplots(3, 1, figsize=(9, 9))

axs[0].plot(x_temp, temp_fria, label='Fria (Trap)')
axs[0].plot(x_temp, temp_tempt, label='Templada (Tri)')
axs[0].plot(x_temp, temp_caliente, label='Caliente (Gauss)')
axs[0].set_title('Variable Linguistica 1: Temperatura (°C)')
axs[0].legend(); axs[0].grid(True)

axs[1].plot(x_vel, vel_baja, label='Baja (Gauss)')
axs[1].plot(x_vel, vel_media, label='Media (Tri)')
axs[1].plot(x_vel, vel_alta, label='Alta (Trap)')
axs[1].set_title('Variable Linguistica 2: Velocidad (%)')
axs[1].legend(); axs[1].grid(True)

axs[2].plot(x_pres, pres_baja, label='Baja (Trap)')
axs[2].plot(x_pres, pres_normal, label='Normal (Tri)')
axs[2].plot(x_pres, pres_alta, label='Alta (Gauss)')
axs[2].set_title('Variable Linguistica 3: Presion (bar)')
axs[2].legend(); axs[2].grid(True)

plt.tight_layout()
plt.show()