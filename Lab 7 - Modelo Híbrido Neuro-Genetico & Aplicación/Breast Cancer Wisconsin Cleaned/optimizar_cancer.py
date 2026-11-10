import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, roc_auc_score

from CromosomaNeuro import CromosomaNeuro
from GANeuro import GANeuro
from interfaz import (
    AptitudNeuro,
    SeleccionTorneo,
    ReproduccionUnCruce,
    MutacionNeuro,
    CapaModular,
    RedNeuronalModular,
)

def ejecutar_optimizacion_cancer():
    print("=" * 70)
    print(" LABORATORIO 7: MODELO HÍBRIDO NEURO-GENÉTICO")
    print(" Dataset 2: Breast Cancer Wisconsin Diagnostic")
    print("=" * 70)

    csv_path = 'breast_cancer_cleaned.csv'
    if not os.path.exists(csv_path):
        csv_path = os.path.join('Breast Cancer Wisconsin Cleaned', csv_path)

    print(f"\n[1] Cargando datos desde '{csv_path}'...")
    df = pd.read_csv(csv_path)

    X_raw = df.drop(columns=['Diagnosis']).values
    y = df[['Diagnosis']].values

    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_raw)

    # Partición 80% Train / 20% Test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Validación interna para el GA (20% del conjunto de entrenamiento = 91 muestras)
    X_ga_train, X_ga_val, y_ga_train, y_ga_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    print(f"    - Muestras Totales:              {len(df)}")
    print(f"    - Características de Entrada:    {X_raw.shape[1]} (mediciones nucleares continuas)")
    print(f"    - Muestras Entrenamiento Total: {len(X_train)}")
    print(f"    - Muestras Entrenamiento GA:     {len(X_ga_train)}")
    print(f"    - Muestras Validación GA:        {len(X_ga_val)}")
    print(f"    - Muestras Prueba Final (20%):   {len(X_test)}")

    # 2. Configuración y ejecución del Algoritmo Genético
    print("\n[2] Inicializando Algoritmo Genético para optimización de hiperparámetros...")
    aptitud_fn = AptitudNeuro(X_ga_train, y_ga_train, X_ga_val, y_ga_val, epocas_eval=600)

    gaparams = {
        "Aptitud": aptitud_fn,
        "Seleccion": SeleccionTorneo(k=3),
        "Reproduccion": ReproduccionUnCruce(num_genes=5),
        "Mutacion": MutacionNeuro(probabilidad=0.15),
        "por-remplazo": 0.25
    }

    tam_poblacion = 14
    num_generaciones = 8
    num_genes = 5

    ga = GANeuro(num_inv=tam_poblacion, num_genes=num_genes, hyperparams=gaparams)
    print(f"    - Población: {tam_poblacion} cromosomas | Generaciones: {num_generaciones}\n")

    mejor_cromosoma = ga.buscar_solucion(max_generaciones=num_generaciones)
    mejores_params = mejor_cromosoma.decodificar()

    print("\n" + "=" * 70)
    print(" RESULTADO DE LA OPTIMIZACIÓN GENÉTICA")
    print("=" * 70)
    print(f"Cromosoma Óptimo: {mejor_cromosoma.a_cadena()}")
    for k, v in mejores_params.items():
        print(f"  * {k}: {v}")
    print(f"Fitness alcanzado: {mejor_cromosoma.score:.4f}")
    print(f"MSE en Validación GA: {mejor_cromosoma.mse_val:.6f}")
    print(f"R^2 en Validación GA: {mejor_cromosoma.r2_val:.4f}")
    print(f"Exactitud (Accuracy) en Validación GA: {mejor_cromosoma.acc_val*100:.2f}%")

    # 3. Entrenamiento completo: Modelo Base vs Modelo Neuro-Genético Óptimo
    print("\n[3] Entrenando Modelo Base y Modelo Neuro-Genético sobre todo el conjunto de Train (455 muestras, 1,500 épocas)...")

    # --- Modelo Base (30 -> [16] -> 1, sigmoid, lr=0.5) ---
    print("\n -> Entrenando Modelo Base (30 -> [16] -> 1, sigmoid, lr=0.5)...")
    np.random.seed(42)
    red_base = RedNeuronalModular()
    red_base.agregar_capa(CapaModular(n_entradas=X_train.shape[1], n_neuronas=16, activacion_nombre='sigmoid', usa_bias=True))
    red_base.agregar_capa(CapaModular(n_entradas=16, n_neuronas=1, activacion_nombre='sigmoid', usa_bias=True))
    red_base.entrenar(X_train, y_train, epocas=1500, tasa_aprendizaje=0.5, guardar_historial=True)

    y_pred_base = red_base.predecir(X_test)
    y_pred_base_bin = (y_pred_base >= 0.5).astype(int)
    mse_base = mean_squared_error(y_test, y_pred_base)
    r2_base = r2_score(y_test, y_pred_base)
    acc_base = accuracy_score(y_test, y_pred_base_bin)
    auc_base = roc_auc_score(y_test, y_pred_base)
    r_base = np.corrcoef(y_test.flatten(), y_pred_base.flatten())[0, 1]

    # --- Modelo Neuro-Genético Óptimo ---
    print(f"\n -> Entrenando Modelo Neuro-Genético ({mejor_cromosoma.a_cadena()})...")
    np.random.seed(42)
    red_optima = aptitud_fn.construir_red(mejores_params)
    red_optima.entrenar(X_train, y_train, epocas=1500, tasa_aprendizaje=mejores_params["tasa_aprendizaje"], guardar_historial=True)

    y_pred_optimo = red_optima.predecir(X_test)
    y_pred_optimo_bin = (y_pred_optimo >= 0.5).astype(int)
    mse_optimo = mean_squared_error(y_test, y_pred_optimo)
    r2_optimo = r2_score(y_test, y_pred_optimo)
    acc_optimo = accuracy_score(y_test, y_pred_optimo_bin)
    auc_optimo = roc_auc_score(y_test, y_pred_optimo)
    r_optimo = np.corrcoef(y_test.flatten(), y_pred_optimo.flatten())[0, 1]

    # 4. Comparación Final en Test (114 muestras)
    print("\n" + "=" * 70)
    print(" COMPARACIÓN FINAL EN CONJUNTO DE PRUEBA (20% - 114 MUESTRAS)")
    print("=" * 70)
    tabla_comparativa = pd.DataFrame({
        "Métrica": [
            "Arquitectura",
            "Tasa de Aprendizaje",
            "Función Activación",
            "Usa Bias",
            "MSE Test (Pérdida)",
            "Coeficiente R^2",
            "Exactitud (Accuracy)",
            "Área bajo la curva ROC (AUC)",
            "Correlación de Pearson (R)"
        ],
        "Modelo Base": [
            "30 -> [16] -> 1",
            "0.5",
            "sigmoid",
            "True",
            f"{mse_base:.6f}",
            f"{r2_base:.4f}",
            f"{acc_base*100:.2f}%",
            f"{auc_base:.4f}",
            f"{r_base:.4f}"
        ],
        "Modelo Neuro-Genético": [
            f"30 -> [{mejores_params['neuronas_c1']}" + (f", {mejores_params['neuronas_c2']}]" if mejores_params['neuronas_c2'] > 0 else "]") + " -> 1",
            f"{mejores_params['tasa_aprendizaje']}",
            f"{mejores_params['activacion']}",
            f"{mejores_params['usa_bias']}",
            f"{mse_optimo:.6f}",
            f"{r2_optimo:.4f}",
            f"{acc_optimo*100:.2f}%",
            f"{auc_optimo:.4f}",
            f"{r_optimo:.4f}"
        ]
    })
    print(tabla_comparativa.to_string(index=False))

    mejora_mse = ((mse_base - mse_optimo) / mse_base) * 100
    mejora_r2 = ((r2_optimo - r2_base) / abs(r2_base)) * 100
    print(f"\nMejora en MSE: {mejora_mse:+.2f}%")
    print(f"Mejora en R^2: {mejora_r2:+.2f}%")

    print("\n--- Muestra de Predicciones vs Diagnóstico Real (Primeros 10 casos) ---")
    comparacion_muestras = pd.DataFrame({
        'Real': y_test.flatten()[:10],
        'Pred. Base': y_pred_base.flatten()[:10],
        'Pred. Neuro-GA': y_pred_optimo.flatten()[:10],
        'Clase Base': y_pred_base_bin.flatten()[:10],
        'Clase Neuro-GA': y_pred_optimo_bin.flatten()[:10]
    })
    print(comparacion_muestras.to_string(index=False))

    # 5. Generación de Gráficas Comparativas
    print("\n[4] Generando gráficas comparativas...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    epocas_base = [p[0] for p in red_base.historial_loss]
    loss_base = [p[1] for p in red_base.historial_loss]
    epocas_opt = [p[0] for p in red_optima.historial_loss]
    loss_opt = [p[1] for p in red_optima.historial_loss]

    axes[0].plot(epocas_base, loss_base, label=f"Modelo Base (MSE Final: {mse_base:.4f})", color="coral", lw=2)
    axes[0].plot(epocas_opt, loss_opt, label=f"Neuro-Genético (MSE Final: {mse_optimo:.4f})", color="teal", lw=2)
    axes[0].set_title("Evolución de Pérdida (MSE) - Breast Cancer", fontsize=12)
    axes[0].set_xlabel("Épocas", fontsize=10)
    axes[0].set_ylabel("Loss (MSE)", fontsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend()

    axes[1].scatter(range(len(y_test)), y_pred_optimo, c=y_test.flatten(), cmap='coolwarm', alpha=0.8, edgecolor='k', s=35)
    axes[1].axhline(0.5, color='black', linestyle='--', label='Umbral (0.5)')
    axes[1].set_title(f"Diagnóstico Neuro-GA (Acc: {acc_optimo*100:.1f}%, R²: {r2_optimo:.3f})", fontsize=12)
    axes[1].set_xlabel("Índice de Muestra", fontsize=10)
    axes[1].set_ylabel("Probabilidad P(Maligno)", fontsize=10)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend()

    plt.tight_layout()
    grafica_path = "comparacion_neuro_genetico_cancer.png"
    plt.savefig(grafica_path, dpi=300)
    print(f" -> Gráfica guardada exitosamente en '{grafica_path}'.")

    plt.figure(figsize=(8, 5))
    generaciones = list(range(1, len(ga.historial_mejor_score) + 1))
    plt.plot(generaciones, ga.historial_mejor_score, marker='o', color='forestgreen', lw=2, label="Mejor Fitness")
    plt.plot(generaciones, ga.historial_avg_score, marker='s', color='orange', linestyle='--', label="Fitness Promedio")
    plt.title("Evolución del Fitness en GA - Breast Cancer Wisconsin", fontsize=12)
    plt.xlabel("Generación", fontsize=10)
    plt.ylabel("Aptitud: 1/(1+MSE) + Accuracy", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    fitness_grafica_path = "evolucion_fitness_ga_cancer.png"
    plt.savefig(fitness_grafica_path, dpi=300)
    print(f" -> Gráfica de evolución de fitness guardada en '{fitness_grafica_path}'.")

    print("\nProceso Neuro-Genético completado con éxito para Breast Cancer.")

if __name__ == "__main__":
    ejecutar_optimizacion_cancer()
