# Breast Cancer Wisconsin: modelo neuro-genético

## 1. Análisis exploratorio

### ¿De qué trata el dataset?

El dataset corresponde al **Breast Cancer Wisconsin Diagnostic Dataset**. Cada fila representa una muestra de tejido mamario y contiene mediciones numéricas calculadas a partir de imágenes digitalizadas de núcleos celulares.

El archivo contiene **569 observaciones**, **30 variables predictoras numéricas** y la variable objetivo `Diagnosis`.

### Problema que se busca resolver

Se busca clasificar cada muestra como:

- `0`: tumor benigno.
- `1`: tumor maligno.

Es un problema de **clasificación binaria supervisada**. La red neuronal aprende la relación entre las mediciones celulares y el diagnóstico.

### Variables

Las 30 variables se organizan en tres grupos de diez mediciones:

| Grupo | Variables |
|---|---|
| Media | `radius1`, `texture1`, `perimeter1`, `area1`, `smoothness1`, `compactness1`, `concavity1`, `concave_points1`, `symmetry1`, `fractal_dimension1` |
| Error estándar | `radius2`, `texture2`, `perimeter2`, `area2`, `smoothness2`, `compactness2`, `concavity2`, `concave_points2`, `symmetry2`, `fractal_dimension2` |
| Peor valor o valor extremo | `radius3`, `texture3`, `perimeter3`, `area3`, `smoothness3`, `compactness3`, `concavity3`, `concave_points3`, `symmetry3`, `fractal_dimension3` |

La variable `Diagnosis` es el objetivo. No se utiliza como entrada.

## 2. Arquitectura neuronal base

El modelo base se encuentra en [`1_Red_Inicial_Cancer.py`](./1_Red_Inicial_Cancer.py).

Arquitectura:

```text
30 entradas -> 16 neuronas sigmoid -> 1 salida sigmoid
```

Configuración:

- Estandarización de las entradas.
- División: 80 % entrenamiento y 20 % prueba.
- `1500` épocas.
- Tasa de aprendizaje: `0.5`.
- Bias activado.
- Función de pérdida: MSE.
- Umbral de clasificación: `0.5`.

Métricas calculadas sobre el conjunto de prueba:

- MSE.
- R².
- Accuracy.
- ROC-AUC.
- Correlación de Pearson.

La pérdida y las métricas se muestran en consola. También se genera:

```text
grafica_red_inicial_cancer.png
```

## 3. Optimización mediante algoritmo genético

El punto de entrada del modelo optimizado es [`2_Red_Optimizada_GA_Cancer.py`](./2_Red_Optimizada_GA_Cancer.py).

Este archivo ejecuta [`optimizar_cancer.py`](./optimizar_cancer.py), que:

1. Divide los datos en entrenamiento, validación interna del GA y prueba final.
2. Crea una población de 14 cromosomas.
3. Ejecuta 8 generaciones.
4. Aplica selección por torneo, cruce de un punto, mutación y elitismo.
5. Evalúa cada configuración mediante MSE, R² y accuracy.
6. Entrena la mejor configuración sobre todo el conjunto de entrenamiento.
7. Compara el modelo base contra el modelo neuro-genético en el mismo conjunto de prueba.

El cromosoma optimiza:

- Número de neuronas de la primera capa oculta.
- Número de neuronas de una segunda capa oculta, incluida la posibilidad de omitirla.
- Función de activación (`sigmoid` o `relu`).
- Tasa de aprendizaje.
- Uso de bias.

La implementación se encuentra en:

- [`CromosomaNeuro.py`](./CromosomaNeuro.py)
- [`GANeuro.py`](./GANeuro.py)
- [`interfaz/aptitud.py`](./interfaz/aptitud.py)
- [`interfaz/seleccion.py`](./interfaz/seleccion.py)
- [`interfaz/reproducion.py`](./interfaz/reproducion.py)
- [`interfaz/mutacion.py`](./interfaz/mutacion.py)

## 4. Comparación y resultados

Ejecutar:

```powershell
python 1_Red_Inicial_Cancer.py
python 2_Red_Optimizada_GA_Cancer.py
```

El segundo comando muestra una tabla con:

- Arquitectura base y optimizada.
- Tasa de aprendizaje.
- Activación.
- Uso de bias.
- MSE de prueba.
- R².
- Accuracy.
- ROC-AUC.
- Correlación.
- Porcentaje de mejora en MSE y R².

Archivos gráficos generados por el flujo neuro-genético:

```text
grafica_red_inicial_cancer.png
grafica_red_optimizada_cancer.png
comparacion_neuro_genetico_cancer.png
evolucion_fitness_ga_cancer.png
```

Para realizar una comparación válida, deben utilizarse las métricas del conjunto de prueba y no las métricas de validación usadas durante la selección genética.

