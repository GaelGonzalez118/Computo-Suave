# Sleep Debt & Screen Time: modelo neuro-genético

## 1. Análisis exploratorio

### ¿De qué trata el dataset?

El dataset describe hábitos nocturnos relacionados con el uso del teléfono, características del sueño, consumo de cafeína y actividad física. Cada fila representa un usuario y una observación de sus hábitos y resultados de sueño.

El archivo contiene **8500 observaciones**. Tiene variables identificadoras, variables categóricas, variables numéricas y dos variables relacionadas con el resultado:

- `next_day_fatigue_score`: puntuación de fatiga del día siguiente.
- `sleep_debt_category`: categoría descriptiva de deuda de sueño.

### Problema que se busca resolver

Se busca predecir la puntuación de fatiga del día siguiente a partir de los hábitos y características fisiológicas del sueño.

Es un problema de **regresión supervisada**, porque `next_day_fatigue_score` es un valor numérico continuo.

### Variables del archivo

| Variable | Tipo | Descripción |
|---|---|---|
| `user_id` | Identificador | Identificador del usuario |
| `age` | Numérica | Edad |
| `gender` | Categórica | Género reportado |
| `occupation_type` | Categórica | Tipo de ocupación |
| `chronotype` | Categórica | Cronotipo |
| `bedtime_phone_minutes` | Numérica | Minutos de uso del teléfono antes de dormir |
| `primary_bedtime_app` | Categórica | Aplicación usada principalmente antes de dormir |
| `screen_brightness_pct` | Numérica | Brillo de pantalla en porcentaje |
| `blue_light_filter_active` | Binaria | Indica si el filtro de luz azul está activo |
| `caffeine_post_5pm_mg` | Numérica | Miligramos de cafeína después de las 17:00 |
| `physical_activity_min` | Numérica | Minutos de actividad física |
| `sleep_latency_min` | Numérica | Minutos necesarios para conciliar el sueño |
| `total_sleep_hours` | Numérica | Horas totales de sueño |
| `deep_sleep_pct` | Numérica | Porcentaje de sueño profundo |
| `rem_sleep_pct` | Numérica | Porcentaje de sueño REM |
| `morning_alarm_snoozes` | Numérica | Número de veces que se pospone la alarma |
| `next_day_fatigue_score` | Objetivo numérico | Puntuación de fatiga del día siguiente |
| `sleep_debt_category` | Categórica | Categoría de deuda de sueño |

El modelo utiliza deliberadamente ocho entradas numéricas:

```text
age
bedtime_phone_minutes
screen_brightness_pct
blue_light_filter_active
caffeine_post_5pm_mg
physical_activity_min
sleep_latency_min
total_sleep_hours
```

Las variables categóricas, `user_id`, `deep_sleep_pct`, `rem_sleep_pct`, `morning_alarm_snoozes` y `sleep_debt_category` no se utilizan en esta versión de la red. Esta decisión evita introducir codificación categórica no documentada y mantiene el experimento centrado en las ocho entradas seleccionadas.

## 2. Arquitectura neuronal base

El modelo base se encuentra en [`1_Red_Inicial_Sleep.py`](./1_Red_Inicial_Sleep.py).

Arquitectura:

```text
8 entradas -> 12 neuronas sigmoid -> 1 salida lineal
```

Configuración:

- Selección de las ocho variables numéricas indicadas.
- Eliminación de filas con valores faltantes en esas variables.
- Estandarización de las entradas.
- División: 80 % entrenamiento y 20 % prueba.
- `2000` épocas.
- Tasa de aprendizaje: `0.005`.
- Bias activado.
- Función de pérdida: MSE.
- Activación lineal en la salida por tratarse de regresión.

Métricas calculadas sobre el conjunto de prueba:

- MSE.
- RMSE.
- MAE.
- R².
- Correlación de Pearson.

También se genera:

```text
grafica_red_inicial_sleep.png
```

## 3. Optimización mediante algoritmo genético

El punto de entrada del modelo optimizado es [`2_Red_Optimizada_GA_Sleep.py`](./2_Red_Optimizada_GA_Sleep.py).

Este archivo ejecuta [`optimizar_sleep_debt.py`](./optimizar_sleep_debt.py), que:

1. Limpia y estandariza las variables seleccionadas.
2. Divide los datos en entrenamiento, validación interna del GA y prueba final.
3. Crea una población de 14 cromosomas.
4. Ejecuta 8 generaciones.
5. Aplica selección por torneo, cruce de un punto, mutación y elitismo.
6. Evalúa las configuraciones usando el MSE de validación.
7. Entrena la mejor configuración sobre todo el conjunto de entrenamiento.
8. Compara el modelo base contra el modelo neuro-genético en el mismo conjunto de prueba.

El cromosoma optimiza:

- Número de neuronas de la primera capa oculta.
- Número de neuronas de una segunda capa oculta, incluida la posibilidad de omitirla.
- Función de activación (`sigmoid` o `relu`).
- Tasa de aprendizaje.
- Uso de bias.

Para este problema de regresión, la tasa codificada por el cromosoma se escala por `0.005` durante la evaluación y el entrenamiento final para mantener una magnitud estable.

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
python 1_Red_Inicial_Sleep.py
python 2_Red_Optimizada_GA_Sleep.py
```

El segundo comando muestra una tabla con:

- Arquitectura base y optimizada.
- Tasa de aprendizaje.
- Activación.
- Uso de bias.
- MSE de prueba.
- RMSE.
- MAE.
- R².
- Correlación.
- Porcentaje de mejora en MSE y R².

Archivos gráficos generados por el flujo neuro-genético:

```text
grafica_red_inicial_sleep.png
grafica_red_optimizada_sleep.png
comparacion_neuro_genetico_sleep.png
evolucion_fitness_ga_sleep.png
```

La comparación final debe hacerse con las métricas del conjunto de prueba. La validación utilizada por el algoritmo genético solo sirve para seleccionar hiperparámetros y no debe presentarse como el resultado final del modelo.
