# Informe de Evaluación del Modelo

## Informe de Clasificación para el Conjunto de Prueba
|              | precisión | recall | f1-score | soporte |
|--------------|-----------|--------|----------|---------|
| **0**        | 0.95      | 0.95   | 0.95     | 758     |
| **1**        | 0.95      | 0.95   | 0.95     | 758     |
| **exactitud**|           |        | 0.95     | 1516    |
| **promedio macro**| 0.95 | 0.95   | 0.95     | 1516    |
| **promedio ponderado** | 0.95 | 0.95 | 0.95   | 1516    |

El informe de clasificación indica que el modelo alcanzó una precisión, recall y puntuación F1 de 0.95 para ambas clases, mostrando un rendimiento equilibrado.

## Matriz de Confusión para el Conjunto de Prueba
[[723 35] 
[ 37 721]]

## Puntuación ROC AUC
- **Puntuación ROC AUC:** 0.9890

La puntuación ROC AUC de 0.9890 indica un excelente rendimiento del modelo en la distinción entre las clases positivas y negativas.

## Importancia de las Características
La tabla a continuación muestra la importancia de cada característica utilizada en el modelo:

| Característica              | Importancia |
|-----------------------------|-------------|
| **age_category**            | 0.404878    |
| **smoking_status**          | 0.135618    |
| **work_type**               | 0.115947    |
| **bmi_category**            | 0.111983    |
| **glucose_level_category**  | 0.066196    |
| **ever_married**            | 0.041129    |
| **hypertension**            | 0.034961    |
| **heart_disease**           | 0.034566    |
| **Residence_type**          | 0.027449    |
| **gender**                  | 0.027273    |

La característica más influyente en el modelo es **age_category**, seguida por **smoking_status** y **work_type**.