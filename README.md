# XGBoost model
# Resumen de la Evaluación del Modelo

## Validación Cruzada y Ajuste de Hiperparámetros
La evaluación del modelo empleó **validación cruzada Stratificada K-Fold** con **5 pliegues** para garantizar que la distribución de las clases objetivo se mantuviera en los conjuntos de entrenamiento y prueba. Esta técnica mejora la fiabilidad del modelo al proporcionar una evaluación más robusta de su rendimiento.

Para optimizar los hiperparámetros del modelo, se utilizó **Optuna**, lo que permitió realizar una búsqueda eficiente y automatizada de los mejores parámetros. Los hiperparámetros óptimos descubiertos fueron:

- **n_estimators**: 178
- **learning_rate**: 0.0554
- **max_depth**: 10
- **min_child_weight**: 2
- **subsample**: 0.9307
- **colsample_bytree**: 0.7241

## Rendimiento del Modelo
El modelo alcanzó una **precisión máxima** de **92.29%** en el conjunto de prueba, con los siguientes detalles del informe de clasificación:

- **Precisión**: 
  - Clase 0: 0.95 
  - Clase 1: 0.90
- **Recall**: 
  - Clase 0: 0.89 
  - Clase 1: 0.95
- **F1-Score**: 
  - Clase 0: 0.92 
  - Clase 1: 0.92

## Matriz de Confusión
La matriz de confusión para el conjunto de prueba es la siguiente:

[[836 111]
 [ 43 903]]
 
## Puntuación ROC AUC
El modelo mostró una **puntuación ROC AUC** de **0.9736**, lo que indica una excelente capacidad de discriminación entre las clases.

## Importancia de las Características
El análisis de la importancia de las características reveló que las más significativas incluían:

- **ever_married_Yes**: 0.3466
- **age_category_Adults**: 0.1534
- **bmi_category_Underweight**: 0.0955

Este enfoque exhaustivo demuestra la efectividad del modelo en la predicción de la variable objetivo y su potencial para aplicaciones prácticas en campos relevantes.
