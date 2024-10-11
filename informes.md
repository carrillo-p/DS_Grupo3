# Resumen de la evaluación de los Modelos

# 1. XGBoost model

# Informe de Desempeño del Modelo

## 1. Exactitud del Modelo:
- **Conjunto de Entrenamiento**: 96.10%
- **Conjunto de Prueba**: 95.35%

## 2. Informe de Clasificación para el Conjunto de Prueba:
| Clase | Precisión | Recall | F1-Score | Soporte |
|-------|-----------|--------|----------|---------|
| 0     | 0.95      | 0.96   | 0.95     | 947     |
| 1     | 0.96      | 0.95   | 0.95     | 947     |
| **Exactitud Total** | **95.35%** |
| **Media Macro** | 0.95      | 0.95   | 0.95     | 1894   |
| **Media Ponderada** | 0.95      | 0.95   | 0.95     | 1894   |

## 3. Matriz de Confusión para el Conjunto de Prueba:
- **Clase 0** predicha correctamente: 906
- **Clase 1** predicha correctamente: 900
- **Falsos Positivos**: 41
- **Falsos Negativos**: 47

## 4. Puntuación ROC AUC:
- **ROC AUC Score**: 0.9871

## 5. Importancia de las Características:
| Característica               | Importancia |
|------------------------------|-------------|
| Edad (age_category)           | 0.450039    |
| Tipo de trabajo (work_type)   | 0.117619    |
| Índice de masa corporal (BMI) | 0.115807    |
| Estado de fumador (smoking_status) | 0.112054    |
| Nivel de glucosa (glucose_level_category) | 0.071738    |
| Enfermedad cardíaca (heart_disease) | 0.031439    |
| Hipertensión (hypertension)   | 0.029050    |
| Estado civil (ever_married)   | 0.027263    |
| Género                        | 0.024509    |
| Tipo de residencia (Residence_type) | 0.020481    |

## 6. Desempeño Promedio del Modelo:
- **Exactitud Promedio en Entrenamiento**: 96.18%
- **Exactitud Promedio en Prueba**: 95.62%
- **Media de Exactitud a lo largo de los pliegues**: 0.9492
---

# 2. Redes Neuronales (NN model)

## Ajuste de Hiperparámetros 
Se usó Optuna para realizar la búsqueda y ajuste de los valores de hiperparámetros con el objetivo de maximizar la métrica de rendimiento del modelo. 
El mejor modelo resultante de esta búsqueda de hiperparámetros alcanzó una precisión de **0.9176**. Este modelo utilizó:

- **n_layers**: 3 capas.
- **n_units**: 121 neuronas.
- **activation**: ReLU.
- **dropout_rate**: 0.0374.
- **batch_size**: 45.
- **epochs**: 50.

## Rendimiento del Modelo
El modelo alcanzó una **precisión máxima** de **91.18%** en el conjunto de prueba, con los siguientes detalles del informe de clasificación:

- **Precisión**: 
  - Clase 0: 0.93 
  - Clase 1: 0.89
- **Recall**: 
  - Clase 0: 0.89 
  - Clase 1: 0.93
- **F1-Score**: 
  - Clase 0: 0.91
  - Clase 1: 0.91

## Matriz de confusión:
La matriz de confusión para el conjunto de prueba es la siguiente:

|      | Predicho 0 | Predicho 1 |
|------|------------|------------|
| Real 0 |    842     |    104     |
| Real 1 |     63     |    885     |

# 3. Lazy Classifier

## Comparación de Modelos de Clasificación

| Modelo                             | Precisión | Recall | F1-Score | Exactitud |
|------------------------------------|-----------|--------|----------|-----------|
| **BaggingClassifier**              | 0.95      | 0.95   | 0.95     | 0.95      |
| **XGBClassifier**                  | 0.95      | 0.95   | 0.95     | 0.95      |
| **RandomForestClassifier**         | 0.95      | 0.95   | 0.95     | 0.95      |
| **LGBMClassifier**                 | 0.95      | 0.95   | 0.95     | 0.95      |
| **ExtraTreesClassifier**           | 0.94      | 0.94   | 0.94     | 0.94      |
| **DecisionTreeClassifier**         | 0.94      | 0.94   | 0.94     | 0.94      |
| **ExtraTreeClassifier**            | 0.93      | 0.93   | 0.93     | 0.93      |
| **KNeighborsClassifier**           | 0.88      | 0.88   | 0.88     | 0.88      |
| **AdaBoostClassifier**             | 0.88      | 0.88   | 0.88     | 0.88      |
| **LabelSpreading**                 | 0.84      | 0.84   | 0.84     | 0.84      |
| **LabelPropagation**               | 0.84      | 0.84   | 0.84     | 0.84      |
| **SVC**                            | 0.79      | 0.79   | 0.79     | 0.78      |
| **NuSVC**                          | 0.78      | 0.78   | 0.78     | 0.78      |
| **LogisticRegression**             | 0.75      | 0.75   | 0.75     | 0.75      |
| **CalibratedClassifierCV**         | 0.75      | 0.75   | 0.75     | 0.75      |
| **LinearSVC**                      | 0.75      | 0.75   | 0.75     | 0.75      |
| **LinearDiscriminantAnalysis**     | 0.74      | 0.74   | 0.74     | 0.74      |
| **RidgeClassifierCV**              | 0.74      | 0.74   | 0.74     | 0.74      |
| **SGDClassifier**                  | 0.74      | 0.74   | 0.74     | 0.74      |
| **BernoulliNB**                    | 0.72      | 0.72   | 0.72     | 0.72      |
| **NearestCentroid**                | 0.72      | 0.72   | 0.72     | 0.71      |
| **QuadraticDiscriminantAnalysis**  | 0.72      | 0.72   | 0.72     | 0.70      |
| **Perceptron**                     | 0.70      | 0.70   | 0.70     | 0.70      |
| **GaussianNB**                     | 0.69      | 0.69   | 0.69     | 0.68      |
| **PassiveAggressiveClassifier**    | 0.69      | 0.69   | 0.69     | 0.69      |
| **DummyClassifier**                | 0.50      | 0.50   | 0.50     | 0.33      |
