import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from ..src.model.train_models_XGBoost import XGBoostStrokeModel
import joblib
import numpy as np
import os

def load_model_and_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, '..', 'src', 'model', 'xgboost_model.joblib')
    scaler_path = os.path.join(current_dir, '..', 'src', 'model', 'xgb_scaler.joblib')
    
    model = XGBoostStrokeModel.load_model(model_path, scaler_path)
    
    X_test = joblib.load(os.path.join(current_dir, '..', 'src', 'data', 'X_test.joblib'))
    y_test = joblib.load(os.path.join(current_dir, '..', 'src', 'data', 'y_test.joblib'))
    
    return model, X_test, y_test

def screen_informe():
    st.markdown("""<h1 style="text-align: center;">Informe del Modelo de Predicción de Ictus</h1>""", unsafe_allow_html=True)

    model, X_test, y_test = load_model_and_data()

    # Realizar predicciones
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Métricas del modelo
    st.subheader("Métricas del Modelo")
    report = classification_report(y_test, y_pred, output_dict=True)
    metrics = {
        "Precisión": report['1']['precision'],
        "Recall": report['1']['recall'],
        "F1-score": report['1']['f1-score'],
        "Accuracy": report['accuracy']
    }
    st.table(pd.DataFrame(metrics, index=[0]))

    # Matriz de confusión
    st.subheader("Matriz de Confusión")
    conf_matrix = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicción')
    ax.set_ylabel('Valor Real')
    ax.set_title('Matriz de Confusión')
    st.pyplot(fig)

    # Curva ROC
    st.subheader("Curva ROC")
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Tasa de Falsos Positivos')
    ax.set_ylabel('Tasa de Verdaderos Positivos')
    ax.set_title('Curva ROC')
    ax.legend(loc="lower right")
    st.pyplot(fig)

    # Importancia de las características
    st.subheader("Importancia de las Características")
    feature_importance = model.model.feature_importances_
    feature_names = model.feature_names
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
    feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
    st.subheader("Importancia de las Características")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, ax=ax)
    ax.set_title("Importancia de las Características")
    st.pyplot(fig)

    # Distribución de probabilidades predichas
    st.subheader("Distribución de Probabilidades Predichas")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(y_prob, kde=True, ax=ax)
    ax.set_xlabel('Probabilidad de Ictus')
    ax.set_ylabel('Frecuencia')
    ax.set_title('Distribución de Probabilidades Predichas')
    st.pyplot(fig)

if __name__ == "__main__":
    screen_informe()