import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def screen_informe():
    st.markdown("""<h1 style="text-align: center;">Informe del Modelo</h1>""", unsafe_allow_html=True)

    # Aquí debes cargar y mostrar las métricas de tu modelo
    st.write("Métricas del modelo:")
    metrics = {
        "Precisión": 0.85,
        "Recall": 0.78,
        "F1-score": 0.81,
        "AUC-ROC": 0.89
    }
    st.table(pd.DataFrame(metrics, index=[0]))

    # Mostrar feature importances
    st.write("Importancia de las características:")
    feature_importance = {
        "age": 0.3,
        "hypertension": 0.2,
        "heart_disease": 0.15,
        # Agrega más características según tu modelo
    }
    fig, ax = plt.subplots()
    sns.barplot(x=list(feature_importance.values()), y=list(feature_importance.keys()), ax=ax)
    ax.set_title("Importancia de las Características")
    st.pyplot(fig)