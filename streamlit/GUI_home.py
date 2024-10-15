import streamlit as st
from BBDD.create_database import create_database_and_table  # Importa la función de creación de la base de datos
from BBDD.database_utils import save_prediction_to_db  # Importa la función para guardar predicciones en la base de datos


# Crear la base de datos y la tabla si no existen
create_database_and_table()


def home_screen():
    st.markdown("""<h1 style="text-align: center;">Bienvenido al Predictor de Ictus</h1>""", unsafe_allow_html=True)
    st.markdown("""
    ¡Hola! Bienvenido a nuestra aplicación de predicción de ictus del Hospital F5. 
    Aquí podrás:
    
    - 🥼 Predecir el riesgo de ictus basado en diferentes factores
    - 📊 Ver los resultados detallados de nuestro modelo de predicción
    
    ¡Explora las diferentes secciones!
    """)