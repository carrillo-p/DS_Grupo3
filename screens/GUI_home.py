import streamlit as st

def home_screen():
    st.markdown("""<h1 style="text-align: center;">Bienvenido al Predictor de Ictus</h1>""", unsafe_allow_html=True)
    st.markdown("""
    ¡Hola! Bienvenido a nuestra aplicación de predicción de ictus del Hospital F5. 
    Aquí podrás:
    
    - 🥼 Predecir el riesgo de ictus basado en diferentes factores
    - 📊 Ver los resultados detallados de nuestro modelo de predicción
    
    ¡Explora las diferentes secciones!
    """)