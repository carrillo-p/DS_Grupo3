import streamlit as st
import joblib
import pandas as pd
'''
from src.Modelos.xgboost_model import XGBoostModel

@st.cache_resource
def load_model():
    return XGBoostModel.load_model('src/Modelos/xgboost_model.joblib')

model = load_model()'''

def screen_predict():
    st.markdown("""<h1 style="text-align: center;">Predictor de Ictus</h1>""", unsafe_allow_html=True)
    st.markdown("""<h3 style="text-align: center;">Ingrese los datos del paciente para predecir el riesgo de ictus</h3>""", unsafe_allow_html=True)

    # Aquí debes agregar los campos de entrada correspondientes a tu dataset
    age = st.slider("Edad", 0, 100, 50)
    hypertension = st.selectbox("Hipertensión", ["No", "Sí"])
    heart_disease = st.selectbox("Enfermedad cardíaca", ["No", "Sí"])
    # Agrega más campos según sea necesario

    if st.button("Predecir Riesgo de Stroke"):
        # Preparar los inputs para el modelo
        inputs = pd.DataFrame({
            'age': [age],
            'hypertension': [1 if hypertension == "Sí" else 0],
            'heart_disease': [1 if heart_disease == "Sí" else 0],
            # Agrega más campos según sea necesario
        })
'''
        # Realizar predicción
        prediction = model.predict(inputs)
        probability = model.predict_proba(inputs)[0][1]

        st.write(f"Predicción: {'Alto riesgo de stroke' if prediction[0] == 1 else 'Bajo riesgo de stroke'}")
        st.write(f"Probabilidad de stroke: {probability:.2%}")'''