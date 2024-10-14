import streamlit as st
import joblib
import pandas as pd
import numpy as np
from src.model.train_models_XGBoost import XGBoostStrokeModel
from screens.aux_functions import create_gauge_chart
from database_utils import save_prediction_to_db 
import tensorflow as tf

@st.cache_resource
def load_model():
    return XGBoostStrokeModel.load_model('src/model/xgboost_model.joblib', 'src/model/xgb_scaler.joblib')
def load_nn_model():
    model = tf.keras.models.load_model('src/model/nn_stroke.keras')
    scaler = joblib.load('src/model/nn_scaler.joblib')
    return model, scaler

xgb_model = load_model()
nn_model, nn_scaler = load_nn_model()

def screen_predict():
    st.markdown("""<h1 style="text-align: center;">Predictor de Ictus</h1>""", unsafe_allow_html=True)
    st.markdown("""<h3 style="text-align: center;">Ingrese los datos del paciente para predecir el riesgo de ictus</h3>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Género", ["Masculino", "Femenino"])
        age = st.slider("Edad", 0, 100, 50)
        hypertension = st.selectbox("Hipertensión", ["No", "Sí"])
        heart_disease = st.selectbox("Enfermedad cardíaca", ["No", "Sí"])
        ever_married = st.selectbox("Alguna vez casado", ["No", "Sí"])

    with col2:
        work_type = st.selectbox("Tipo de trabajo", ["Privado", "Autónomo", "Gubernamental", "Niño", "Nunca ha trabajado"])
        residence_type = st.selectbox("Tipo de residencia", ["Urbana", "Rural"])
        avg_glucose_level = st.slider("Nivel promedio de glucosa", 50.0, 300.0, 100.0)
        bmi = st.slider("IMC (Índice de Masa Corporal)", 10.0, 50.0, 25.0)
        smoking_status = st.selectbox("Estado de fumador", ["Nunca fumó", "Fumador", "Exfumador"])

    if st.button("Predecir Riesgo de Ictus"):
        # Preparar los inputs para el modelo
        inputs = pd.DataFrame({
            'gender': [1 if gender == "Masculino" else 0],
            'hypertension': [1 if hypertension == "Sí" else 0],
            'heart_disease': [1 if heart_disease == "Sí" else 0],
            'ever_married': [1 if ever_married == "Sí" else 0],
            'work_type': [0 if work_type == "Privado" else 1 if work_type == "Autónomo" else 2 if work_type == "Gubernamental" else 3 if work_type == "Niño" else 4],
            'Residence_type': [1 if residence_type == "Urbana" else 0],
            'smoking_status': [0 if smoking_status == "Nunca fumó" else 1 if smoking_status == "Exfumador" else 2],
            'bmi_category': [0 if bmi < 18.5 else 1 if bmi < 25 else 2 if bmi < 30 else 3],
            'age_category': [0 if age < 13 else 1 if age < 18 else 2 if age < 60 else 3],
            'glucose_level_category': [0 if avg_glucose_level < 100 else 1 if avg_glucose_level < 140 else 2]
        })

        print("Datos enviados al modelo:")
        print(inputs)

        # Realizar predicción
        xgb_probabilities = xgb_model.predict_proba(inputs)[0][1]
        inputs_nn = nn_scaler.transform(inputs)
        nn_probabilities = nn_model.predict(inputs_nn)
        nn_probability = nn_probabilities[0]
        final_probabilities = 0.6 * xgb_probabilities + 0.4 * nn_probability
        final_prediction = 1 if final_probabilities >= 0.5 else 0

         # Guardar la predicción en la base de datos
        save_prediction_to_db(inputs)

        # Mostrar resultados
        st.subheader("Resultados de la Predicción")
        col1, col2 = st.columns(2)

        with col1:
            final_probabilities = float(final_probabilities)
            fig = create_gauge_chart(final_probabilities * 100, "Probabilidad de Ictus")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.metric("Predicción", "Alto riesgo de ictus" if final_prediction == 1 else "Bajo riesgo de ictus")
            st.write(f"Probabilidad de ictus: {final_probabilities:.2%}")

        # Añadir recomendaciones basadas en el riesgo
        st.subheader("Recomendaciones")

        if final_prediction == 1:
            st.warning("Se recomienda consultar a un médico para una evaluación más detallada.")
        else:
            st.success("Mantener un estilo de vida saludable para prevenir riesgos futuros.")

        # Factores de riesgo identificados
        st.subheader("Factores de riesgo identificados")
        risk_factors = []
        if inputs['hypertension'].values[0] == 1:
            risk_factors.append("Hipertensión")
        if inputs['heart_disease'].values[0] == 1:
            risk_factors.append("Enfermedad cardíaca")
        if inputs['smoking_status'].values[0] == 1:
            risk_factors.append("Fumador activo")
        if inputs['bmi_category'].values[0] == 'Obesidad':
            risk_factors.append("Obesidad")
        if inputs['glucose_level_category'].values[0] == 'Alto':
            risk_factors.append("Nivel alto de glucosa")

        if risk_factors:
            for factor in risk_factors:
                st.write(f"- {factor}")
        else:
            st.write("No se identificaron factores de riesgo principales.")