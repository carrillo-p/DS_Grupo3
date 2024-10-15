import argparse
import pandas as pd
import joblib
import tensorflow as tf
from src.model.train_models_XGBoost import XGBoostStrokeModel

def load_models():
    xgb_model = XGBoostStrokeModel.load_model('src/model/xgboost_model.joblib', 'src/model/xgb_scaler.joblib')
    nn_model = tf.keras.models.load_model('src/model/nn_stroke.keras')
    nn_scaler = joblib.load('src/model/nn_scaler.joblib')
    return xgb_model, nn_model, nn_scaler

def predict_stroke_risk(inputs, xgb_model, nn_model, nn_scaler):
    xgb_probabilities = xgb_model.predict_proba(inputs)[0][1]
    inputs_nn = nn_scaler.transform(inputs)
    nn_probabilities = nn_model.predict(inputs_nn)
    nn_probability = nn_probabilities[0]
    final_probabilities = 0.6 * xgb_probabilities + 0.4 * nn_probability
    final_prediction = 1 if final_probabilities >= 0.5 else 0
    return final_probabilities, final_prediction

def main():
    parser = argparse.ArgumentParser(description="Predictor de Ictus")
    parser.add_argument("--gender", choices=["Masculino", "Femenino"], required=True, help="Género del paciente")
    parser.add_argument("--age", type=int, required=True, help="Edad del paciente")
    parser.add_argument("--hypertension", choices=["No", "Sí"], required=True, help="¿El paciente tiene hipertensión?")
    parser.add_argument("--heart_disease", choices=["No", "Sí"], required=True, help="¿El paciente tiene enfermedad cardíaca?")
    parser.add_argument("--ever_married", choices=["No", "Sí"], required=True, help="¿El paciente ha estado casado alguna vez?")
    parser.add_argument("--work_type", choices=["Privado", "Autónomo", "Gubernamental", "Niño", "Nunca ha trabajado"], required=True, help="Tipo de trabajo del paciente")
    parser.add_argument("--residence_type", choices=["Urbana", "Rural"], required=True, help="Tipo de residencia del paciente")
    parser.add_argument("--avg_glucose_level", type=float, required=True, help="Nivel promedio de glucosa del paciente")
    parser.add_argument("--bmi", type=float, required=True, help="IMC (Índice de Masa Corporal) del paciente")
    parser.add_argument("--smoking_status", choices=["Nunca fumó", "Fumador", "Exfumador"], required=True, help="Estado de fumador del paciente")

    args = parser.parse_args()

    inputs = pd.DataFrame({
        'gender': [1 if args.gender == "Masculino" else 0],
        'hypertension': [1 if args.hypertension == "Sí" else 0],
        'heart_disease': [1 if args.heart_disease == "Sí" else 0],
        'ever_married': [1 if args.ever_married == "Sí" else 0],
        'work_type': [0 if args.work_type == "Privado" else 1 if args.work_type == "Autónomo" else 2 if args.work_type == "Gubernamental" else 3 if args.work_type == "Niño" else 4],
        'Residence_type': [1 if args.residence_type == "Urbana" else 0],
        'smoking_status': [0 if args.smoking_status == "Nunca fumó" else 1 if args.smoking_status == "Exfumador" else 2],
        'bmi_category': [0 if args.bmi < 18.5 else 1 if args.bmi < 25 else 2 if args.bmi < 30 else 3],
        'age_category': [0 if args.age < 13 else 1 if args.age < 18 else 2 if args.age < 60 else 3],
        'glucose_level_category': [0 if args.avg_glucose_level < 100 else 1 if args.avg_glucose_level < 140 else 2]
    })

    xgb_model, nn_model, nn_scaler = load_models()
    final_probabilities, final_prediction = predict_stroke_risk(inputs, xgb_model, nn_model, nn_scaler)

    print("\nResultados de la Predicción:")
    print(f"Probabilidad de ictus: {final_probabilities:.2%}")
    print(f"Predicción: {'Alto riesgo de ictus' if final_prediction == 1 else 'Bajo riesgo de ictus'}")

    print("\nRecomendaciones:")
    if final_prediction == 1:
        print("Se recomienda consultar a un médico para una evaluación más detallada.")
    else:
        print("Mantener un estilo de vida saludable para prevenir riesgos futuros.")

    print("\nFactores de riesgo identificados:")
    risk_factors = []
    if inputs['hypertension'].values[0] == 1:
        risk_factors.append("Hipertensión")
    if inputs['heart_disease'].values[0] == 1:
        risk_factors.append("Enfermedad cardíaca")
    if inputs['smoking_status'].values[0] == 1:
        risk_factors.append("Fumador activo")
    if inputs['bmi_category'].values[0] == 3:
        risk_factors.append("Obesidad")
    if inputs['glucose_level_category'].values[0] == 2:
        risk_factors.append("Nivel alto de glucosa")

    if risk_factors:
        for factor in risk_factors:
            print(f"- {factor}")
    else:
        print("No se identificaron factores de riesgo principales.")

if __name__ == "__main__":
    main()