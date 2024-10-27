import os
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

# Cargar variables de entorno
load_dotenv()

# Configuración base
BASE_DIR = Path(__file__).resolve().parent.parent
CREDENTIALS_PATH = os.getenv('FIREBASE_CREDENTIALS_PATH')
PROJECT_ID = os.getenv('FIREBASE_PROJECT_ID')

# Esquema de la base de datos
FIREBASE_SCHEMA = {
    "patient_predictions": {
        "fields": {
            "age": int,
            "gender": str,
            "hypertension": int,
            "heart_disease": int,
            "ever_married": str,
            "work_type": str,
            "Residence_type": str,
            "smoking_status": str,
            "bmi_category": str,
            "age_category": str,
            "glucose_level_category": str,
            "prediction": int,
            "prediction_probability": float,
            "timestamp": "timestamp"
        },
        "validations": {
            "gender": ["Male", "Female"],
            "ever_married": ["Yes", "No"],
            "work_type": ["Private", "Self-employed", "Govt_job", "Never_worked", "children"],
            "Residence_type": ["Urban", "Rural"],
            "smoking_status": ["formerly smoked", "never smoked", "smokes", "Unknown"],
            "bmi_category": ["Underweight", "Normal", "Overweight", "Obese"],
            "age_category": ["Young", "Middle", "Elderly"],
            "glucose_level_category": ["Normal", "Pre-diabetes", "Diabetes"]
        }
    },
    "model_metrics": {
        "fields": {
            "avg_prediction": float,
            "entropy": float,
            "ks_statistic": float,
            "timestamp": "timestamp"
        }
    },
    "new_data": {
        "fields": {
            "age": int,
            "gender": str,
            "hypertension": int,
            "heart_disease": int,
            "ever_married": str,
            "work_type": str,
            "Residence_type": str,
            "smoking_status": str,
            "bmi_category": str,
            "age_category": str,
            "glucose_level_category": str,
            "prediction": int,
            "prediction_probability": float,
            "stroke": int
        },
        "validations": {
            "gender": ["Male", "Female"],
            "ever_married": ["Yes", "No"],
            "work_type": ["Private", "Self-employed", "Govt_job", "Never_worked", "children"],
            "Residence_type": ["Urban", "Rural"],
            "smoking_status": ["formerly smoked", "never smoked", "smokes", "Unknown"],
            "bmi_category": ["Underweight", "Normal", "Overweight", "Obese"],
            "age_category": ["Young", "Middle", "Elderly"],
            "glucose_level_category": ["Normal", "Pre-diabetes", "Diabetes"]
        }
    }
}
