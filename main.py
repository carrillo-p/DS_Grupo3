import streamlit as st
import mlflow
from threading import Thread
from azureml.core import Workspace
from src.model.mlflow_xgboost import XGBoostStrokeModel, background_worker
from pathlib import Path

@st.cache_resource  # Usar cache para el modelo
def initialize_model():
    global model
    
    # Crear directorio para modelos si no existe
    model_dir = Path('src/model')
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / 'xgboost_model.joblib'
    scaler_path = model_dir / 'xgb_scaler.joblib'
    
    try:
        if model_path.exists() and scaler_path.exists():
            model = XGBoostStrokeModel(model_path=str(model_path), scaler_path=str(scaler_path))
        else:
            raise Exception("Se requiere un modelo pre-entrenado para B1")
    except Exception as e:
        raise
    
# Inicializar el modelo
model = initialize_model()

bg_thread = Thread(target=background_worker, args=(model,))
bg_thread.start()

# Configuración de la página
st.set_page_config(
    page_title="PREDICTUS - Predicción de Ictus",
    page_icon="🧠",
    layout="wide"
)

from screens.GUI_home import home_screen
from screens.GUI_predict import screen_predict
from screens.GUI_report import screen_informe
from screens.GUI_info import screen_info
from screens.GUI_add import screen_add


if 'screen' not in st.session_state:
    st.session_state.screen = 'home'

def change_screen(new_screen):
    st.session_state.screen = new_screen

st.sidebar.header("Menú de Navegación")
if st.sidebar.button("Home"):
    change_screen("home")
if st.sidebar.button("Predicción de Stroke"):
    change_screen("predict")
if st.sidebar.button("Métricas de Rendimiento"):
    change_screen("informe")
if st.sidebar.button("Información del Modelo"):
    change_screen("info")
if st.sidebar.button("Añadrir nuevo caso"):
    change_screen("nuevo")

if st.session_state.screen == 'home':
    home_screen()
elif st.session_state.screen == 'predict':
    screen_predict()
elif st.session_state.screen == 'informe':
    screen_informe()
elif st.session_state.screen == 'info':
    screen_info()
elif st.session_state.screen == 'nuevo':
    screen_add()

