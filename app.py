from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware
import streamlit.web.bootstrap as bootstrap
import streamlit as st
import os
import time
from threading import Thread
from src.model.train_models_XGBoost import XGBoostStrokeModel
import logging
from pathlib import Path

app = FastAPI()

def initialize_model():
    global model
    
    # Crear directorio para modelos si no existe
    model_dir = Path('src/model')
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / 'xgboost_model.joblib'
    scaler_path = model_dir / 'xgb_scaler.joblib'
    
    try:
        if model_path.exists() and scaler_path.exists():
            logging.info("Cargando modelo pre-entrenado...")
            model = XGBoostStrokeModel(model_path=str(model_path), scaler_path=str(scaler_path))
        else:
            logging.warning("No se encontró modelo pre-entrenado. El entrenamiento en B1 no es recomendado.")
            raise Exception("Se requiere un modelo pre-entrenado para B1")
    except Exception as e:
        logging.error(f"Error en la inicialización del modelo: {e}")
        raise


def background_worker(model):
    while True:
        try:
            model.check_and_retrain()
        except Exception as e:
            logging.error(f"Error en el worker de reentrenamiento: {e}")
        time.sleep(model.check_interval)

def run_streamlit():
    # Inicialización del modelo
    model = initialize_model()
    bg_thread = Thread(target=background_worker, args=(model,))
    bg_thread.daemon = True  # Hacer el thread daemon para que termine cuando la aplicación principal termine
    bg_thread.start()
    
    # Configuración de la página
    st.set_page_config(
        page_title="PREDICTUS - Predicción de Ictus",
        page_icon="🧠",
        layout="wide"
    )

    # Importar las pantallas
    from screens.GUI_home import home_screen
    from screens.GUI_predict import screen_predict
    from screens.GUI_report import screen_informe
    from screens.GUI_info import screen_info
    from screens.GUI_add import screen_add


    # Inicializar el estado de la sesión si no existe
    if 'screen' not in st.session_state:
        st.session_state.screen = 'home'

    def change_screen(new_screen):
        st.session_state.screen = new_screen

    # Sidebar navigation
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

    # Renderizar la pantalla correspondiente
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

# Crear una aplicación WSGI para Streamlit
streamlit_app = bootstrap.run(
    run_streamlit,
    '',
    [],
    flag_options={
        'server.address': '0.0.0.0',
        'server.port': int(os.getenv('PORT', '8000')),
        'server.baseUrlPath': ''
    }
)

# Montar la aplicación Streamlit en FastAPI
app.mount("/", WSGIMiddleware(streamlit_app))

if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))