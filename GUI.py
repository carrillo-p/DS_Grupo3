import streamlit as st

# Configuración de la página
st.set_page_config(
    page_title="PREDICTUS - Predicción de Ictus",
    page_icon="🧠",
    layout="wide"
)

from screens.GUI_home import home_screen
from screens.GUI_predict import screen_predict
from screens.GUI_report import screen_informe
from BBDD.create_database import create_database_and_table  # Importa la función de creación de la base de datos
from screens.GUI_img import screen_image_prediction  # Importa la nueva pantalla para la predicción con imágenes

def main():
    create_database_and_table()  # Llama a la función importada

if __name__ == "__main__":
    main()

if 'screen' not in st.session_state:
    st.session_state.screen = 'home'

def change_screen(new_screen):
    st.session_state.screen = new_screen

st.sidebar.header("Menú de Navegación")
if st.sidebar.button("Home"):
    change_screen("home")
if st.sidebar.button("Predicción de Stroke"):
    change_screen("predict")
if st.sidebar.button("Clasificación de Imágenes"):  # Nueva opción en el menú
    change_screen("image_prediction")
if st.sidebar.button("Informe de Modelos"):
    change_screen("informe")

if st.session_state.screen == 'home':
    home_screen()
elif st.session_state.screen == 'predict':
    screen_predict()
elif st.session_state.screen == 'image_prediction':  # Mostrar la nueva pantalla
    screen_image_prediction()
elif st.session_state.screen == 'informe':
    screen_informe()