import streamlit as st
from screens.aux_functions import load_css, load_image
from BBDD.create_database import create_database_and_table  # Importa la función de creación de la base de datos
from BBDD.database_utils import save_prediction_to_db  # Importa la función para guardar predicciones en la base de datos


def home_screen():
    load_css('style.css')

    # Logo
    st.markdown('<div style="text-align: center;">', unsafe_allow_html=True)
    image = load_image('predictus.png')
    st.image(image, width=150)
    st.markdown('</div>', unsafe_allow_html=True)

    # Introducción
    st.markdown('<p class="medium-font">Bienvenido a PREDICTUS, nuestro innovador servicio de predicción de ictus. Utilizamos inteligencia artificial de vanguardia para ayudar en la prevención y detección temprana.</p>', unsafe_allow_html=True)

    # Servicios
    st.markdown('## Nuestros Servicios', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="service-box">
        <h3>🧠 Predicción de Riesgo de Ictus</h3>
        <p>Utilizando algoritmos avanzados de machine learning, evaluamos su riesgo personal de sufrir un ictus basándonos en diversos factores de salud.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="service-box">
        <h3>📊 Análisis Detallado</h3>
        <p>Proporcionamos un informe completo sobre los factores de riesgo identificados y recomendaciones personalizadas para mejorar su salud.</p>
        </div>
        """, unsafe_allow_html=True)

    # Llamada a la acción
    st.markdown('## ¿Listo para comenzar?', unsafe_allow_html=True)
    if st.button('Iniciar Predicción'):
        st.session_state.screen = 'predict'

    # Información adicional
    st.markdown('### Sobre el Ictus', unsafe_allow_html=True)
    st.markdown("""
    El ictus es una emergencia médica que ocurre cuando el suministro de sangre al cerebro se interrumpe. 
    La detección temprana y la prevención son cruciales. PREDICTUS está diseñado para ayudarle a 
    entender y gestionar su riesgo personal utilizando la más avanzada tecnología de inteligencia artificial.
    """)

    # Pie de página
    st.markdown('---')
    st.markdown('© 2024 PREDICTUS - Tecnología Avanzada para la Prevención de Ictus. Todos los derechos reservados.')

if __name__ == "__main__":
    home_screen()