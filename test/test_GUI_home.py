import pytest
import os
import sys

# Añadir el directorio raíz del proyecto al sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from screens.GUI_home import home_screen

def test_home_screen(mocker):
    # Mock de st.markdown para evitar errores de Streamlit
    mocker.patch('streamlit.markdown')
    
    # Llamar a la función
    home_screen()
    
    # Verificar que st.markdown fue llamado al menos una vez
    assert mocker.patch('streamlit.markdown').call_count > 0