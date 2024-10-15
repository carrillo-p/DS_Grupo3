import pytest
import sys
import os

# Añadir el directorio raíz del proyecto al sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from screens.GUI_predict import load_model, load_nn_model

def test_load_model():
    model = load_model()
    assert model is not None

def test_load_nn_model():
    model, scaler = load_nn_model()
    assert model is not None
    assert scaler is not None