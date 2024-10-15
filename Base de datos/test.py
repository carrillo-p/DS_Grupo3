import pytest
import pandas as pd
from database_utils_mysql import get_database_connection, save_prediction_to_db

def test_database_connection():
    """Test para verificar que la conexión a la base de datos se establece correctamente."""
    try:
        engine = get_database_connection()
        connection = engine.connect()
        assert connection is not None
        connection.close()
    except Exception as e:
        pytest.fail(f"No se pudo conectar a la base de datos: {e}")

def test_save_prediction_to_db():
    """Test para verificar que los datos se guardan correctamente en la tabla patient_predictions."""
    # Crear un DataFrame de prueba con algunos datos ficticios
    test_data = pd.DataFrame({
        'age': [45],
        'gender': ['Male'],
        'hypertension': [0],
        'heart_disease': [1],
        'ever_married': ['Yes'],
        'work_type': ['Private'],
        'Residence_type': ['Urban'],
        'smoking_status': ['never smoked'],
        'bmi_category': ['Normal'],
        'age_category': ['Adult'],
        'glucose_level_category': ['Normal'],
        'stroke': [0],
        'prediction': [1],
        'prediction_probability': [0.85]
    })

    try:
        save_prediction_to_db(test_data)
        # Si la inserción no lanza errores, el test pasa
        assert True
    except Exception as e:
        pytest.fail(f"Error al guardar los datos en la base de datos: {e}")
