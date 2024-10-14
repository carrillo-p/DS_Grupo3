# tests/test_database_utils.py
import os
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from sqlalchemy.engine import Engine
from database_utils import get_database_connection, save_prediction_to_db

@pytest.fixture
def mock_env_vars(monkeypatch):
    monkeypatch.setenv('DB_HOST', 'localhost')
    monkeypatch.setenv('DB_PORT', '5432')
    monkeypatch.setenv('DB_USER', 'test_user')
    monkeypatch.setenv('DB_PASSWORD', 'test_password')
    monkeypatch.setenv('DB_NAME', 'test_db')

@patch('database_utils.create_engine')
def test_get_database_connection(mock_create_engine, mock_env_vars):
    # Simular la creación del motor de la base de datos
    mock_engine = MagicMock(spec=Engine)
    mock_create_engine.return_value = mock_engine

    engine = get_database_connection()

    # Comprobar que el motor no sea None
    assert engine is not None
    mock_create_engine.assert_called_once_with(
        'postgresql+psycopg2://test_user:test_password@localhost:5432/test_db'
    )

@patch('database_utils.get_database_connection')
def test_save_prediction_to_db(mock_get_database_connection, mock_env_vars):
    # Simular la conexión a la base de datos
    mock_engine = MagicMock()
    mock_get_database_connection.return_value = mock_engine

    # Crear un DataFrame de ejemplo
    sample_data = pd.DataFrame({
        'age': [25],
        'stroke': [0]
    })

    save_prediction_to_db(sample_data)

    # Verificar que to_sql se llama correctamente
    mock_engine.to_sql.assert_called_once_with('patient_predictions', mock_engine, if_exists='append', index=False)

