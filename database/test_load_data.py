# tests/test_create_database.py
import os
import pytest
from unittest.mock import patch, MagicMock
import psycopg2
from create_database import load_dotenv

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

@pytest.fixture
def mock_env_vars(monkeypatch):
    monkeypatch.setenv('DB_HOST', 'localhost')
    monkeypatch.setenv('DB_PORT', '5432')
    monkeypatch.setenv('DB_USER', 'test_user')
    monkeypatch.setenv('DB_PASSWORD', 'test_password')
    monkeypatch.setenv('DB_NAME', 'test_db')

@patch('psycopg2.connect')
def test_create_database(mock_connect, mock_env_vars):
    # Simular la creación de la base de datos
    mock_connection = MagicMock()
    mock_connect.return_value = mock_connection
    mock_cursor = MagicMock()
    mock_connection.cursor.return_value = mock_cursor

    from create_database import connection  # Importar después de la configuración del mock

    # Comprobar que la conexión se realiza
    mock_connect.assert_called_once_with(
        host='localhost',
        port='5432',
        user='test_user',
        password='test_password'
    )

    # Verificar que se ejecuta la consulta para crear la base de datos
    mock_cursor.execute.assert_called_once_with(
        'SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s',
        ['test_db']
    )

    # Comprobar que el cursor se cierra
    mock_cursor.close.assert_called_once()
    mock_connection.close.assert_called_once()

@patch('psycopg2.connect')
def test_create_table(mock_connect, mock_env_vars):
    # Simular la creación de la tabla
    mock_connection = MagicMock()
    mock_connect.return_value = mock_connection
    mock_cursor = MagicMock()
    mock_connection.cursor.return_value = mock_cursor

    from create_database import connection  # Importar después de la configuración del mock

    # Comprobar que la consulta para crear la tabla se ejecuta
    mock_cursor.execute.assert_called_with(
        '''
        CREATE TABLE IF NOT EXISTS patient_predictions (
            id SERIAL PRIMARY KEY,
            age INTEGER,
            gender VARCHAR(10),
            hypertension INTEGER,
            heart_disease INTEGER,
            ever_married VARCHAR(5),
            work_type VARCHAR(50),
            Residence_type VARCHAR(10),
            smoking_status VARCHAR(20),
            bmi_category VARCHAR(20),
            age_category VARCHAR(20),
            glucose_level_category VARCHAR(20),
            stroke INTEGER,
            prediction INTEGER,
            prediction_probability FLOAT
        );
        '''
    )

    # Comprobar que el cursor se cierra
    mock_cursor.close.assert_called_once()
    mock_connection.close.assert_called_once()
