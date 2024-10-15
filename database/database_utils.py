# Archivo: src/database_utils.py

import pandas as pd
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler, LabelEncoder

def get_database_connection():
    db_username = 'tu_usuario'  # Reemplaza con tu usuario de PostgreSQL
    db_password = 'tu_contraseña'  # Reemplaza con tu contraseña de PostgreSQL
    db_host = 'localhost'  # O la IP del servidor si no está en tu máquina local
    db_port = '5432'  # Puerto por defecto de PostgreSQL
    db_name = 'health_data'

    engine = create_engine(f'postgresql+psycopg2://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}')
    return engine

def save_prediction_to_db(data):
    engine = get_database_connection()
    data.to_sql('patient_predictions', engine, if_exists='append', index=False)