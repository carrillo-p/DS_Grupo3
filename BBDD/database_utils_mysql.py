# Archivo: src/database_utils.py

import pandas as pd
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler, LabelEncoder
from dotenv import load_dotenv
import os

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

def get_database_connection():
    db_user = os.getenv('DB_USER')  # Usuario de MySQL
    db_password = os.getenv('DB_PASSWORD')  # Contraseña de MySQL
    db_host = os.getenv('DB_HOST')  # Host o dirección IP del servidor MySQL
    db_port = os.getenv('DB_PORT')  # Puerto de conexión de MySQL (por defecto 3306)
    db_name = os.getenv('DB_NAME')  # Nombre de la base de datos

    # Crear el motor de conexión para MySQL
    engine = create_engine(f'mysql+mysqlconnector://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}')
    return engine

def save_prediction_to_db(data):
    engine = get_database_connection()
    data.to_sql('patient_predictions', engine, if_exists='append', index=False)