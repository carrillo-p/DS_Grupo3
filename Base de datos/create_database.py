import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

# Obtener los parámetros de conexión a PostgreSQL desde las variables de entorno
db_host = os.getenv('DB_HOST')
db_port = os.getenv('DB_PORT')
db_user = os.getenv('DB_USER')
db_password = os.getenv('DB_PASSWORD')
db_name = os.getenv('DB_NAME')


# Conectar a PostgreSQL (sin especificar la base de datos para poder crearla)
connection = psycopg2.connect(
    host=db_host,
    port=db_port,
    user=db_user,
    password=db_password
)

# Crear un cursor para ejecutar comandos
connection.autocommit = True  # Necesario para ejecutar CREATE DATABASE fuera de una transacción
cursor = connection.cursor()

# Crear la base de datos si no existe
cursor.execute(sql.SQL("SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s"), [db_name])
exists = cursor.fetchone()
if not exists:
    cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name)))
    print(f"Base de datos '{db_name}' creada con éxito.")
else:
    print(f"La base de datos '{db_name}' ya existe.")

# Cerrar la conexión inicial
cursor.close()
connection.close()

# Conectarse a la nueva base de datos para crear las tablas
connection = psycopg2.connect(
    host=db_host,
    port=db_port,
    user=db_user,
    password=db_password,
    dbname=db_name
)

# Crear el cursor para interactuar con la base de datos
cursor = connection.cursor()

# Crear la tabla patient_predictions si no existe
create_table_query = '''
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
cursor.execute(create_table_query)
connection.commit()
print("Tabla 'patient_predictions' creada con éxito.")

# Cerrar la conexión y el cursor
cursor.close()
connection.close()
