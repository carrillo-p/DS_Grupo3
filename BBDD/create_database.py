def create_database_and_table():

    import mysql.connector
    from mysql.connector import errorcode
    from dotenv import load_dotenv
    import os

    # Cargar las variables de entorno desde el archivo .env
    load_dotenv()

    # Obtener los parámetros de conexión a MySQL desde las variables de entorno
    db_host = os.getenv('DB_HOST')
    db_port = os.getenv('DB_PORT')
    db_user = os.getenv('DB_USER')
    db_password = os.getenv('DB_PASSWORD')
    db_name = os.getenv('DB_NAME')

    try:
        # Intentar conectar a MySQL
        connection = mysql.connector.connect(
            host=db_host,
            port=db_port,
            user=db_user,
            password=db_password
        )
        print("Conexión exitosa a MySQL.")

        # Crear un cursor para ejecutar comandos si la conexión fue exitosa
        cursor = connection.cursor()

        # Crear la base de datos si no existe
        cursor.execute(f"SHOW DATABASES LIKE '{db_name}'")
        exists = cursor.fetchone()
        if not exists:
            cursor.execute(f"CREATE DATABASE {db_name}")
            print(f"Base de datos '{db_name}' creada con éxito.")
        else:
            print(f"La base de datos '{db_name}' ya existe.")

        # Cerrar la conexión inicial
        cursor.close()
        connection.close()

        # Conectarse a la nueva base de datos para crear las tablas
        connection = mysql.connector.connect(
            host=db_host,
            port=db_port,
            user=db_user,
            password=db_password,
            database=db_name
        )

        # Crear el cursor para interactuar con la base de datos
        cursor = connection.cursor()

        # Crear la tabla patient_predictions si no existe
        create_table_query = '''
        CREATE TABLE IF NOT EXISTS patient_predictions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            age INT,
            gender VARCHAR(10),
            hypertension INT,
            heart_disease INT,
            ever_married VARCHAR(5),
            work_type VARCHAR(50),
            Residence_type VARCHAR(10),
            smoking_status VARCHAR(20),
            bmi_category VARCHAR(20),
            age_category VARCHAR(20),
            glucose_level_category VARCHAR(20),
            prediction INT,
            prediction_probability FLOAT
        );
        '''
        cursor.execute(create_table_query)
        connection.commit()
        print("Tabla 'patient_predictions' creada con éxito.")

        # Cerrar la conexión y el cursor
        cursor.close()
        connection.close()

    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("Error: Nombre de usuario o contraseña incorrectos.")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("Error: La base de datos no existe.")
        else:
            print(f"Error al conectar a MySQL: {err}")