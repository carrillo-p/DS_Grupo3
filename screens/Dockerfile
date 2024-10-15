FROM python:3.12-slim

# Directorio de trabajo
WORKDIR /app

# COPY init.sql /docker-entrypoint-initdb.d/init.sql
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    libmariadb-dev \
    libmariadb-dev-compat \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*
# Copiar archivos 
COPY . .
# Instalar requirements
RUN pip install --no-cache-dir -r requirements.txt

# Expone el puerto
EXPOSE 8501

# Compando de inicio
CMD ["streamlit", "run", "GUI.py"]