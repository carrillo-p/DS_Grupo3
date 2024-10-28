from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware
import os
import logging
from logging.handlers import RotatingFileHandler

# Importar y ejecutar la aplicación principal desde main.py
import main

app = FastAPI()

# Configuración del WSGI con la aplicación Streamlit en main.py
app.mount("/", WSGIMiddleware(main.streamlit_app))

if __name__ == "__main__":
    # Configurar logging con rotación de archivos para evitar consumo excesivo de disco
    log_handler = RotatingFileHandler(
        'app.log',
        maxBytes=1024 * 1024,  # 1MB
        backupCount=3
    )
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[log_handler]
    )
    
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        workers=2,  # Limitar el número de workers para B3
        loop="uvloop",  # Usar uvloop para mejor rendimiento
        limit_concurrency=50  # Limitar concurrencia
    )