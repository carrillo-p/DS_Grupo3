from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware
import os
import logging

# Importar y ejecutar la aplicación principal desde main.py
import main

app = FastAPI()

# Configuración del WSGI con la aplicación Streamlit en main.py
app.mount("/", WSGIMiddleware(main.streamlit_app))

if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))