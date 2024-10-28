from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware
import streamlit.web.bootstrap as bootstrap
import streamlit as st
import os

app = FastAPI()

# Configurar la aplicación Streamlit
def run_streamlit():
    bootstrap.run("main.py", "", [], {})

# Envolver Streamlit en FastAPI
@app.get("/")
def read_root():
    return WSGIMiddleware(run_streamlit)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)