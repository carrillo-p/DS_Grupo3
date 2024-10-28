from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware
import streamlit.web.bootstrap as bootstrap

app = FastAPI()

def run_streamlit():
    bootstrap.run("main.py", "", [], {})

app.mount("/", WSGIMiddleware(run_streamlit))