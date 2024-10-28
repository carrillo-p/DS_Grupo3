from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware
import streamlit.web.bootstrap as bootstrap
import streamlit as st
import os

import streamlit as st

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}