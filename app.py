from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware
import streamlit.web.bootstrap as bootstrap
import streamlit as st
import os

import streamlit as st

def main():
    st.title("Test App")
    st.write("Hello World!")

if __name__ == "__main__":
    main()