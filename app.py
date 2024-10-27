import streamlit as st
import subprocess
import sys
import os

def main():
    subprocess.run([f"{sys.executable}", "-m", "streamlit", "run", "main.py"],
                  check=True)

if __name__ == "__main__":
    main()