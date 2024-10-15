import pytest
import pandas as pd
import plotly.graph_objects as go
import sys
import os

# Añadir el directorio raíz del proyecto al sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from screens.aux_functions import load_data, create_gauge_chart

def test_load_data():
    df = load_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty

def test_create_gauge_chart():
    fig = create_gauge_chart(50, "Test Chart")
    assert isinstance(fig, go.Figure)