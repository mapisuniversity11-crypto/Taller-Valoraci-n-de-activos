"""App del Taller de Valoración de Activos (Streamlit).

Ejecutar:  streamlit run app.py

Esta capa SOLO maneja interfaz: lee entradas, llama a los módulos de src/
y muestra resultados. Ningún cálculo financiero debe vivir aquí.
"""
from datetime import date, timedelta

import pandas as pd
import streamlit as st

from src import config

st.set_page_config(page_title="Valoración y riesgo", layout="wide")
st.title("Aplicación de valoración y riesgo")
st.caption("Modelo binomial, GBM, opciones y asignación sectorial por VaR — Universidad Externado de Colombia")

# ---------------------------------------------------------------------------
# Paso 1. Entradas (barra lateral)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Entradas")

    st.subheader("Acciones y sectores")
    universo = st.data_editor(
        pd.DataFrame(
            {"Ticker": list(config.TICKERS_SECTORES), "Sector": list(config.TICKERS_SECTORES.values())}
        ),
        num_rows="fixed",
        hide_index=True,
        help="Puede corregir el sector manualmente si falla el metadato.",
    )

    st.subheader("Fechas")
    hoy = date.today()
    fecha_fin = st.date_input("Último día con datos", hoy)
    fecha_inicio = st.date_input(
        "Fecha inicial", hoy - timedelta(days=int(365.25 * config.ANIOS_HISTORIA))
    )
    st.caption("En yfinance `end` es exclusiva: se descarga hasta fecha_fin + 1 día.")

    st.subheader("Cartera y riesgo")
    capital = st.number_input("Capital (USD)", value=config.CAPITAL, step=1_000.0)
    alpha = st.number_input("Confianza VaR", value=config.ALPHA, min_value=0.90, max_value=0.999, format="%.3f")
    h_var = st.number_input("Horizonte VaR (sesiones)", value=config.HORIZONTE_VAR, min_value=1)
    r = st.number_input("Tasa libre de riesgo r (decimal, anual)", value=config.TASA_LIBRE_RIESGO, format="%.4f")
    q = st.number_input("Dividendos q (decimal, continuo anual)", value=0.0, format="%.4f")
    semilla = st.number_input("Semilla", value=config.SEMILLA, step=1)

    st.subheader("Opción")
    ticker_opcion = st.selectbox(
        "Acción subyacente", list(universo["Ticker"]),
        index=list(universo["Ticker"]).index(config.TICKER_OPCION),
    )
    tipo_opcion = st.radio("Tipo", ["call", "put"], horizontal=True)
    estilo_opcion = st.radio("Estilo", ["europea", "americana"], horizontal=True)
    n_pasos = st.number_input("Pasos del árbol N", value=500, min_value=1)

    ejecutar = st.button("Ejecutar análisis", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Salidas
# ---------------------------------------------------------------------------
tabs = st.tabs(["Datos", "GBM", "Opciones", "Riesgo", "Asignación", "Evaluación"])

if not ejecutar:
    st.info("Configure las entradas y presione **Ejecutar análisis**.")
    st.stop()

with tabs[0]:
    st.subheader("Datos y calidad")
    st.write("Pendiente: descarga, reporte de calidad, precios normalizados, histogramas, correlaciones y parámetros.")

with tabs[1]:
    st.write("Pendiente (entrega 2): trayectorias GBM y cuantiles.")

with tabs[2]:
    st.write("Pendiente: árbol CRR (entrega 1) y comparación con mercado (entrega 2).")

with tabs[3]:
    st.write("Pendiente (entrega 2): VaR/ES histórico, normal y Monte Carlo.")

with tabs[4]:
    st.write("Pendiente (entrega 3): asignación por mínimo VaR y estrategias.")

with tabs[5]:
    st.write("Pendiente (entrega 2): rolling del GBM, RMSE y MAPE.")
