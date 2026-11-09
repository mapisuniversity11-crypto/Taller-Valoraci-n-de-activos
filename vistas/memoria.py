"""Cálculos pesados guardados en memoria (st.cache_data).

Streamlit vuelve a ejecutar toda la app con cada clic. Estos cálculos solo se repiten cuando cambian
sus entradas (datos, fechas, pesos, semilla...); si no, se reutiliza el resultado anterior. Los números
son exactamente los mismos: solo cambia la velocidad.
"""
import pandas as pd
import streamlit as st

from src import asignacion, modelos, riesgo, rolling


@st.cache_data(show_spinner=False, max_entries=8)
def rolling_gbm(precios: pd.DataFrame, inicio, h: int, ventana: int):
    return rolling.rolling_gbm(precios, inicio, h, ventana)


@st.cache_data(show_spinner=False, max_entries=8)
def backtest(precios: pd.DataFrame, inicio, carteras: dict, sectores: dict, capital: float, costo: float,
             n_cal: int, w_min: float, w_max: float, max_sector: float):
    return asignacion.backtest(precios, inicio, carteras, sectores, capital, costo, n_cal, w_min, w_max, max_sector)


@st.cache_data(show_spinner=False, max_entries=8)
def tabla_hurst(x: pd.DataFrame, semilla: int) -> pd.DataFrame:
    return modelos.tabla_hurst(x, semilla)


@st.cache_data(show_spinner=False, max_entries=16)
def var_montecarlo(precios: pd.DataFrame, w: pd.Series, capital: float, alpha: float, h: int, n: int, semilla: int):
    return riesgo.var_montecarlo(precios, w, capital, alpha, h, n, semilla)


@st.cache_data(show_spinner=False, max_entries=16)
def excepciones(precios: pd.DataFrame, w: pd.Series, capital: float, alpha: float, inicio):
    return riesgo.excepciones(precios, w, capital, alpha, inicio)


@st.cache_data(show_spinner=False, max_entries=8)
def correlacion_movil(x: pd.DataFrame, ventana: int) -> pd.Series:
    return modelos.correlacion_promedio_movil(x, ventana)


@st.cache_data(show_spinner=False, max_entries=16)
def simular_manada(cov: pd.DataFrame, w: pd.Series, capital: float, h: float, horizonte: int, n: int, semilla: int):
    return modelos.simular_manada(cov, w, capital, h, horizonte, n, semilla)
