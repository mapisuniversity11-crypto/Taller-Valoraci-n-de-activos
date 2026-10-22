"""Valores por defecto de la aplicación.

Todo lo que el usuario puede cambiar en la barra lateral de la app arranca
con estos valores. Cambiar algo aquí = cambiar el caso base del taller.
Ver justificación de cada valor en docs/decisiones.md.
"""

# --- Universo de inversión (Paso 2.1) -------------------------------------
# 10 acciones en USD, mismo calendario (NYSE/NASDAQ), 5 sectores x 2.
# El sector es editable en la app por si falla el metadato de Yahoo.
TICKERS_SECTORES: dict[str, str] = {
    "AAPL": "Tecnología",
    "MSFT": "Tecnología",
    "JPM": "Financiero",
    "BAC": "Financiero",
    "XOM": "Energía",
    "CVX": "Energía",
    "JNJ": "Salud",
    "PFE": "Salud",
    "KO": "Consumo básico",
    "PG": "Consumo básico",
}

# Acción con opciones negociadas que usaremos para el contrato (Paso 5).
TICKER_OPCION: str = "AAPL"

# --- Datos (Paso 2.2 y 2.4) ------------------------------------------------
# Se necesitan >= 756 retornos conjuntos (504 calibración + 252 evaluación).
# 756 sesiones ~ 3 años; se piden ~4 años de margen por festivos y descartes.
ANIOS_HISTORIA: float = 4.0
MIN_RETORNOS: int = 756
N_CALIBRACION: int = 504
N_EVALUACION: int = 252

# --- Parámetros generales (Paso 3) -----------------------------------------
D: int = 252                 # sesiones por año
CAPITAL: float = 100_000.0   # USD
TASA_LIBRE_RIESGO: float = 0.04   # r, decimal, continua anual (revisar al ejecutar)
SEMILLA: int = 42

# --- Riesgo (Paso 10) ------------------------------------------------------
ALPHA: float = 0.99
HORIZONTE_VAR: int = 10      # sesiones

# --- Asignación (Paso 11) ---------------------------------------------------
W_MIN: float = 0.02
W_MAX: float = 0.20
MAX_SECTOR: float = 0.35

# --- Simulación (Pasos 4 y 10) ---------------------------------------------
N_TRAYECTORIAS_GBM: int = 5_000
N_ESCENARIOS_VAR: int = 20_000
DT: float = 1 / D

# --- Control del CRR (Paso 8) ----------------------------------------------
CONTROL_CRR = dict(S0=100.0, K=100.0, T=1.0, r=0.05, q=0.0, sigma=0.20)
CONTROL_CRR_VALOR: float = 10.4506
