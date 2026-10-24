"""Descarga, limpieza y validación de precios (Paso 2).

Flujo:
    descargar_yahoo()  ->  limpiar()  ->  separar_ventanas()
    (o cargar_csv() si la descarga falla)

Convenciones:
- `ajustados`: cierre con auto_adjust=True (dividendos y splits). Se usa para retornos y calibración.
- `sin_ajustar`: cierre observado (auto_adjust=False, columna Close). Se usa como spot S0 de la opción.
  Nota: yfinance ajusta este "Close" por splits, pero no por dividendos.
- Índice: fechas (DatetimeIndex) ordenadas; columnas: tickers.
"""
from __future__ import annotations

import io
import json
import platform
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

CARPETA_RESPALDO = Path(__file__).resolve().parent.parent / "data" / "raw"
ARCHIVO_AJUSTADOS = "precios_ajustados.csv"
ARCHIVO_SIN_AJUSTAR = "precios_sin_ajustar.csv"
ARCHIVO_META = "metadatos.json"


@dataclass
class DatosCrudos:
    """Precios tal como llegan de la fuente, antes de limpiar."""
    ajustados: pd.DataFrame
    sin_ajustar: pd.DataFrame
    meta: dict = field(default_factory=dict)


@dataclass
class DatosLimpios:
    ajustados: pd.DataFrame          # sesiones comunes completas
    sin_ajustar: pd.DataFrame        # mismas fechas que `ajustados`
    calidad_por_ticker: pd.DataFrame
    resumen: dict
    meta: dict


# ---------------------------------------------------------------------------
# Versiones y metadatos (Paso 2.2: "guarden datos y versiones de bibliotecas")
# ---------------------------------------------------------------------------
def versiones_bibliotecas() -> dict:
    versiones = {"python": platform.python_version()}
    for nombre in ["yfinance", "pandas", "numpy", "scipy", "streamlit", "plotly"]:
        try:
            modulo = __import__(nombre)
            versiones[nombre] = getattr(modulo, "__version__", "desconocida")
        except ImportError:
            versiones[nombre] = "no instalada"
    return versiones


# ---------------------------------------------------------------------------
# Descarga (Paso 2.2)
# ---------------------------------------------------------------------------
def _extraer_cierre(df: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """yfinance devuelve columnas MultiIndex (campo, ticker); nos quedamos con 'Close'."""
    if isinstance(df.columns, pd.MultiIndex):
        cierre = df["Close"]
    else:  # un solo ticker
        cierre = df[["Close"]].rename(columns={"Close": tickers[0]})
    cierre = cierre.reindex(columns=tickers)
    cierre.index = pd.to_datetime(cierre.index).tz_localize(None)
    cierre.index.name = "Fecha"
    return cierre


def _completar_vacios(yf, tabla: pd.DataFrame, tickers: list[str], ajuste: bool, comunes: dict,
                      reintentos: int, pausa: float, registro: dict) -> pd.DataFrame:
    """Vuelve a descargar, una por una, las acciones que llegaron vacías."""
    series = {t: tabla[t] for t in tickers}
    for t in tickers:
        intentos = 0
        while series[t].dropna().empty and intentos < reintentos:
            intentos += 1
            time.sleep(pausa * intentos)
            uno = yf.download([t], auto_adjust=ajuste, **{**comunes, "threads": False})
            if uno is not None and not uno.empty:
                series[t] = _extraer_cierre(uno, [t])[t]
        if intentos:
            registro[f"{t} ({'ajustado' if ajuste else 'sin ajustar'})"] = intentos
    completa = pd.concat(series, axis=1).sort_index()   # unión de fechas
    completa.index.name = "Fecha"
    return completa[tickers]


def descargar_yahoo(tickers: list[str], inicio: date, ultimo_dia: date,
                    reintentos: int = 3, pausa: float = 2.0) -> DatosCrudos:
    """Descarga cierres diarios ajustados y sin ajustar.

    `ultimo_dia` es el último día que el usuario QUIERE incluir. Como en yfinance
    `end` es exclusiva, se pide end = ultimo_dia + 1 día.
    """
    import yfinance as yf

    fin_exclusivo = ultimo_dia + timedelta(days=1)
    comunes = dict(start=inicio, end=fin_exclusivo, interval="1d",
                   progress=False, group_by="column", threads=True)

    crudo_aj = yf.download(tickers, auto_adjust=True, **comunes)
    crudo_sin = yf.download(tickers, auto_adjust=False, **comunes)
    if crudo_aj is None or crudo_aj.empty:
        raise RuntimeError("Yahoo Finance no devolvió datos. Use el respaldo o suba un CSV.")

    ajustados = _extraer_cierre(crudo_aj, tickers)
    sin_ajustar = _extraer_cierre(crudo_sin, tickers)

    # Yahoo a veces deja vacía alguna acción en descargas múltiples (límite de
    # peticiones o falla momentánea). Se reintenta esa acción sola.
    reintentados: dict[str, int] = {}
    ajustados = _completar_vacios(yf, ajustados, tickers, True, comunes, reintentos, pausa, reintentados)
    sin_ajustar = _completar_vacios(yf, sin_ajustar, tickers, False, comunes, reintentos, pausa, reintentados)

    vacios = sorted({t for t in tickers if ajustados[t].dropna().empty or sin_ajustar[t].dropna().empty})
    if vacios:
        raise RuntimeError(
            f"Sin datos para: {', '.join(vacios)} tras {reintentos} reintentos. "
            "Puede ser un límite temporal de Yahoo: espere un minuto y vuelva a ejecutar, "
            "o revise el ticker."
        )

    meta = {
        "fuente": "Yahoo Finance (yfinance)",
        "consulta": datetime.now().isoformat(timespec="seconds"),
        "inicio": str(inicio),
        "ultimo_dia_solicitado": str(ultimo_dia),
        "end_exclusivo_usado": str(fin_exclusivo),
        "tickers": tickers,
        "reintentos_por_accion": reintentados,
        "versiones": versiones_bibliotecas(),
    }
    return DatosCrudos(ajustados, sin_ajustar, meta)


def sectores_yahoo(tickers: list[str]) -> dict[str, str | None]:
    """Sector según el metadato de Yahoo. Devuelve None si falla (se corrige a mano en la app)."""
    import yfinance as yf

    sectores = {}
    for t in tickers:
        try:
            sectores[t] = yf.Ticker(t).info.get("sector")
        except Exception:
            sectores[t] = None
    return sectores


# ---------------------------------------------------------------------------
# Dividendos q_i y tasa libre de riesgo r (Paso 3: "documenten la aproximación")
# ---------------------------------------------------------------------------
def q_desde_dividendos(dividendos_12m: float, precio: float) -> float:
    """Tasa de dividendo continua anual aproximada.

    y = D_12m / S   (rendimiento por dividendo de los últimos 12 meses, en decimal)
    q = ln(1 + y)   (equivalente continuo)
    """
    if precio <= 0:
        raise ValueError("El precio debe ser positivo.")
    return float(np.log1p(dividendos_12m / precio))


def dividendos_yahoo(tickers: list[str], sin_ajustar: pd.DataFrame) -> dict:
    """q_i por acción con dividendos pagados en los 365 días previos a la última fecha.

    Usa el último cierre SIN ajustar como precio (mismo precio que S0 de la opción).
    Devuelve {ticker: {"dividendos_12m", "precio", "fecha_precio", "q"}}; q = None si falla.
    """
    import yfinance as yf

    resultado = {}
    for t in tickers:
        serie = sin_ajustar[t].dropna()
        fecha = serie.index[-1]
        precio = float(serie.iloc[-1])
        try:
            div = yf.Ticker(t).dividends
            div.index = pd.to_datetime(div.index).tz_localize(None)
            d12 = float(div[(div.index > fecha - pd.Timedelta(days=365)) & (div.index <= fecha)].sum())
            q = q_desde_dividendos(d12, precio)
        except Exception:
            d12, q = None, None
        resultado[t] = {"dividendos_12m": d12, "precio": precio,
                        "fecha_precio": fecha.strftime("%Y-%m-%d"), "q": q}
    return resultado


def tasa_libre_riesgo_yahoo(ultimo_dia: date) -> dict | None:
    """r continua a partir del rendimiento de la letra del Tesoro a 13 semanas (^IRX).

    Yahoo cotiza ^IRX en porcentaje anual (4.25 = 4,25 %). Se convierte:
    y = IRX / 100,   r = ln(1 + y).
    Devuelve None si falla (se usa el valor manual).
    """
    import yfinance as yf

    try:
        df = yf.download(["^IRX"], start=ultimo_dia - timedelta(days=14),
                         end=ultimo_dia + timedelta(days=1), auto_adjust=False, progress=False)
        cierre = _extraer_cierre(df, ["^IRX"])["^IRX"].dropna()
        y = float(cierre.iloc[-1]) / 100
        if not 0 <= y <= 0.20:   # control de cordura: valor fuera de rango -> se usa el manual
            return None
        return {"irx_porcentaje": float(cierre.iloc[-1]), "fecha": cierre.index[-1].strftime("%Y-%m-%d"),
                "r": float(np.log1p(y)), "fuente": "Yahoo Finance ^IRX (T-bill 13 semanas)"}
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Respaldo CSV (Paso 2.2 y 2.3)
# ---------------------------------------------------------------------------
def guardar_respaldo(crudos: DatosCrudos, carpeta: Path = CARPETA_RESPALDO) -> Path:
    """Guarda los precios CRUDOS (antes de limpiar) y sus metadatos."""
    carpeta.mkdir(parents=True, exist_ok=True)
    crudos.ajustados.to_csv(carpeta / ARCHIVO_AJUSTADOS)
    crudos.sin_ajustar.to_csv(carpeta / ARCHIVO_SIN_AJUSTAR)
    (carpeta / ARCHIVO_META).write_text(json.dumps(crudos.meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return carpeta


def existe_respaldo(carpeta: Path = CARPETA_RESPALDO) -> bool:
    return (carpeta / ARCHIVO_AJUSTADOS).exists() and (carpeta / ARCHIVO_SIN_AJUSTAR).exists()


def _leer_csv_precios(origen) -> pd.DataFrame:
    df = pd.read_csv(origen, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Fecha"
    return df.apply(pd.to_numeric, errors="coerce")


def cargar_respaldo(carpeta: Path = CARPETA_RESPALDO) -> DatosCrudos:
    """Carga el respaldo guardado en el repositorio."""
    ajustados = _leer_csv_precios(carpeta / ARCHIVO_AJUSTADOS)
    sin_ajustar = _leer_csv_precios(carpeta / ARCHIVO_SIN_AJUSTAR)
    ruta_meta = carpeta / ARCHIVO_META
    meta = json.loads(ruta_meta.read_text(encoding="utf-8")) if ruta_meta.exists() else {}
    meta = {**meta, "cargado_desde": "respaldo del repositorio (data/raw)",
            "cargado_en": datetime.now().isoformat(timespec="seconds")}
    return DatosCrudos(ajustados, sin_ajustar, meta)


def cargar_csv(archivo_ajustados, archivo_sin_ajustar, procedencia: str) -> DatosCrudos:
    """Carga CSV subidos por el usuario. `procedencia` es obligatoria (Paso 2.3)."""
    if not procedencia.strip():
        raise ValueError("Indique la procedencia de los CSV (fuente y fecha de descarga).")
    ajustados = _leer_csv_precios(archivo_ajustados)
    sin_ajustar = _leer_csv_precios(archivo_sin_ajustar)
    meta = {"fuente": "CSV cargado por el usuario", "procedencia": procedencia.strip(),
            "cargado_en": datetime.now().isoformat(timespec="seconds"),
            "versiones": versiones_bibliotecas()}
    return DatosCrudos(ajustados, sin_ajustar, meta)


def a_csv_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    df.to_csv(buffer)
    return buffer.getvalue().encode("utf-8")


# ---------------------------------------------------------------------------
# Limpieza y calidad (Paso 2.3)
# ---------------------------------------------------------------------------
def limpiar(crudos: DatosCrudos, tickers: list[str]) -> DatosLimpios:
    """Ordena, quita duplicados, marca precios no positivos y deja sesiones comunes completas.

    NO rellena faltantes (ni con ceros ni con forward-fill): una fecha en la que
    alguna acción no tiene precio válido se descarta para todas.
    """
    faltan = [t for t in tickers if t not in crudos.ajustados.columns]
    if faltan:
        raise ValueError(f"Los datos no contienen: {', '.join(faltan)}")

    aj = crudos.ajustados[tickers].copy()
    sin = crudos.sin_ajustar.reindex(columns=tickers).copy()

    # 1. Ordenar fechas
    aj = aj.sort_index()
    sin = sin.sort_index()

    # 2. Duplicados de fecha (se conserva la última observación)
    dup_aj = int(aj.index.duplicated(keep="last").sum())
    aj = aj[~aj.index.duplicated(keep="last")]
    sin = sin[~sin.index.duplicated(keep="last")]

    # 3. Precios no positivos -> faltantes (no son precios válidos)
    no_pos = (aj <= 0).sum()
    aj = aj.where(aj > 0)
    sin = sin.where(sin > 0)

    # 4. Faltantes por acción (incluye los no positivos ya marcados)
    faltantes = aj.isna().sum()

    # 5. Sesiones comunes completas: todas las acciones con precio ajustado y sin ajustar
    sin = sin.reindex(aj.index)
    completas = aj.notna().all(axis=1) & sin.notna().all(axis=1)
    fechas_descartadas = aj.index[~completas]
    aj_ok = aj[completas]
    sin_ok = sin[completas]

    calidad = pd.DataFrame({
        "Observaciones": crudos.ajustados[tickers].notna().sum(),
        "Primera fecha": [crudos.ajustados[t].first_valid_index() for t in tickers],
        "Última fecha": [crudos.ajustados[t].last_valid_index() for t in tickers],
        "Precios ≤ 0": no_pos,
        "Faltantes": faltantes,
    })
    calidad.index.name = "Ticker"

    n_retornos = max(len(aj_ok) - 1, 0)
    resumen = {
        "fechas_originales": int(len(crudos.ajustados)),
        "duplicados_eliminados": dup_aj,
        "sesiones_descartadas": int(len(fechas_descartadas)),
        "fechas_descartadas": [d.strftime("%Y-%m-%d") for d in fechas_descartadas],
        "sesiones_comunes": int(len(aj_ok)),
        "retornos_conjuntos": n_retornos,
        "desde": aj_ok.index.min().strftime("%Y-%m-%d") if len(aj_ok) else None,
        "hasta": aj_ok.index.max().strftime("%Y-%m-%d") if len(aj_ok) else None,
    }
    return DatosLimpios(aj_ok, sin_ok, calidad, resumen, crudos.meta)


# ---------------------------------------------------------------------------
# Ventanas de calibración y evaluación (Paso 2.4)
# ---------------------------------------------------------------------------
def separar_ventanas(precios: pd.DataFrame, n_calibracion: int = 504, n_evaluacion: int = 252) -> dict:
    """Divide en retornos de calibración inicial (504) y de evaluación (últimos 252).

    Trabaja en términos de RETORNOS: con n retornos hay n + 1 precios.
    - evaluación: los últimos `n_evaluacion` retornos.
    - calibración: los `n_calibracion` retornos inmediatamente anteriores.
    Devuelve los PRECIOS de cada ventana (incluido el precio inicial necesario
    para calcular el primer retorno), listos para calibracion.py.
    """
    n_ret = len(precios) - 1
    requeridos = n_calibracion + n_evaluacion
    if n_ret < requeridos:
        raise ValueError(
            f"Solo hay {n_ret} retornos conjuntos; se necesitan al menos {requeridos}. "
            "Amplíen el período de descarga (fecha inicial más antigua)."
        )
    # Posiciones de precios: el retorno k usa los precios k-1 y k.
    inicio_eval = len(precios) - 1 - n_evaluacion          # precio base del primer retorno de evaluación
    inicio_cal = inicio_eval - n_calibracion
    calibracion = precios.iloc[inicio_cal: inicio_eval + 1]
    evaluacion = precios.iloc[inicio_eval:]
    return {
        "calibracion": calibracion,
        "evaluacion": evaluacion,
        "retornos_disponibles": n_ret,
        "retornos_no_usados": n_ret - requeridos,
    }
