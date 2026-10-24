"""Evaluación del GBM con ventana móvil de 252 retornos (Paso 13).

Para cada acción y fecha de origen t del período de evaluación:
  1. Se toman los 252 log-retornos disponibles hasta t (253 precios ajustados) y se reestiman
     σ̂_t = √252 · s y μ̂_t total = 252 · x̄ + ½σ̂_t².
  2. Pronóstico puntual a H sesiones = media condicional del GBM (Ec. 12):
         P̂_{t+H|t} = P_t · exp(μ̂_t · H / 252)
  3. Error e_t = P_{t+H} − P̂_{t+H|t}. Con los m pares de la prueba (Ec. 13):
         RMSE = √(Σ e² / m)            (en unidades del precio ajustado)
         MAPE = 100/m · Σ |e| / P_{t+H} (en %)
  4. Referencia: pronóstico ingenuo P̂ = P_t.
Complemento (no lo pide el instructivo): cobertura de la banda P10–P90 del GBM a H sesiones.
No se usa información posterior a t para pronosticar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import gbm

D = 252


def rolling_gbm(precios: pd.DataFrame, inicio_eval: pd.Timestamp, h: int = 1,
                ventana: int = 252) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Devuelve el detalle por acción y la tabla resumen con RMSE, MAPE y cobertura."""
    if h < 1:
        raise ValueError("H debe ser al menos 1 sesión.")
    x = np.log(precios / precios.shift(1))
    media = x.rolling(ventana).mean()                   # usa los retornos hasta t (incluido)
    desv = x.rolling(ventana).std(ddof=1)
    sigma = np.sqrt(D) * desv
    mu = D * media + 0.5 * sigma ** 2
    objetivo = precios.shift(-h)
    fecha_obj = pd.Series(precios.index, index=precios.index).shift(-h)

    detalle, filas = {}, []
    origenes = precios.index[precios.index >= inicio_eval]
    for t in precios.columns:
        df = pd.DataFrame({
            "Precio en t": precios[t], "Observado": objetivo[t], "Fecha objetivo": fecha_obj,
            "Pronóstico GBM": precios[t] * np.exp(mu[t] * h / D),
            "μ̂": mu[t], "σ̂": sigma[t],
        }).loc[origenes].dropna()
        df = df[df["Observado"] > 0]                    # solo objetivos observados y positivos
        df["Ingenuo"] = df["Precio en t"]
        df["Error GBM"] = df["Observado"] - df["Pronóstico GBM"]
        df["Error ingenuo"] = df["Observado"] - df["Ingenuo"]
        tt = h / D
        df["P10"] = [gbm.cuantil_gbm(p, m, s, tt, 0.10) for p, m, s in zip(df["Precio en t"], df["μ̂"], df["σ̂"])]
        df["P90"] = [gbm.cuantil_gbm(p, m, s, tt, 0.90) for p, m, s in zip(df["Precio en t"], df["μ̂"], df["σ̂"])]
        df["Dentro de la banda"] = df["Observado"].between(df["P10"], df["P90"])
        detalle[t] = df

        m = len(df)
        rmse_g = float(np.sqrt((df["Error GBM"] ** 2).mean()))
        rmse_i = float(np.sqrt((df["Error ingenuo"] ** 2).mean()))
        mape_g = float(100 * (df["Error GBM"].abs() / df["Observado"]).mean())
        mape_i = float(100 * (df["Error ingenuo"].abs() / df["Observado"]).mean())
        filas.append({"Ticker": t, "m": m, "RMSE GBM": rmse_g, "RMSE ingenuo": rmse_i,
                      "MAPE GBM (%)": mape_g, "MAPE ingenuo (%)": mape_i,
                      "¿Le gana al ingenuo?": "Sí" if mape_g < mape_i else "No",
                      "Cobertura P10–P90": float(df["Dentro de la banda"].mean()),
                      "Desde": df.index.min(), "Hasta": df["Fecha objetivo"].max()})
    return detalle, pd.DataFrame(filas)
