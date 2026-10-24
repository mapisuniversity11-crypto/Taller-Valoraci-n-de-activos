"""Retornos y calibración del modelo (Paso 3).

Convenciones: tasas y retornos en decimal, tiempo en años, D = 252 sesiones por año.
Desviaciones y covarianzas muestrales con divisor n − 1 (ddof=1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

D_DEFECTO = 252


def retornos_simples(precios: pd.DataFrame) -> pd.DataFrame:
    """R_{i,t} = P_{i,t} / P_{i,t-1} − 1   (Ec. 1)."""
    return (precios / precios.shift(1) - 1).iloc[1:]


def log_retornos(precios: pd.DataFrame) -> pd.DataFrame:
    """x_{i,t} = ln(P_{i,t} / P_{i,t-1})   (Ec. 1)."""
    return np.log(precios / precios.shift(1)).iloc[1:]


def calibrar(precios: pd.DataFrame, q: pd.Series | dict | None = None, D: int = D_DEFECTO) -> pd.DataFrame:
    """Parámetros anuales por acción a partir de precios ajustados.

    σ̂_i        = √D · s_{x,i}                  (Ec. 2, s con divisor n − 1)
    μ̂_i total  = D · x̄_i + ½ σ̂_i²             (Ec. 2)
    a_i        = μ̂_i total − q_i               (drift del spot exdividendo)
    """
    x = log_retornos(precios)
    q = pd.Series(q if q is not None else 0.0, index=precios.columns, dtype=float) \
        if not isinstance(q, pd.Series) else q.reindex(precios.columns).fillna(0.0)

    media_diaria = x.mean()
    s_diaria = x.std(ddof=1)
    sigma = np.sqrt(D) * s_diaria
    mu_total = D * media_diaria + 0.5 * sigma ** 2

    return pd.DataFrame({
        "n": x.count(),
        "Media diaria x̄": media_diaria,
        "Desv. diaria s": s_diaria,
        "σ anual": sigma,
        "μ total anual": mu_total,
        "q": q,
        "a = μ − q": mu_total - q,
    })


def covarianza_diaria(precios: pd.DataFrame, tipo: str = "simple") -> pd.DataFrame:
    """Covarianza diaria muestral (n − 1) en las mismas fechas.

    tipo="simple" -> Σ_R de retornos simples (es la que usa el VaR normal, Ec. 9).
    tipo="log"    -> covarianza de log-retornos (para simular GBM conjunto).
    """
    r = retornos_simples(precios) if tipo == "simple" else log_retornos(precios)
    return r.cov(ddof=1)


def correlacion(precios: pd.DataFrame) -> pd.DataFrame:
    """Correlación de log-retornos en las mismas fechas."""
    return log_retornos(precios).corr()


def normalizar(precios: pd.DataFrame, base: float = 100.0) -> pd.DataFrame:
    """Precios normalizados a `base` en la primera fecha (para comparar trayectorias)."""
    return base * precios / precios.iloc[0]
