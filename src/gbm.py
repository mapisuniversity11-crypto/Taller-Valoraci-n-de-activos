"""Simulación GBM bajo la medida física P (Paso 4) y cuantiles teóricos (Ec. 4).

Transición exacta (Ec. 3):
    S_{t+Δt} = S_t · exp[(a − ½σ²)Δt + σ√Δt · Z],   Z ~ N(0, 1)

con a = μ total − q (drift del spot exdividendo), σ anual y Δt = 1/252.
Como el precio se obtiene con una exponencial, nunca puede ser negativo.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

D = 252
HORIZONTES = {"1 mes": 21, "3 meses": 63, "6 meses": 126, "12 meses": 252}   # en sesiones


def simular_gbm(s0: float, a: float, sigma: float, sesiones: int, n: int = 5_000,
                semilla: int = 42, dt: float = 1 / D) -> np.ndarray:
    """Matriz (n, sesiones + 1) de trayectorias; la columna 0 es S0 (Ec. 3)."""
    if s0 <= 0 or sigma < 0 or sesiones < 1 or n < 1:
        raise ValueError("Se requiere S0 > 0, σ ≥ 0, al menos 1 sesión y 1 trayectoria.")
    rng = np.random.default_rng(semilla)
    z = rng.standard_normal((n, sesiones))
    pasos = (a - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
    log_s = np.log(s0) + np.cumsum(pasos, axis=1)
    return np.hstack([np.full((n, 1), s0), np.exp(log_s)])


def cuantil_gbm(s0: float, a: float, sigma: float, t: float, beta: float) -> float:
    """Q_β(S_T) = S0 · exp[(a − ½σ²)T + σ√T · Φ⁻¹(β)]   (Ec. 4)."""
    return float(s0 * np.exp((a - 0.5 * sigma ** 2) * t + sigma * np.sqrt(t) * norm.ppf(beta)))


@dataclass
class ResumenGBM:
    s0: float
    t_anios: float
    p10: np.ndarray          # percentil 10 en cada sesión
    p50: np.ndarray          # mediana en cada sesión
    p90: np.ndarray          # percentil 90 en cada sesión
    finales: np.ndarray      # precios al final del horizonte
    media_sim: float
    error_estandar: float
    media_teorica: float     # S0 · e^{aT}
    z: float                 # (media simulada − teórica) / error estándar
    todos_positivos: bool
    prob_subir: float        # P(S_T > S0) en la simulación

    @property
    def mediana(self) -> float:
        return float(self.p50[-1])

    @property
    def rango(self) -> tuple[float, float]:
        return float(self.p10[-1]), float(self.p90[-1])

    @property
    def media_ok(self) -> bool:
        """La media simulada es compatible con la teórica (a menos de 2 errores estándar)."""
        return abs(self.z) < 2


def resumir(trayectorias: np.ndarray, a: float) -> ResumenGBM:
    """Bandas 10–90, mediana, distribución final y comprobaciones del Paso 4."""
    s0 = float(trayectorias[0, 0])
    sesiones = trayectorias.shape[1] - 1
    t = sesiones / D
    finales = trayectorias[:, -1]
    p10, p50, p90 = np.percentile(trayectorias, [10, 50, 90], axis=0)
    media = float(finales.mean())
    ee = float(finales.std(ddof=1) / np.sqrt(len(finales)))
    teorica = float(s0 * np.exp(a * t))
    return ResumenGBM(
        s0=s0, t_anios=t, p10=p10, p50=p50, p90=p90, finales=finales,
        media_sim=media, error_estandar=ee, media_teorica=teorica,
        z=(media - teorica) / ee if ee > 0 else 0.0,
        todos_positivos=bool((trayectorias > 0).all()),
        prob_subir=float((finales > s0).mean()),
    )
