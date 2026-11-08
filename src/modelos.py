"""Modelos adicionales de la pestaña Modelos.

Cada modelo relaja un supuesto del GBM y se aplica a los datos del caso:

* Browniano estándar (aritmético): P_t = P_0 + m·t + s·W_t. Los movimientos se miden en
  dólares, no en porcentaje; sirve para mostrar por qué el GBM usa el logaritmo.
* Merton (difusión con saltos): el log-precio sigue un GBM más saltos que llegan como un
  proceso de Poisson de intensidad λ, con tamaño normal N(μ_J, δ_J²) en logaritmo.
* Browniano fraccional: el exponente de Hurst H mide si los retornos tienen memoria
  (H = 0,5 sin memoria, H > 0,5 tendencia, H < 0,5 reversión).

Unidades: tasas y volatilidades anuales en decimal, tiempo en años, D = 252 sesiones.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm, poisson

from .crr import black_scholes

D = 252


# ---------------------------------------------------------------------------
# Browniano estándar (aritmético)
# ---------------------------------------------------------------------------
@dataclass
class BrownianoCalibrado:
    m: float   # drift en USD por año
    s: float   # volatilidad en USD por año (√252 · desviación de ΔP diario)


def calibrar_browniano(precios: pd.Series, d: int = D) -> BrownianoCalibrado:
    """Calibra con los cambios diarios en dólares ΔP_t = P_t − P_{t−1} (divisor n − 1)."""
    dp = precios.diff().dropna()
    return BrownianoCalibrado(m=float(d * dp.mean()), s=float(np.sqrt(d) * dp.std(ddof=1)))


def simular_browniano(p0: float, m: float, s: float, sesiones: int, n: int = 5_000,
                      semilla: int = 42, dt: float = 1 / D) -> np.ndarray:
    """Matriz (n, sesiones + 1): P_{t+Δt} = P_t + m·Δt + s·√Δt·Z. Usa la misma Z que gbm.simular_gbm
    con la misma semilla, para que la comparación sea con los mismos choques."""
    rng = np.random.default_rng(semilla)
    z = rng.standard_normal((n, sesiones))
    pasos = m * dt + s * np.sqrt(dt) * z
    return np.hstack([np.full((n, 1), p0), p0 + np.cumsum(pasos, axis=1)])


def prob_negativo(p0: float, m: float, s: float, t: float) -> float:
    """P(P_T < 0) del browniano aritmético = Φ((−P0 − m·T) / (s·√T))."""
    if t <= 0 or s <= 0:
        return 0.0
    return float(norm.cdf((-p0 - m * t) / (s * np.sqrt(t))))


def volatilidad_por_nivel(precios: pd.Series, bloque: int = 63) -> pd.DataFrame:
    """En bloques trimestrales sin traslape: precio medio, desviación del cambio diario en USD y en %.

    Si la desviación en USD crece con el precio mientras la de % se mantiene, los movimientos son
    proporcionales al precio (como supone el GBM) y no fijos en dólares (como el browniano estándar).
    """
    dp = precios.diff()
    rp = precios.pct_change()
    filas = []
    for i in range(1, len(precios) - bloque + 1, bloque):
        tramo = slice(i, i + bloque)
        filas.append({
            "Desde": precios.index[i], "Precio medio": float(precios.iloc[tramo].mean()),
            "Movimiento diario USD": float(dp.iloc[tramo].std(ddof=1)),
            "Movimiento diario %": float(rp.iloc[tramo].std(ddof=1)),
        })
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Merton: difusión con saltos
# ---------------------------------------------------------------------------
@dataclass
class MertonCalibrado:
    sigma: float          # volatilidad de los días normales (anual)
    lam: float            # saltos por año
    mu_j: float           # tamaño medio del salto (log-retorno)
    delta_j: float        # dispersión del tamaño del salto
    fechas: pd.DatetimeIndex
    tamanos: pd.Series    # log-retorno de cada día de salto
    umbral: float         # |x − mediana| > umbral se considera salto
    sigma_total: float    # volatilidad total implícita: √(σ² + λ(μ_J² + δ_J²))


def calibrar_merton(x: pd.Series, k: float = 3.0, d: int = D) -> MertonCalibrado:
    """Detecta saltos como días con |x − mediana| > k · σ robusta (σ robusta = 1,4826 · MAD).

    La MAD no se infla con los propios saltos, así que el umbral no depende de ellos.
    σ se calcula con los días normales; λ = saltos / años de la muestra; μ_J y δ_J con los
    días de salto, descontando el drift de un día normal.
    """
    x = x.dropna()
    med = float(x.median())
    s_rob = 1.4826 * float((x - med).abs().median())
    umbral = k * s_rob
    es_salto = (x - med).abs() > umbral
    normales, saltos = x[~es_salto], x[es_salto]
    sigma = float(np.sqrt(d) * normales.std(ddof=1))
    anios = len(x) / d
    lam = len(saltos) / anios
    tam = saltos - normales.mean()
    mu_j = float(tam.mean()) if len(saltos) else 0.0
    delta_j = float(tam.std(ddof=1)) if len(saltos) > 1 else 0.0
    sigma_total = float(np.sqrt(sigma ** 2 + lam * (mu_j ** 2 + delta_j ** 2)))
    return MertonCalibrado(sigma, lam, mu_j, delta_j, saltos.index, saltos, umbral, sigma_total)


def precio_merton(s0: float, k: float, t: float, r: float, q: float, sigma: float, lam: float,
                  mu_j: float, delta_j: float, tipo: str = "call", n_terminos: int = 80) -> float:
    """Fórmula de Merton (1976) para la opción europea: promedio de Black-Scholes ponderado por
    la probabilidad de que ocurran n saltos antes del vencimiento.

        κ = exp(μ_J + ½δ_J²) − 1,   λ' = λ(1 + κ)
        σ_n² = σ² + n·δ_J²/T,       r_n = r − λκ + n·ln(1 + κ)/T
        V = Σ_n  Poisson(n; λ'T) · BS(S0, K, T, r_n, q, σ_n)
    """
    if t <= 0:
        return float(max(s0 - k, 0.0) if tipo == "call" else max(k - s0, 0.0))
    kappa = np.exp(mu_j + 0.5 * delta_j ** 2) - 1
    lam_p = lam * (1 + kappa)
    total = 0.0
    for n in range(n_terminos):
        peso = poisson.pmf(n, lam_p * t)
        if n > 5 and peso < 1e-14:
            break
        sig_n = np.sqrt(sigma ** 2 + n * delta_j ** 2 / t)
        r_n = r - lam * kappa + n * np.log1p(kappa) / t
        total += peso * black_scholes(s0, k, t, r_n, q, sig_n, tipo)      # Hull, cap. 27
    return float(total)


def prob_algun_salto(lam: float, t: float) -> float:
    """Probabilidad de al menos un salto en T años: 1 − e^{−λT}."""
    return float(1 - np.exp(-lam * t))


# ---------------------------------------------------------------------------
# Browniano fraccional y exponente de Hurst
# ---------------------------------------------------------------------------
ESCALAS_HURST = (1, 2, 4, 8, 16, 32)


def _desv_agregadas(arr: np.ndarray, escalas) -> np.ndarray:
    """Desviación (n − 1) de las sumas móviles de m retornos, para cada m."""
    cs = np.concatenate([[0.0], np.cumsum(arr)])
    return np.array([np.std(cs[m:] - cs[:-m], ddof=1) for m in escalas])


def volatilidad_por_horizonte(x: pd.Series, escalas=ESCALAS_HURST) -> pd.DataFrame:
    """Desviación de los retornos acumulados a m sesiones (sumas móviles) frente a la regla √m del GBM."""
    arr = np.asarray(pd.Series(x).dropna(), dtype=float)
    reales = _desv_agregadas(arr, escalas)
    s1 = float(np.std(arr, ddof=1))
    return pd.DataFrame({"Sesiones": list(escalas), "Real": reales,
                         "Regla √m (GBM)": [s1 * np.sqrt(m) for m in escalas]})


def hurst(x, escalas=ESCALAS_HURST) -> float:
    """Exponente de Hurst por varianza agregada: desv(suma de m retornos) ∝ m^H.
    H es la pendiente de log(desviación) contra log(m)."""
    arr = np.asarray(pd.Series(x).dropna(), dtype=float)
    return float(np.polyfit(np.log(escalas), np.log(_desv_agregadas(arr, escalas)), 1)[0])


def intervalo_sin_memoria(x, n_perm: int = 200, semilla: int = 42, escalas=ESCALAS_HURST) -> tuple[float, float]:
    """Rango P5–P95 de H al desordenar los días al azar (lo que daría una serie sin memoria del mismo
    tamaño). Incluye el sesgo de muestra finita del estimador."""
    rng = np.random.default_rng(semilla)
    arr = np.asarray(pd.Series(x).dropna(), dtype=float)
    hs = [hurst(rng.permutation(arr), escalas) for _ in range(n_perm)]
    return float(np.percentile(hs, 5)), float(np.percentile(hs, 95))


def tabla_hurst(x: pd.DataFrame, semilla: int = 42, n_perm: int = 200) -> pd.DataFrame:
    filas = []
    for i, t in enumerate(x.columns):
        h = hurst(x[t])
        lo, hi = intervalo_sin_memoria(x[t], n_perm, semilla + i)
        if h > hi:
            veredicto = "Tendencia"
        elif h < lo:
            veredicto = "Reversión"
        else:
            veredicto = "Sin memoria"
        filas.append({"Ticker": t, "H": h, "Sin memoria desde": lo, "Sin memoria hasta": hi, "Lectura": veredicto})
    return pd.DataFrame(filas)


def simular_fbm(h: float, sesiones: int = 252, n: int = 1, semilla: int = 42) -> np.ndarray:
    """Trayectorias (n, sesiones + 1) de browniano fraccional con B_H(1 año) de varianza 1.

    Se generan incrementos con covarianza γ(k) = ½(|k+1|^{2H} − 2|k|^{2H} + |k−1|^{2H}) por Cholesky.
    Con la misma semilla, los tres valores de H usan los mismos choques.
    """
    k = np.arange(sesiones)
    gam = 0.5 * (np.abs(k + 1) ** (2 * h) - 2 * np.abs(k) ** (2 * h) + np.abs(k - 1) ** (2 * h))
    cov = gam[np.abs(k[:, None] - k[None, :])]
    chol = np.linalg.cholesky(cov + 1e-12 * np.eye(sesiones))
    z = np.random.default_rng(semilla).standard_normal((n, sesiones))
    incr = z @ chol.T * (1 / sesiones) ** h
    return np.hstack([np.zeros((n, 1)), np.cumsum(incr, axis=1)])


def factor_var_hurst(h_var: int, h: float) -> float:
    """Cuánto cambia el VaR a h_var días si se escala con h^H en lugar de √h: h^H / h^0,5."""
    return float(h_var ** h / np.sqrt(h_var))
