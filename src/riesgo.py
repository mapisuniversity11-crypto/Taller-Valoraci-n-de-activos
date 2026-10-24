"""VaR y ES por tres métodos, excepciones, desempeño fuera de muestra y estrés (Paso 10).

Definiciones (capital B, pesos w, horizonte h sesiones, confianza α):
    Pérdida   L_h = V_0 − V_h            (positiva = se pierde dinero)
    VaR_α,h   = Q_α(L_h)                 (cuantil α de la pérdida)
    ES_α,h    = E[L_h | L_h ≥ VaR_α,h]   (pérdida promedio en la cola)

Convenciones documentadas:
- Cuantil empírico: método "higher" (la menor pérdida observada tal que al menos una fracción α de
  las pérdidas es menor o igual). Es conservador.
- ES: promedio de las pérdidas mayores o IGUALES al VaR (los empates entran en la cola).
- Histórico: bloques de h días solapados (uno por cada día de inicio): hay más bloques, pero
  bloques vecinos comparten días y no son independientes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

D = 252
METODO_CUANTIL = "higher"


def pesos_iguales(tickers: list[str]) -> pd.Series:
    return pd.Series(1 / len(tickers), index=tickers)


def var_es(perdidas: np.ndarray, alpha: float) -> tuple[float, float]:
    perdidas = np.asarray(perdidas, dtype=float)
    var = float(np.quantile(perdidas, alpha, method=METODO_CUANTIL))
    return var, float(perdidas[perdidas >= var].mean())


# ---------------------------------------------------------------------------
# Los tres métodos
# ---------------------------------------------------------------------------
def perdidas_historicas(precios: pd.DataFrame, w: pd.Series, capital: float, h: int) -> pd.Series:
    """L = −B Σ w_i R_i,h, con R_i,h el retorno compuesto de cada bloque de h días (solapados)."""
    r_h = (precios.shift(-h) / precios - 1).dropna()
    return -capital * (r_h[w.index] @ w)


def var_historico(precios, w, capital, alpha, h) -> dict:
    L = perdidas_historicas(precios, w, capital, h)
    var, es = var_es(L.to_numpy(), alpha)
    return {"metodo": "Histórico", "var": var, "es": es, "perdidas": L.to_numpy(),
            "n": len(L), "peor": float(L.max()), "fecha_peor": L.idxmax()}


def var_normal(cov_diaria: pd.DataFrame, w: pd.Series, capital: float, alpha: float, h: int) -> dict:
    """VaR⁰ = B · Φ⁻¹(α) · √(h · wᵀ Σ_R w)   (Ec. 9, media cero).
    ES normal (media cero): B · σ_h · φ(Φ⁻¹(α)) / (1 − α)."""
    w = w[cov_diaria.index]
    sigma_h = float(np.sqrt(h * w @ cov_diaria.to_numpy() @ w))
    z = norm.ppf(alpha)
    return {"metodo": "Normal", "var": capital * z * sigma_h,
            "es": capital * sigma_h * norm.pdf(z) / (1 - alpha), "sigma_h": sigma_h}


def var_montecarlo(precios: pd.DataFrame, w: pd.Series, capital: float, alpha: float, h: int,
                   n: int = 20_000, semilla: int = 42) -> dict:
    """Escenarios GBM conjuntos bajo P a h días, preservando correlaciones.

    Log-retorno a h días de cada acción ~ Normal((μ_i − ½σ_i²)·h/252, h·Σ_x), con μ_i el retorno
    total anual (precios ajustados) y Σ_x la covarianza diaria de log-retornos. La cartera se
    mantiene sin rebalanceo dentro de h: V_h = B Σ w_i exp(X_i).
    """
    x = np.log(precios / precios.shift(1)).dropna()[w.index]
    cov_x = x.cov(ddof=1).to_numpy()
    sigma2_anual = np.diag(cov_x) * D
    mu_total = D * x.mean().to_numpy() + 0.5 * sigma2_anual
    deriva = (mu_total - 0.5 * sigma2_anual) * h / D
    chol = np.linalg.cholesky(h * cov_x)
    rng = np.random.default_rng(semilla)
    X = deriva + rng.standard_normal((n, len(w))) @ chol.T
    v_h = capital * (np.exp(X) @ w.to_numpy())
    L = capital - v_h
    var, es = var_es(L, alpha)
    return {"metodo": "Monte Carlo", "var": var, "es": es, "perdidas": L, "n": n}


# ---------------------------------------------------------------------------
# Excepciones del VaR diario (ventana móvil de 252 sesiones)
# ---------------------------------------------------------------------------
def excepciones(precios: pd.DataFrame, w: pd.Series, capital: float, alpha: float,
                inicio_eval: pd.Timestamp, ventana: int = 252) -> pd.DataFrame:
    """Para cada día de evaluación t: VaR diario normal con los `ventana` retornos anteriores a t
    (sin datos futuros) y pérdida real de la cartera ese día (pesos fijos)."""
    r = (precios / precios.shift(1) - 1).dropna()[w.index]
    perdida_real = -capital * (r @ w)
    z = norm.ppf(alpha)
    filas = []
    posiciones = np.where(r.index >= inicio_eval)[0]
    for p in posiciones:
        if p < ventana:
            continue
        hist = r.iloc[p - ventana:p]
        sigma = float(np.sqrt(w @ hist.cov(ddof=1).to_numpy() @ w))
        var_d = capital * z * sigma
        filas.append({"Fecha": r.index[p], "Pérdida real": float(perdida_real.iloc[p]), "VaR diario": var_d})
    df = pd.DataFrame(filas).set_index("Fecha")
    df["Excepción"] = df["Pérdida real"] > df["VaR diario"]
    return df


def zona_basilea(n_excepciones: int, n_dias: int, alpha: float) -> str:
    """Semáforo de Basilea (definido para 250 días al 99 %; se escala con los días evaluados)."""
    escala = n_dias / 250 * (1 - alpha) / 0.01
    if n_excepciones <= 4 * escala:
        return "verde"
    return "amarilla" if n_excepciones <= 9 * escala else "roja"


# ---------------------------------------------------------------------------
# Desempeño fuera de muestra
# ---------------------------------------------------------------------------
@dataclass
class FueraDeMuestra:
    valor: pd.Series        # valor diario de la cartera
    caida: pd.Series        # caída desde el máximo previo (drawdown), en fracción
    costos: float
    rentabilidad: float
    caida_maxima: float
    fecha_caida_maxima: pd.Timestamp


def fuera_de_muestra(precios_eval: pd.DataFrame, w: pd.Series, capital: float, costo: float) -> FueraDeMuestra:
    """Cartera comprada el primer día de evaluación y mantenida (sin rebalanceo).
    Se paga `costo` (fracción) sobre el monto comprado; lo invertido es B·(1 − costo)."""
    costos = capital * costo
    acciones = (capital - costos) * w / precios_eval.iloc[0][w.index]
    valor = precios_eval[w.index] @ acciones
    caida = valor / valor.cummax() - 1
    return FueraDeMuestra(valor, caida, costos, float(valor.iloc[-1] / capital - 1),
                          float(caida.min()), caida.idxmin())


# ---------------------------------------------------------------------------
# Estrés
# ---------------------------------------------------------------------------
def estres(w: pd.Series, sectores: dict[str, str], cov_diaria: pd.DataFrame, capital: float,
           alpha: float, h: int) -> tuple[pd.DataFrame, dict]:
    """Los tres escenarios del instructivo, en USD de pérdida. Devuelve la tabla y qué sector se
    tomó como el de mayor peso (y si hubo empate)."""
    sector_de = pd.Series({t: sectores[t] for t in w.index})
    peso_sector = w.groupby(sector_de, sort=False).sum()
    s_max = peso_sector.idxmax()           # en empate, el primero en el orden de la cartera
    w_s = float(peso_sector[s_max])
    # Volatilidad ×1,5 y correlación 0,8 entre todas las acciones
    sig = np.sqrt(np.diag(cov_diaria.to_numpy())) * 1.5
    corr = np.full((len(sig), len(sig)), 0.8)
    np.fill_diagonal(corr, 1.0)
    cov_e = pd.DataFrame(np.outer(sig, sig) * corr, index=cov_diaria.index, columns=cov_diaria.columns)
    filas = [
        ("Todas las acciones caen 10 %", capital * 0.10),
        (f"{s_max} cae 20 % y el resto 5 %", capital * (0.20 * w_s + 0.05 * (1 - w_s))),
        ("Volatilidad ×1,5 y correlación 0,8 (VaR normal)", var_normal(cov_e, w, capital, alpha, h)["var"]),
    ]
    info = {"sector_mayor": s_max, "empate": bool((peso_sector == w_s).sum() > 1)}
    return pd.DataFrame(filas, columns=["Escenario", "Pérdida"]), info
