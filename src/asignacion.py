"""Asignación del capital y estrategias (Pasos 11 y 12).

Solo se usan el riesgo (covarianza de entrenamiento, media cero) y los límites de concentración:
no entran rentabilidad esperada, Sharpe ni momentum.

Mínimo VaR conjunto (Ec. 10):
    mín_w  B · Φ⁻¹(α) · √(h · wᵀ Σ_R w)
    s.a.   Σ w_i = 1,   0,02 ≤ w_i ≤ 0,20,   Σ_{i∈sector s} w_i ≤ 0,35
Como B, Φ⁻¹(α) y h son constantes positivas, minimizar el VaR equivale a minimizar wᵀΣw.

Presupuesto de riesgo (Ec. 11):  f = mín{1, V / V_riesgo},  A_i = f·B·w_i,  E = (1 − f)·B.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm


def var_cartera(w: pd.Series, cov: pd.DataFrame, capital: float, alpha: float, h: int) -> float:
    """VaR normal con media cero (Ec. 9)."""
    w = w.reindex(cov.index).fillna(0.0)
    return float(capital * norm.ppf(alpha) * np.sqrt(h * w @ cov.to_numpy() @ w))


def var_individual(cov: pd.DataFrame, capital: float, alpha: float, h: int) -> pd.Series:
    """VaR de invertir todo el capital en cada acción por separado."""
    return capital * norm.ppf(alpha) * np.sqrt(h * pd.Series(np.diag(cov.to_numpy()), index=cov.index))


def pesos_iguales(tickers: list[str]) -> pd.Series:
    return pd.Series(1 / len(tickers), index=tickers)


def pesos_inverso_var(cov: pd.DataFrame, capital: float, alpha: float, h: int) -> pd.Series:
    """w_i ∝ 1/v_i. Ignora las correlaciones: no es una solución óptima."""
    inv = 1 / var_individual(cov, capital, alpha, h)
    return inv / inv.sum()


def peso_por_sector(w: pd.Series, sectores: dict[str, str]) -> pd.Series:
    sector_de = pd.Series({t: sectores[t] for t in w.index})
    return w.groupby(sector_de, sort=False).sum()


# ---------------------------------------------------------------------------
# Factibilidad y verificación de límites
# ---------------------------------------------------------------------------
def factibilidad(tickers: list[str], sectores: dict[str, str], w_min: float, w_max: float,
                 max_sector: float) -> list[str]:
    """Lista de problemas que harían imposible cumplir las restricciones (vacía si es factible)."""
    problemas = []
    n = len(tickers)
    if n * w_min > 1 + 1e-12:
        problemas.append(f"Con {n} acciones y un mínimo de {w_min:.0%} cada una se supera el 100 %.")
    if n * w_max < 1 - 1e-12:
        problemas.append(f"Con {n} acciones y un máximo de {w_max:.0%} cada una no se alcanza el 100 %.")
    cuenta = pd.Series([sectores[t] for t in tickers]).value_counts()
    for s, k in cuenta.items():
        if k * w_min > max_sector + 1e-12:
            problemas.append(f"El sector {s} tiene {k} acciones: sus mínimos ({k * w_min:.0%}) superan el tope de {max_sector:.0%}.")
    capacidad = sum(min(max_sector, k * w_max) for k in cuenta)
    if capacidad < 1 - 1e-12:
        problemas.append(f"Los topes por acción y por sector solo permiten invertir {capacidad:.0%} del capital.")
    return problemas


def _p(v: float, dec: int = 1) -> str:
    return f"{v * 100:.{dec}f} %".replace(".", ",")


def verificar(w: pd.Series, sectores: dict[str, str], w_min: float, w_max: float, max_sector: float,
              tol: float = 1e-6) -> list[tuple[bool, str]]:
    ps = peso_por_sector(w, sectores)
    return [
        (bool(abs(w.sum() - 1) < tol), f"Los pesos suman {_p(w.sum())}."),
        (bool((w >= w_min - tol).all() and (w <= w_max + tol).all()),
         f"Cada acción está entre {_p(w_min, 0)} y {_p(w_max, 0)} (mínimo {_p(w.min())}, máximo {_p(w.max())})."),
        (bool((ps <= max_sector + tol).all()),
         f"Ningún sector supera el {_p(max_sector, 0)} (el mayor es {ps.idxmax()} con {_p(ps.max())})."),
    ]


# ---------------------------------------------------------------------------
# Mínimo VaR conjunto (Ec. 10)
# ---------------------------------------------------------------------------
@dataclass
class ResultadoOptimo:
    pesos: pd.Series
    exito: bool
    mensaje: str
    problemas: list[str] = field(default_factory=list)


def minimo_var(cov: pd.DataFrame, sectores: dict[str, str], w_min: float = 0.02, w_max: float = 0.20,
               max_sector: float = 0.35) -> ResultadoOptimo:
    tickers = list(cov.index)
    problemas = factibilidad(tickers, sectores, w_min, w_max, max_sector)
    if problemas:
        return ResultadoOptimo(pd.Series(np.nan, index=tickers), False, "Restricciones inviables.", problemas)
    S = cov.to_numpy()
    escala = 1 / np.mean(np.diag(S))           # mejora la estabilidad numérica del solver
    restricciones = [{"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones_like(w)}]
    for s in dict.fromkeys(sectores[t] for t in tickers):
        mascara = np.array([sectores[t] == s for t in tickers], dtype=float)
        restricciones.append({"type": "ineq", "fun": lambda w, m=mascara: max_sector - m @ w,
                              "jac": lambda w, m=mascara: -m})
    res = minimize(lambda w: escala * w @ S @ w, np.full(len(tickers), 1 / len(tickers)),
                   jac=lambda w: 2 * escala * S @ w, method="SLSQP",
                   bounds=[(w_min, w_max)] * len(tickers), constraints=restricciones,
                   options={"ftol": 1e-12, "maxiter": 500})
    w = pd.Series(np.clip(res.x, w_min, w_max), index=tickers)
    w = w / w.sum()
    return ResultadoOptimo(w, bool(res.success), str(res.message))


# ---------------------------------------------------------------------------
# Corrección de concentración sectorial (Paso 12)
# ---------------------------------------------------------------------------
def corregir_sectores(w: pd.Series, sectores: dict[str, str], max_sector: float = 0.35,
                      w_min: float = 0.02, w_max: float = 0.20, iter_max: int = 50) -> tuple[pd.Series, list[str]]:
    """Recorta los sectores que superan el tope y reparte el exceso entre las acciones con espacio,
    en proporción a su peso. Devuelve los pesos corregidos y la lista de ajustes hechos."""
    w = w.clip(w_min, w_max)
    w = w / w.sum()
    pasos = []
    for _ in range(iter_max):
        ps = peso_por_sector(w, sectores)
        excedidos = ps[ps > max_sector + 1e-10]
        if excedidos.empty:
            break
        exceso = 0.0
        for s, peso in excedidos.items():
            miembros = [t for t in w.index if sectores[t] == s]
            factor = max_sector / peso
            exceso += peso - max_sector
            w[miembros] = w[miembros] * factor
            pasos.append(f"{s} pesaba {peso:.1%}: se recortó a {max_sector:.0%}.")
        ps = peso_por_sector(w, sectores)
        receptores = [t for t in w.index if ps[sectores[t]] < max_sector - 1e-10 and w[t] < w_max - 1e-10]
        if not receptores:
            break
        w[receptores] += exceso * w[receptores] / w[receptores].sum()
        w = w.clip(w_min, w_max)
        w = w / w.sum()
    return w, pasos


# ---------------------------------------------------------------------------
# Acciones enteras y presupuesto de riesgo
# ---------------------------------------------------------------------------
def acciones_enteras(w: pd.Series, capital: float, precios: pd.Series) -> pd.DataFrame:
    """Compra con acciones enteras (redondeo hacia abajo) al precio dado. Los pesos se recalculan
    sobre el capital total; lo que sobra queda como efectivo."""
    monto = capital * w
    acciones = np.floor(monto / precios[w.index])
    invertido = acciones * precios[w.index]
    return pd.DataFrame({"Peso objetivo": w, "Monto objetivo": monto, "Precio": precios[w.index],
                         "Acciones": acciones.astype(int), "Monto invertido": invertido,
                         "Peso real": invertido / capital})


@dataclass
class Presupuesto:
    f: float
    var_total: float          # V_riesgo: VaR con todo el capital invertido
    var_ajustado: float
    invertido: pd.Series      # A_i
    efectivo: float           # E


def presupuesto_riesgo(w: pd.Series, cov: pd.DataFrame, capital: float, alpha: float, h: int,
                       v_max: float) -> Presupuesto:
    """Ec. 11. El efectivo se supone sin riesgo; los pesos dentro de la parte invertida no cambian."""
    v_riesgo = var_cartera(w, cov, capital, alpha, h)
    f = min(1.0, v_max / v_riesgo) if v_riesgo > 0 else 1.0
    return Presupuesto(f, v_riesgo, f * v_riesgo, f * capital * w, (1 - f) * capital)


# ---------------------------------------------------------------------------
# Backtest de estrategias en el período de evaluación (Paso 12)
# ---------------------------------------------------------------------------
def backtest(precios: pd.DataFrame, inicio_eval: pd.Timestamp, pesos_fijos: dict[str, pd.Series],
             sectores: dict[str, str], capital: float, costo: float, ventana: int = 504,
             w_min: float = 0.02, w_max: float = 0.20, max_sector: float = 0.35) -> tuple[pd.DataFrame, dict]:
    """Valor diario de cada estrategia desde `inicio_eval`.

    - Estrategias de `pesos_fijos`: compra el primer día y mantiene (sin rebalanceo).
    - "Mínimo VaR mensual": el primer día hábil de cada mes recalcula el mínimo VaR con los `ventana`
      retornos anteriores (sin datos futuros) y rebalancea.
    El costo se cobra sobre el monto comprado o vendido en cada operación.
    """
    tick = list(next(iter(pesos_fijos.values())).index)
    p = precios[tick]
    eval_idx = p.index[p.index >= inicio_eval]
    resultados, info = {}, {}
    for nombre, w in pesos_fijos.items():
        acciones = capital * (1 - costo) * w / p.loc[eval_idx[0]]
        resultados[nombre] = p.loc[eval_idx] @ acciones
        info[nombre] = {"costos": capital * costo, "rebalanceos": 0}

    # Mínimo VaR con rebalanceo mensual
    r = p / p.shift(1) - 1
    acciones = pd.Series(0.0, index=tick)
    efectivo, costos, n_reb, valores = capital, 0.0, 0, []
    mes_previo = None
    for fecha in eval_idx:
        precio_hoy = p.loc[fecha]
        if fecha.month != mes_previo:
            pos = p.index.get_loc(fecha)
            hist = r.iloc[max(1, pos - ventana):pos]
            w_obj = minimo_var(hist.cov(ddof=1), sectores, w_min, w_max, max_sector).pesos
            valor = float(acciones @ precio_hoy + efectivo)
            objetivo = valor * w_obj / precio_hoy
            negociado = float((objetivo - acciones).abs() @ precio_hoy)
            c = negociado * costo
            acciones = (valor - c) * w_obj / precio_hoy
            efectivo, costos, n_reb = 0.0, costos + c, n_reb + 1
            mes_previo = fecha.month
        valores.append(float(acciones @ precio_hoy + efectivo))
    resultados["Mínimo VaR · rebalanceo mensual"] = pd.Series(valores, index=eval_idx)
    info["Mínimo VaR · rebalanceo mensual"] = {"costos": costos, "rebalanceos": n_reb}

    df = pd.DataFrame(resultados)
    for nombre in df:
        v = df[nombre]
        info[nombre].update({"rentabilidad": float(v.iloc[-1] / capital - 1),
                             "caida_maxima": float((v / v.cummax() - 1).min()),
                             "valor_final": float(v.iloc[-1])})
    return df, info
