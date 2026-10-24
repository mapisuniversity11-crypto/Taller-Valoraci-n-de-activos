"""Contratos de mercado, elección y comparación con el CRR (Pasos 5, 6, 7 y 9).

Convenciones del instructivo:
- Para VALORAR se usa el tiempo en días calendario: T = d_c / 365.
- Para PROYECTAR con el GBM se usan sesiones bursátiles y su equivalente anual: T_GBM = sesiones / 252.
- Proyección y probabilidades: medida física P, drift a = μ total − q.
- Valoración: medida neutral al riesgo Q, drift r − q (árbol CRR).
- El strike pertenece a la opción, no a la acción.
"""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from src import crr, gbm

CARPETA = Path(__file__).resolve().parent.parent / "data" / "raw"
OBJETIVOS_DIAS = (30, 91, 182)          # ~1, 3 y 6 meses
CUANTILES = (0.25, 0.50, 0.75)
COLUMNAS = ["contractSymbol", "strike", "lastPrice", "bid", "ask", "volume", "openInterest",
            "impliedVolatility", "lastTradeDate"]


# ---------------------------------------------------------------------------
# Descarga y respaldo de la cadena de opciones (Paso 5.1)
# ---------------------------------------------------------------------------
def elegir_vencimientos(disponibles: list[str], hoy: date, objetivos=OBJETIVOS_DIAS) -> list[str]:
    """Para cada objetivo (en días), el vencimiento real más cercano, sin repetir."""
    fechas = sorted({pd.Timestamp(f).date() for f in disponibles if pd.Timestamp(f).date() > hoy})
    elegidos = []
    for obj in objetivos:
        libres = [f for f in fechas if f not in elegidos]
        if libres:
            elegidos.append(min(libres, key=lambda f: abs((f - hoy).days - obj)))
    return [f.isoformat() for f in sorted(elegidos)]


def descargar_cadena(ticker: str, hoy: date | None = None) -> tuple[pd.DataFrame, dict]:
    """Calls y puts de los 3 vencimientos más cercanos a 1, 3 y 6 meses."""
    import yfinance as yf

    hoy = hoy or date.today()
    tk = yf.Ticker(ticker)
    vencs = elegir_vencimientos(list(tk.options), hoy)
    if len(vencs) < 3:
        raise RuntimeError(f"{ticker} no tiene suficientes vencimientos listados.")
    partes = []
    for v in vencs:
        cadena = tk.option_chain(v)
        for tipo, df in (("call", cadena.calls), ("put", cadena.puts)):
            df = df.reindex(columns=COLUMNAS).copy()
            df["tipo"], df["vencimiento"] = tipo, v
            partes.append(df)
    tabla = pd.concat(partes, ignore_index=True)
    meta = {"ticker": ticker, "fuente": "Yahoo Finance (yfinance) · Ticker.option_chain",
            "consulta": datetime.now().isoformat(timespec="seconds"), "vencimientos": vencs,
            "nota": "Las cotizaciones de Yahoo pueden ser diferidas."}
    return tabla, meta


def guardar_cadena(tabla: pd.DataFrame, meta: dict, carpeta: Path = CARPETA) -> None:
    carpeta.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(carpeta / f"opciones_{meta['ticker']}.csv", index=False)
    (carpeta / f"opciones_{meta['ticker']}.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                                             encoding="utf-8")


def cargar_cadena(ticker: str, carpeta: Path = CARPETA) -> tuple[pd.DataFrame, dict] | None:
    ruta = carpeta / f"opciones_{ticker}.csv"
    if not ruta.exists():
        return None
    meta_ruta = carpeta / f"opciones_{ticker}.json"
    meta = json.loads(meta_ruta.read_text(encoding="utf-8")) if meta_ruta.exists() else {"ticker": ticker}
    return pd.read_csv(ruta), meta


# ---------------------------------------------------------------------------
# Tabla de candidatos (Paso 5.2 y 5.3)
# ---------------------------------------------------------------------------
def cotizacion_valida(bid, ask) -> bool:
    """Paso 9: se exige 0 < bid ≤ ask para poder comparar."""
    return bool(pd.notna(bid) and pd.notna(ask) and 0 < bid <= ask)


def candidatos(tabla: pd.DataFrame, tipo: str, s0: float, a: float, sigma: float, hoy: date) -> pd.DataFrame:
    """Para cada vencimiento, los strikes listados más cercanos a los cuantiles 25, 50 y 75
    del precio proyectado (Ec. 4). Si dos cuantiles caen en el mismo strike, se usa el vecino."""
    filas = []
    for venc, grupo in tabla[tabla["tipo"] == tipo].groupby("vencimiento"):
        f_venc = pd.Timestamp(venc).date()
        dc = (f_venc - hoy).days
        sesiones = int(np.busday_count(hoy, f_venc))
        t_val, t_gbm = dc / 365, sesiones / gbm.D
        strikes = np.sort(grupo["strike"].dropna().unique())
        usados = set()
        for beta in CUANTILES:
            objetivo = gbm.cuantil_gbm(s0, a, sigma, t_gbm, beta)
            orden = strikes[np.argsort(np.abs(strikes - objetivo))]
            k = next((x for x in orden if x not in usados), None)
            if k is None:
                continue
            usados.add(k)
            fila = grupo[grupo["strike"] == k].iloc[0]
            bid, ask = fila["bid"], fila["ask"]
            valida = cotizacion_valida(bid, ask)
            medio = (bid + ask) / 2 if valida else np.nan
            filas.append({
                "Vencimiento": venc, "Días": dc, "Sesiones": sesiones, "T valoración": t_val, "T GBM": t_gbm,
                "Cuantil": f"P{int(beta * 100)}", "Precio proyectado": objetivo, "Strike": float(k),
                "Bid": bid, "Ask": ask, "Medio": medio,
                "Spread %": (ask - bid) / medio if valida else np.nan,
                "Volumen": fila["volume"], "Interés abierto": fila["openInterest"],
                "Vol. implícita": fila["impliedVolatility"], "Último precio": fila["lastPrice"],
                "Cotización válida": valida, "Contrato": fila["contractSymbol"],
            })
    return pd.DataFrame(filas)


def sugerir(cands: pd.DataFrame, spread_max: float = 0.10) -> tuple[int, str]:
    """Índice del contrato sugerido y su justificación.

    Regla: entre los contratos con bid/ask válido y spread menor a `spread_max`, el de mayor
    interés abierto (más líquido); en empate, el vencimiento más cercano a 3 meses.
    """
    if cands.empty:
        raise ValueError("No hay candidatos.")
    validos = cands[cands["Cotización válida"] & (cands["Spread %"] < spread_max)]
    base = validos if not validos.empty else cands
    oi = base["Interés abierto"].fillna(0)
    mejores = base[oi == oi.max()]
    idx = int((mejores["Días"] - 91).abs().idxmin())
    c = cands.loc[idx]
    if not validos.empty:
        oi_txt = f"{int(c['Interés abierto']):,}".replace(",", ".")
        spread_txt = f"{c['Spread %'] * 100:.1f}".replace(".", ",")
        motivo = (f"Es el más líquido ({oi_txt} contratos abiertos) entre los que tienen bid/ask válido y un "
                  f"spread menor al {spread_max * 100:.0f} % (este tiene {spread_txt} %).")
    else:
        motivo = ("Ningún candidato tiene bid/ask válido con spread razonable en esta consulta; se sugiere "
                  "el de mayor interés abierto y la comparación con el mercado queda pendiente.")
    return idx, motivo


# ---------------------------------------------------------------------------
# Probabilidades de ejercicio y de beneficio (Paso 6, Ec. 5)
# ---------------------------------------------------------------------------
def probabilidades(s0: float, a: float, sigma: float, k: float, t_gbm: float, t_val: float, r: float,
                   tipo: str, prima: float, costos: float = 0.0, n: int = 20_000, semilla: int = 42) -> dict:
    """Bajo P (drift a), con S_T simulado por el GBM:
    - P(ejercicio): P(S_T > K) para call, P(S_T < K) para put.
    - P(beneficio): P(Π_T > 0), con Π_T = H(S_T) − A·e^{rT} − c_T   (Ec. 5),
      donde A es el ask pagado por acción y c_T los costos terminales.
    """
    rng = np.random.default_rng(semilla)
    z = rng.standard_normal(n)
    st = s0 * np.exp((a - 0.5 * sigma ** 2) * t_gbm + sigma * np.sqrt(t_gbm) * z)
    h = crr.payoff(st, k, tipo)
    costo_total = prima * np.exp(r * t_val) + costos
    pi = h - costo_total
    ejercicio = (st > k) if tipo == "call" else (st < k)
    # Probabilidad de ejercicio exacta bajo el GBM (para contrastar con la simulación)
    d2 = (np.log(s0 / k) + (a - 0.5 * sigma ** 2) * t_gbm) / (sigma * np.sqrt(t_gbm))
    exacta = norm.cdf(d2) if tipo == "call" else norm.cdf(-d2)
    equilibrio = k + costo_total if tipo == "call" else k - costo_total
    return {"p_ejercicio": float(ejercicio.mean()), "p_ejercicio_exacta": float(exacta),
            "p_beneficio": float((pi > 0).mean()), "punto_equilibrio": float(equilibrio),
            "costo_total": float(costo_total), "st": st, "pi": pi,
            "ganancia_media": float(pi.mean())}


def sensibilidad(s0, a, sigma, k, t_gbm, t_val, r, tipo, prima, delta=0.05, semilla=42) -> pd.DataFrame:
    """Paso 6: qué cambia en las probabilidades al mover a y σ en ±delta."""
    filas = []
    for etiqueta, a_i, s_i in [("Caso base", a, sigma), (f"a + {delta:.0%}", a + delta, sigma),
                               (f"a − {delta:.0%}", a - delta, sigma), (f"σ + {delta:.0%}", a, sigma + delta),
                               (f"σ − {delta:.0%}", a, max(sigma - delta, 0.01))]:
        p = probabilidades(s0, a_i, s_i, k, t_gbm, t_val, r, tipo, prima, semilla=semilla)
        filas.append({"Escenario": etiqueta, "Drift a": a_i, "Volatilidad σ": s_i,
                      "P(ejercicio)": p["p_ejercicio"], "P(beneficio)": p["p_beneficio"]})
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Comparación con el mercado (Paso 9, Ec. 8)
# ---------------------------------------------------------------------------
def comparar(v_crr: float, bid, ask) -> dict:
    """m = (bid + ask)/2,  δ = V_CRR − m,  δ_rel = δ / m.  Solo si 0 < bid ≤ ask."""
    if not cotizacion_valida(bid, ask):
        return {"valida": False}
    m = (bid + ask) / 2
    delta = v_crr - m
    posicion = "dentro" if bid <= v_crr <= ask else ("por encima" if v_crr > ask else "por debajo")
    return {"valida": True, "medio": m, "delta": delta, "delta_rel": delta / m, "posicion": posicion}
