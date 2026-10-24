"""Resúmenes e interpretaciones en lenguaje sencillo para la pestaña Datos.

Aquí solo hay cálculos descriptivos sobre los datos ya calibrados (no se usan
para valorar ni asignar) y las frases que los explican. Todo se recalcula con
los datos de cada ejecución, así que las frases se actualizan solas.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _pct(v: float, dec: int = 1) -> str:
    return f"{v * 100:.{dec}f} %".replace(".", ",")


def _num(v: float, dec: int = 2) -> str:
    return f"{v:.{dec}f}".replace(".", ",")


# ---------------------------------------------------------------------------
# Correlaciones
# ---------------------------------------------------------------------------
def orden_por_sector(tickers: list[str], sectores: dict[str, str]) -> list[str]:
    """Tickers agrupados por sector (en el orden en que aparece cada sector)."""
    orden_sect = list(dict.fromkeys(sectores[t] for t in tickers))
    return [t for s in orden_sect for t in tickers if sectores[t] == s]


def pares_correlacion(corr: pd.DataFrame, sectores: dict[str, str], n: int = 3) -> pd.DataFrame:
    """Los n pares más correlacionados y los n menos correlacionados."""
    tick = list(corr.columns)
    filas = [
        {"Par": f"{a} – {b}", "Correlación": corr.loc[a, b],
         "Sectores": sectores[a] if sectores[a] == sectores[b] else f"{sectores[a]} / {sectores[b]}"}
        for i, a in enumerate(tick) for b in tick[i + 1:]
    ]
    df = pd.DataFrame(filas).sort_values("Correlación", ascending=False)
    altos = df.head(n).assign(Grupo="Más correlacionados")
    bajos = df.tail(n).sort_values("Correlación").assign(Grupo="Menos correlacionados")
    return pd.concat([altos, bajos])[["Grupo", "Par", "Sectores", "Correlación"]].reset_index(drop=True)


def correlacion_media_por_grupo(corr: pd.DataFrame, sectores: dict[str, str]) -> tuple[float, float]:
    """(correlación promedio entre acciones del mismo sector, entre sectores distintos)."""
    tick = list(corr.columns)
    mismo, distinto = [], []
    for i, a in enumerate(tick):
        for b in tick[i + 1:]:
            (mismo if sectores[a] == sectores[b] else distinto).append(corr.loc[a, b])
    return (float(np.mean(mismo)) if mismo else float("nan"),
            float(np.mean(distinto)) if distinto else float("nan"))


def lectura_correlaciones(corr: pd.DataFrame, sectores: dict[str, str]) -> str:
    m, d = correlacion_media_por_grupo(corr, sectores)
    pares = pares_correlacion(corr, sectores, n=1)
    alto, bajo = pares.iloc[0], pares.iloc[1]
    texto = (f"Las acciones del mismo sector tienen una correlación promedio de <b>{_num(m)}</b>; "
             f"entre sectores distintos es de <b>{_num(d)}</b>. ")
    if m > d:
        texto += ("Las del mismo sector tienden a moverse juntas, así que repartir el capital entre "
                  "sectores reduce el riesgo de la cartera. ")
    texto += (f"El par que más se mueve junto es <b>{alto['Par']}</b> ({_num(alto['Correlación'])}) "
              f"y el que menos, <b>{bajo['Par']}</b> ({_num(bajo['Correlación'])}).")
    return texto


# ---------------------------------------------------------------------------
# Distribuciones de los log-retornos
# ---------------------------------------------------------------------------
def resumen_distribucion(x: pd.DataFrame, umbral: float = 3.0) -> pd.DataFrame:
    """Mejor y peor día, y días extremos (|x − x̄| > umbral · s) por acción.

    Bajo una distribución normal, la probabilidad de superar 3 desviaciones
    estándar es 0,27 %; si hay muchos más días así, las colas son más pesadas
    que las que supone el GBM.
    """
    media, s = x.mean(), x.std(ddof=1)
    extremos = ((x - media).abs() > umbral * s).sum()
    esperado = len(x) * 0.0027 if umbral == 3.0 else np.nan
    return pd.DataFrame({
        "Mejor día": x.max(),
        "Fecha mejor día": [d.date() for d in x.idxmax()],
        "Peor día": x.min(),
        "Fecha peor día": [d.date() for d in x.idxmin()],
        "Días extremos (>3σ)": extremos.astype(int),
        "Esperados si fuera normal": esperado,
    })


def lectura_dispersion(x: pd.DataFrame) -> str:
    s = x.std(ddof=1)
    alta, baja = s.idxmax(), s.idxmin()
    return (f"<b>{alta}</b> es la acción con movimientos diarios más amplios (desviación diaria de "
            f"{_pct(s[alta], 2)}) y <b>{baja}</b> la más estable ({_pct(s[baja], 2)}). "
            "Cada caja abarca la mitad central de los días; los puntos sueltos son días atípicos.")


def lectura_histograma(x: pd.Series, ticker: str) -> str:
    media, s = x.mean(), x.std(ddof=1)
    extremos = int(((x - media).abs() > 3 * s).sum())
    esperados = len(x) * 0.0027
    texto = (f"En {len(x)} días, <b>{ticker}</b> tuvo <b>{extremos}</b> movimientos mayores a 3 desviaciones "
             f"estándar; con una distribución normal se esperarían unos {_num(esperados, 1)}. ")
    if extremos > 1.5 * esperados:
        texto += ("Los días extremos son más frecuentes de lo que supone el GBM (colas pesadas): "
                  "el modelo tiende a subestimar la probabilidad de movimientos fuertes.")
    else:
        texto += "La curva normal, que es lo que supone el GBM, describe razonablemente bien estos datos."
    return texto


# ---------------------------------------------------------------------------
# Riesgo vs. retorno
# ---------------------------------------------------------------------------
def lectura_riesgo_retorno(parametros: pd.DataFrame) -> str:
    sig, mu = parametros["σ anual"], parametros["μ total anual"]
    return (f"En la ventana de calibración, <b>{mu.idxmax()}</b> tuvo el mayor retorno total ({_pct(mu.max())} anual) "
            f"y <b>{mu.idxmin()}</b> el menor ({_pct(mu.min())}). La más riesgosa fue <b>{sig.idxmax()}</b> "
            f"(volatilidad de {_pct(sig.max())}) y la menos, <b>{sig.idxmin()}</b> ({_pct(sig.min())}). "
            "Son datos históricos, no pronósticos: más volatilidad no garantiza más retorno.")


# ---------------------------------------------------------------------------
# GBM (Paso 4)
# ---------------------------------------------------------------------------
def _usd(v: float) -> str:
    return f"USD {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def lectura_gbm(ticker: str, horizonte: str, r) -> str:
    """r: gbm.ResumenGBM."""
    bajo, alto = r.rango
    return (f"En <b>{horizonte}</b>, según el modelo, <b>{ticker}</b> tiene un 80 % de probabilidad de "
            f"terminar entre <b>{_usd(bajo)}</b> y <b>{_usd(alto)}</b>. El precio más probable (la mediana) es "
            f"<b>{_usd(r.mediana)}</b>, frente a {_usd(r.s0)} de hoy, y la probabilidad de terminar por encima "
            f"del precio actual es de <b>{_pct(r.prob_subir, 0)}</b>. El modelo supone que la acción seguirá "
            "creciendo y moviéndose como en los últimos dos años.")


def comprobaciones_gbm(r) -> list[tuple[bool, str]]:
    """Lista de chequeo del Paso 4: (cumple, texto)."""
    return [
        (r.todos_positivos, "Todos los precios simulados son positivos."),
        (r.media_ok,
         f"La media simulada ({_usd(r.media_sim)}) coincide con la teórica S₀·e^(aT) ({_usd(r.media_teorica)}): "
         f"la diferencia es de {_num(abs(r.z))} errores estándar "
         f"({'menos de 2, aceptable' if r.media_ok else 'más de 2, revisar'})."),
    ]


# ---------------------------------------------------------------------------
# Opciones (Pasos 5–9)
# ---------------------------------------------------------------------------
def lectura_crr(europea: float, americana: float, tipo: str) -> str:
    dif = americana - europea
    if dif > 0.005:
        return (f"La {tipo} americana vale <b>{_usd(americana)}</b> y la europea <b>{_usd(europea)}</b>: la diferencia "
                f"({_usd(dif)}) es lo que vale el derecho a ejercerla antes del vencimiento.")
    return (f"La {tipo} americana y la europea valen prácticamente lo mismo (<b>{_usd(americana)}</b>): en este caso "
            "no conviene ejercer antes del vencimiento, así que ese derecho extra no agrega valor.")


def lectura_contrato(c, p: dict, tipo: str, ticker: str) -> str:
    """c: fila del candidato elegido; p: resultado de opciones.probabilidades."""
    lado = "por encima" if tipo == "call" else "por debajo"
    return (f"Comprar esta {tipo} cuesta <b>{_usd(c['Ask'])}</b> por acción ({_usd(c['Ask'] * 100)} por contrato de "
            f"100 acciones). Según el modelo, hay <b>{_pct(p['p_ejercicio'], 0)}</b> de probabilidad de que se ejerza "
            f"({ticker} {lado} de {_usd(c['Strike'])} al vencimiento), pero solo <b>{_pct(p['p_beneficio'], 0)}</b> de "
            f"que deje ganancia: para ganar, {ticker} debe terminar {lado} de <b>{_usd(p['punto_equilibrio'])}</b>, "
            "porque además hay que recuperar lo que se pagó por la opción.")


def lectura_sensibilidad(tabla) -> str:
    base = tabla.iloc[0]["P(beneficio)"]
    a_up = tabla.iloc[1]["P(beneficio)"]
    s_up = tabla.iloc[3]["P(beneficio)"]
    return (f"Con los parámetros calibrados, la probabilidad de ganancia es de <b>{_pct(base, 0)}</b>. Si el drift "
            f"<b>a</b> fuera 5 puntos mayor, subiría a <b>{_pct(a_up, 0)}</b>; si la volatilidad <b>σ</b> fuera 5 "
            f"puntos mayor, sería de <b>{_pct(s_up, 0)}</b>. El resultado depende bastante de supuestos que no "
            "conocemos con certeza.")


def lectura_mercado(cmp: dict, v_am: float, bid, ask, vol_impl, sigma: float) -> str:
    if not cmp.get("valida"):
        return ("<b>Comparación pendiente:</b> en esta consulta el bid/ask del contrato no es utilizable (por ejemplo, "
                "en cero porque el mercado está cerrado). No se reemplaza por otro dato: vuelva a ejecutar el análisis "
                "en horario de bolsa o elija otro contrato.")
    texto = (f"El precio teórico del árbol (<b>{_usd(v_am)}</b>) está <b>{cmp['posicion']}</b> del rango del mercado "
             f"({_usd(bid)} – {_usd(ask)}); la diferencia frente al precio medio es de {_usd(cmp['delta'])} "
             f"({_pct(cmp['delta_rel'])}). ")
    if abs(cmp["delta_rel"]) <= 0.05:
        texto += "Es una diferencia pequeña: el modelo queda muy cerca del mercado. "
    mercado_mas_caro = cmp["delta"] < 0
    if pd.notna(vol_impl) and vol_impl > 0:
        mayor, menor = vol_impl > sigma * 1.05, vol_impl < sigma * 0.95
        if mayor and mercado_mas_caro:
            texto += (f"El mercado usa una volatilidad implícita de {_pct(vol_impl)}, mayor que la histórica del modelo "
                      f"({_pct(sigma)}): espera más movimiento, y por eso la opción cotiza más cara que el modelo. ")
        elif menor and not mercado_mas_caro:
            texto += (f"El mercado usa una volatilidad implícita de {_pct(vol_impl)}, menor que la histórica del modelo "
                      f"({_pct(sigma)}): espera menos movimiento, y por eso la opción cotiza más barata que el modelo. ")
        else:
            comp = "mayor que" if mayor else ("menor que" if menor else "parecida a")
            texto += (f"La volatilidad implícita del mercado ({_pct(vol_impl)}) es {comp} la histórica del modelo "
                      f"({_pct(sigma)}), así que la volatilidad no explica por sí sola esta diferencia. ")
    texto += ("Otras causas posibles: la tasa de interés y los dividendos esperados, el estilo americano, la liquidez y "
              "que las cotizaciones pueden estar diferidas. Una diferencia por sí sola no indica una oportunidad de "
              "arbitraje: comprar y vender tiene costos.")
    return texto
