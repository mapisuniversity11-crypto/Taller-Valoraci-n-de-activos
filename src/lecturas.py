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


# ---------------------------------------------------------------------------
# Riesgo (Paso 10)
# ---------------------------------------------------------------------------
def lectura_var(capital: float, alpha: float, h: int, var: float, es: float, peor: float, metodo: str) -> str:
    veces = round(alpha * 100)
    return (f"En {veces} de cada 100 períodos de {h} días, la cartera de {_usd(capital)} no debería perder más de "
            f"<b>{_usd(var)}</b> (método {metodo}). Si llega uno de esos períodos malos (el {100 - veces} % restante), "
            f"la pérdida promedio sería de <b>{_usd(es)}</b>. La peor pérdida real de {h} días en los datos fue de "
            f"<b>{_usd(peor)}</b>: el VaR no es la peor pérdida posible, es un umbral que se supera pocas veces.")


def lectura_metodos(hist: float, normal: float, mc: float) -> str:
    alto = max(("histórico", hist), ("normal", normal), ("Monte Carlo", mc), key=lambda x: x[1])
    return (f"Los tres métodos dan cifras del mismo orden; el más conservador es el <b>{alto[0]}</b>. El histórico usa "
            "las pérdidas que realmente ocurrieron, incluidos los días extremos (colas pesadas), por eso suele dar más. "
            "El normal supone una campana y media cero. El Monte Carlo simula el GBM con el crecimiento esperado de "
            "las acciones, que compensa parte de la pérdida.")


def lectura_excepciones(n_exc: int, n_dias: int, alpha: float, zona: str) -> str:
    esperadas = n_dias * (1 - alpha)
    texto = (f"En los últimos {n_dias} días hubo <b>{n_exc}</b> días en que la pérdida real superó el VaR diario; con "
             f"un VaR del {alpha:.0%} se esperaban unos {_num(esperadas, 1)}. Resultado: <b>zona {zona}</b> del "
             "semáforo de Basilea")
    if zona == "verde" and n_exc < esperadas / 2:
        texto += (": el VaR funcionó, e incluso fue algo conservador, porque el último año fue tranquilo para "
                  "esta cartera.")
    elif zona == "verde":
        texto += ": el VaR funcionó como se esperaba."
    else:
        texto += ": hubo más pérdidas grandes de las esperadas; el VaR subestimó el riesgo."
    return texto


def lectura_fuera_muestra(f, capital: float, fecha_inicio) -> str:
    return (f"Una cartera de {_usd(capital)} comprada el {fecha_inicio:%Y-%m-%d} con estos pesos valdría hoy "
            f"<b>{_usd(f.valor.iloc[-1])}</b> (<b>{_pct(f.rentabilidad)}</b>, después de pagar {_usd(f.costos)} de "
            f"costos de compra). Su peor caída desde un máximo fue de <b>{_pct(f.caida_maxima)}</b>.")


def lectura_estres(tabla, var: float) -> str:
    veces = tabla["Pérdida"] / var
    return (f"En estos escenarios de crisis la cartera perdería entre {_usd(tabla['Pérdida'].min())} y "
            f"{_usd(tabla['Pérdida'].max())}, es decir, entre {_num(veces.min(), 1)} y {_num(veces.max(), 1)} veces "
            "el VaR. Recuerdan que una crisis puede costar bastante más que un período malo normal.")


# ---------------------------------------------------------------------------
# Asignación (Pasos 11 y 12)
# ---------------------------------------------------------------------------
def _lista(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " y " + items[-1]


def lectura_asignacion(w_min, var_min: float, var_ig: float, var_inv: float, w_max: float) -> str:
    topes = [t for t in w_min.index if w_min[t] >= w_max - 1e-6]
    minimos = [t for t in w_min.index if w_min[t] <= 0.02 + 1e-6]
    texto = (f"La cartera de <b>mínimo VaR</b> tiene un VaR de <b>{_usd(var_min)}</b>, un "
             f"<b>{_pct(1 - var_min / var_ig, 0)}</b> menos que con pesos iguales ({_usd(var_ig)}). ")
    if topes:
        texto += f"Pone el máximo permitido en {_lista(topes)}, las acciones más estables o menos correlacionadas"
        texto += f", y el mínimo en {_lista(minimos)}. " if minimos else ". "
    texto += (f"La cartera inversa al VaR ({_usd(var_inv)}) también reduce el riesgo, pero como ignora las "
              "correlaciones no llega al mínimo.")
    return texto


def lectura_compra(tabla, capital: float, var_real: float) -> str:
    efectivo = capital - tabla["Monto invertido"].sum()
    return (f"Comprando acciones enteras se invierten <b>{_usd(tabla['Monto invertido'].sum())}</b> y quedan "
            f"<b>{_usd(efectivo)}</b> en efectivo. Con los pesos reales resultantes, el VaR de la cartera es de "
            f"{_usd(var_real)}.")


def lectura_correccion(pasos: list[str], max_sector: float) -> str:
    if not pasos:
        return (f"En la cartera inversa al VaR ningún sector superaba el {_pct(max_sector, 0)}, así que no hizo falta "
                "corregir la concentración sectorial: la cartera corregida es igual a la original.")
    return "Ajustes hechos: " + " ".join(pasos)


def lectura_presupuesto(p, v_max: float) -> str:
    if p.f >= 1:
        return (f"Con todo el capital invertido el VaR sería de {_usd(p.var_total)}, que ya está dentro del presupuesto "
                f"de {_usd(v_max)}: se invierte todo y no queda efectivo.")
    return (f"Con todo el capital invertido el VaR sería de {_usd(p.var_total)}, por encima del presupuesto de "
            f"{_usd(v_max)}. Por eso se invierte solo el <b>{_pct(p.f, 0)}</b> ({_usd(p.invertido.sum())}) y se dejan "
            f"<b>{_usd(p.efectivo)}</b> en efectivo sin riesgo; así el VaR baja a {_usd(p.var_ajustado)}.")


def lectura_estrategias(info: dict) -> str:
    mejor = max(info, key=lambda k: info[k]["rentabilidad"])
    menor_caida = max(info, key=lambda k: info[k]["caida_maxima"])
    mensual = info.get("Mínimo VaR · rebalanceo mensual", {})
    texto = (f"En el último año, la estrategia con mayor rentabilidad fue <b>{mejor}</b> "
             f"({_pct(info[mejor]['rentabilidad'])}) y la de menor caída fue <b>{menor_caida}</b> "
             f"({_pct(info[menor_caida]['caida_maxima'])}). ")
    if mensual:
        texto += (f"El mínimo VaR con rebalanceo mensual hizo {mensual['rebalanceos']} rebalanceos y pagó "
                  f"{_usd(mensual['costos'])} en costos. ")
    texto += ("Estas carteras buscan controlar el riesgo, no maximizar la ganancia: menos riesgo no garantiza más "
              "rentabilidad en un año particular.")
    return texto


# ---------------------------------------------------------------------------
# Evaluación del GBM (Paso 13)
# ---------------------------------------------------------------------------
def lectura_evaluacion(res, h: int) -> str:
    gana = int((res["¿Le gana al ingenuo?"] == "Sí").sum())
    mg, mi = res["MAPE GBM (%)"].mean(), res["MAPE ingenuo (%)"].mean()
    cob = res["Cobertura P10–P90"].mean()
    dias = "un día" if h == 1 else f"{h} días"
    return (f"El GBM le gana al pronóstico ingenuo en <b>{gana} de {len(res)}</b> acciones, pero por márgenes mínimos "
            f"(MAPE promedio de {_num(mg, 3)} % frente a {_num(mi, 3)} %). A {dias}, el pronóstico del GBM es casi igual "
            "al precio de hoy, porque el crecimiento esperado en tan poco tiempo es muy pequeño: ningún modelo adivina "
            "de forma consistente el precio exacto. Donde el GBM sí funciona es en el <b>rango</b>: el precio real cayó "
            f"dentro de la banda P10–P90 el <b>{_pct(cob, 0)}</b> de las veces, cerca del 80 % esperado.")


def lectura_evaluacion_accion(fila, ticker: str) -> str:
    return (f"En <b>{ticker}</b>, el GBM se equivoca en promedio por <b>{_usd(fila['RMSE GBM'])}</b> "
            f"({_num(fila['MAPE GBM (%)'], 2)} % del precio); el ingenuo, por {_usd(fila['RMSE ingenuo'])} "
            f"({_num(fila['MAPE ingenuo (%)'], 2)} %). El precio real quedó dentro de la banda el "
            f"{_pct(fila['Cobertura P10–P90'], 0)} de las veces.")


# ---------------------------------------------------------------------------
# Modelos adicionales: browniano estándar, Merton y browniano fraccional
# ---------------------------------------------------------------------------
INTRO_BROWNIANO = (
    "Es el modelo más simple de un precio que se mueve al azar: cada día la acción sube o baja una "
    "<b>cantidad fija de dólares</b>, sin importar cuánto valga. Lo usamos como punto de partida para "
    "responder una pregunta práctica: <b>¿por qué proyectamos con el GBM, que mide los movimientos en "
    "porcentaje, y no con este modelo?</b>")

INTRO_MERTON = (
    "El GBM supone que el precio se mueve de forma suave, día a día. En la realidad hay <b>saltos</b>: un "
    "reporte de resultados, una demanda o una noticia macro pueden mover la acción varios puntos en una "
    "sola sesión. El modelo de Merton agrega esos saltos al GBM y responde: <b>¿cuántos saltos hay al año, "
    "de qué tamaño, y cuánto más debería costar la opción por ese riesgo?</b>")

INTRO_FRACCIONAL = (
    "El GBM supone que lo que pasó ayer no dice nada sobre mañana: los retornos <b>no tienen memoria</b>. "
    "El browniano fraccional permite que sí la tengan, y su exponente de Hurst <b>H</b> lo mide. "
    "Pregunta financiera: <b>¿nuestras acciones siguen tendencias, tienden a devolverse, o se comportan "
    "como supone el GBM?</b> Esto importa porque el VaR a varios días se calcula con la regla de la raíz "
    "del tiempo, que solo es correcta si no hay memoria.")


def variables_browniano(cal, p0: float) -> list[tuple[str, str, str, str, str]]:
    return [
        ("P₀", "Precio inicial", _usd(p0), "Último cierre: el punto de partida de todas las trayectorias.",
         "Si sube, todo el abanico se desplaza hacia arriba, pero el ancho en dólares no cambia."),
        ("m", "Tendencia en USD", f"{_usd(cal.m)}/año", "Cuántos dólares gana en promedio la acción por año.",
         "Si sube, el centro del abanico crece en línea recta (no en curva, como en el GBM)."),
        ("s", "Volatilidad en USD", f"{_usd(cal.s)}/año", "Cuántos dólares se mueve la acción en un año típico.",
         "Si sube, el rango se ensancha; con s alto y horizonte largo aparecen precios negativos."),
    ]


def lectura_browniano(ticker: str, p_neg: dict, corr_usd: float, corr_pct: float, horizonte: str) -> str:
    anios = max(p_neg)
    texto = (f"En {horizonte}, el browniano estándar dibuja para <b>{ticker}</b> un abanico simétrico en dólares, mientras "
             "el GBM es más amplio hacia arriba que hacia abajo, porque una acción puede subir sin límite pero no "
             "caer por debajo de cero. ")
    if p_neg[anios] < 1e-5:
        texto += (f"Con estos parámetros, la probabilidad de un precio negativo es prácticamente cero incluso a {anios} "
                  "años, pero el modelo la permite, y para una acción es imposible. ")
    else:
        texto += (f"Con estos parámetros, la probabilidad de terminar con un precio negativo es <b>{_pct(p_neg[1], 3)}</b> "
                  f"a 1 año y <b>{_pct(p_neg[anios], 3)}</b> a {anios} años, algo imposible para una acción; el GBM "
                  "nunca lo produce. ")
    if corr_usd > 0.3 and abs(corr_pct) < corr_usd:
        texto += (f"La evidencia más fuerte está en los datos: cuando {ticker} valía más, también se movía más dólares "
                  f"por día (correlación {_num(corr_usd)}), mientras que su movimiento en porcentaje no dependía del "
                  f"precio (correlación {_num(corr_pct)}). Los movimientos son <b>proporcionales</b> al precio, como "
                  "supone el GBM.")
    else:
        texto += (f"En estos datos, el movimiento diario en dólares de {ticker} no crece claramente con el precio "
                  f"(correlación {_num(corr_usd)}); aun así, el GBM sigue siendo preferible porque nunca da precios "
                  "negativos.")
    return texto


def variables_merton(m, umbral_k: float) -> list[tuple[str, str, str, str, str]]:
    return [
        ("σ", "Volatilidad normal", _pct(m.sigma), "El movimiento de los días sin noticias fuertes.",
         "Si sube, la opción se encarece en todos los strikes, como en el GBM."),
        ("λ", "Saltos por año", _num(m.lam, 1),
         f"Cuántos días al año la acción se movió más de {_num(umbral_k, 0)} veces lo normal.",
         "Si sube, hay más probabilidad de un movimiento brusco antes del vencimiento y la opción vale más."),
        ("μ_J", "Tamaño medio del salto", _pct(m.mu_j, 2),
         "Si los saltos en promedio son hacia arriba (positivo) o hacia abajo (negativo).",
         "Si es negativo, los saltos son sobre todo caídas: encarece las puts más que las calls."),
        ("δ_J", "Variación del salto", _pct(m.delta_j, 2), "Qué tan distintos son los saltos entre sí.",
         "Si sube, los saltos pueden ser muy grandes en cualquier dirección: encarece calls y puts."),
    ]


def lectura_saltos(ticker: str, m, n_dias: int) -> str:
    sub = int((m.tamanos > 0).sum())
    baj = len(m.tamanos) - sub
    texto = (f"En los últimos {n_dias} días de negociación, <b>{ticker}</b> tuvo <b>{len(m.tamanos)} días de salto</b> "
             f"({sub} hacia arriba y {baj} hacia abajo), unos <b>{_num(m.lam, 1)} por año</b>. ")
    if len(m.tamanos):
        mayor = m.tamanos.abs().idxmax()
        texto += (f"El más fuerte fue el {mayor:%d/%m/%Y}, con un movimiento de {_pct(m.tamanos[mayor], 1)}. ")
    texto += (f"Si se separan esos días, la volatilidad de los días normales es {_pct(m.sigma)}; con los saltos, la "
              f"volatilidad total sube a {_pct(m.sigma_total)}. Esa diferencia es el riesgo que el GBM reparte de forma "
              "suave y que en realidad llega de golpe.")
    return texto


def lectura_merton(v_bs: float, v_mer: float, cmp, p_salto: float, tipo: str, dias: int) -> str:
    dif = v_mer - v_bs
    texto = (f"Hay un <b>{_pct(p_salto, 0)}</b> de probabilidad de que ocurra al menos un salto en los {dias} días que "
             f"faltan al vencimiento. Con saltos, la {tipo} europea vale <b>{_usd(v_mer)}</b> frente a {_usd(v_bs)} del "
             f"modelo sin saltos: {'una prima adicional' if dif >= 0 else 'un descuento'} de {_usd(abs(dif))} "
             "por acción. ")
    if cmp and cmp.get("valida"):
        medio = cmp["medio"]
        if abs(v_mer - medio) < abs(v_bs - medio):
            texto += (f"Merton queda <b>más cerca del precio del mercado</b> ({_usd(medio)}): parte de lo que el mercado "
                      "cobra de más sobre el GBM es el riesgo de saltos.")
        else:
            texto += (f"El modelo sin saltos queda más cerca del precio del mercado ({_usd(medio)}): en este contrato, "
                      "agregar saltos no mejora la valoración.")
    return texto


def variables_fraccional(h_sel: float, ticker: str) -> list[tuple[str, str, str, str, str]]:
    return [
        ("H", "Exponente de Hurst", _num(h_sel, 2) + f" ({ticker})",
         "Mide la memoria de los retornos. Con H = 0,5 no hay memoria y el GBM es correcto.",
         "Si sube por encima de 0,5, una subida tiende a seguir de otra (tendencia); si baja de 0,5, a una subida "
         "le sigue una caída (reversión)."),
        ("√h", "Regla de la raíz del tiempo", "VaR",
         "Para pasar del riesgo de 1 día al de h días se multiplica por √h. Equivale a suponer H = 0,5.",
         "Con H > 0,5 el riesgo a varios días crece más rápido que √h y el VaR queda corto; con H < 0,5, se "
         "sobreestima."),
    ]


def lectura_hurst(tabla, h_var: int) -> str:
    sin = tabla[tabla["Lectura"] == "Sin memoria"]["Ticker"].tolist()
    ten = tabla[tabla["Lectura"] == "Tendencia"]["Ticker"].tolist()
    rev = tabla[tabla["Lectura"] == "Reversión"]["Ticker"].tolist()
    texto = f"<b>{len(sin)} de {len(tabla)}</b> acciones tienen un H compatible con una serie sin memoria"
    texto += "." if not sin else f" ({_lista(sin)})." if len(sin) <= 4 else "."
    if ten:
        texto += f" {_lista(ten)} {'muestra' if len(ten) == 1 else 'muestran'} tendencia (H por encima de lo esperado)."
    if rev:
        texto += f" {_lista(rev)} {'muestra' if len(rev) == 1 else 'muestran'} algo de reversión (H por debajo)."
    if len(sin) == len(tabla):
        texto += (" Para toda la cartera, el supuesto del GBM de que los retornos no tienen memoria es razonable, y "
                  f"escalar el VaR con √{h_var} es adecuado.")
    elif len(sin) >= len(tabla) / 2:
        texto += (" Para la mayoría de la cartera, el supuesto del GBM de que los retornos no tienen memoria es "
                  f"razonable, y escalar el VaR con √{h_var} es adecuado.")
    else:
        texto += (" Varias acciones se apartan del supuesto del GBM: conviene revisar el escalamiento del VaR a "
                  "varios días.")
    return texto


def lectura_escalamiento(ticker: str, h: float, fac: float, h_var: int, v, veredicto: str = "") -> str:
    ult = v.iloc[-1]
    texto = (f"A {int(ult['Sesiones'])} sesiones, la volatilidad real de <b>{ticker}</b> fue {_pct(ult['Real'])}, frente a "
             f"{_pct(ult['Regla √m (GBM)'])} que predice la regla de la raíz del tiempo. ")
    if fac > 1:
        texto += f"Si el VaR a {h_var} días se escalara con H = {_num(h)}, sería un {_pct(fac - 1)} <b>mayor</b>."
    else:
        texto += f"Si el VaR a {h_var} días se escalara con H = {_num(h)}, sería un {_pct(1 - fac)} <b>menor</b>."
    if veredicto == "Sin memoria":
        texto += (" Pero esa diferencia está dentro de lo que produce el azar con esta cantidad de datos, así que no hay "
                  "razón para cambiar la regla √h que usa la pestaña Riesgo.")
    return texto


# ---------------------------------------------------------------------------
# Ornstein-Uhlenbeck · Vasicek
# ---------------------------------------------------------------------------
INTRO_OU = (
    "El GBM deja que el precio se aleje sin límite de donde está hoy. Algunas variables no se comportan así: "
    "<b>vuelven a un nivel normal</b>, como un resorte. Ornstein-Uhlenbeck mide <b>a qué nivel vuelven y qué tan "
    "rápido</b>; aplicado a la tasa de interés se llama modelo de <b>Vasicek</b>. Lo usamos para dos preguntas: "
    "<b>¿cuánto puede moverse la tasa del Tesoro durante la vida de la opción, y cuánto cambia la prima?</b> y "
    "<b>si dos acciones del mismo sector se separan, ¿vuelven a juntarse?</b>")


def _anios_o_dias(anios: float) -> str:
    if not np.isfinite(anios):
        return "no se cierra"
    if anios < 1:
        return f"{anios * 252:.0f} sesiones"
    return f"{_num(anios, 1)} años"


def variables_vasicek(ou, nombre: str, formato) -> list[tuple[str, str, str, str, str]]:
    """ou: modelos.OUCalibrado. formato: función que convierte un valor de la variable en texto."""
    return [
        ("θ", "Nivel de largo plazo", formato(ou.theta),
         f"El valor al que tiende a volver {nombre}.",
         "Si sube, todo el abanico futuro se desplaza hacia ese nivel más alto."),
        ("κ", "Velocidad de regreso", f"{_num(ou.kappa, 2)}/año",
         f"Qué tan rápido se cierra la distancia a θ. Vida media: {_anios_o_dias(ou.vida_media)}.",
         "Si sube, el regreso es más rápido y el rango futuro es más estrecho; con κ = 0 no hay regreso y es "
         "un camino al azar."),
        ("σ", "Volatilidad", f"{formato(ou.sigma)}/año",
         "El tamaño de los choques que lo alejan de su nivel normal.",
         "Si sube, el rango futuro se ensancha, pero (si κ > 0) no crece sin límite como en el GBM."),
    ]


def lectura_tasa(ou, r_hoy: float, rango: tuple[float, float], meses: int) -> str:
    bajo, alto = rango
    texto = (f"Hoy la tasa continua del Tesoro es <b>{_pct(r_hoy, 2)}</b>. Con su historia, Vasicek estima un nivel de "
             f"largo plazo de <b>{_pct(ou.theta, 2)}</b>")
    if ou.evidencia == "ninguna":
        texto += (", pero la prueba no encuentra evidencia de que la tasa regrese a un nivel fijo en esta muestra, así "
                  "que ese θ es <b>poco confiable</b>: la tasa del Tesoro sigue las decisiones de la Reserva Federal, que "
                  "pueden cambiar su nivel de forma duradera. Conviene leer el abanico como un rango de sensibilidad. ")
    else:
        texto += (f" y una vida media de {_anios_o_dias(ou.vida_media)}: la tasa {'sube' if ou.theta > r_hoy else 'baja'} "
                  f"poco a poco hacia ese nivel. ")
    texto += (f"En {meses} meses, el 80 % de los escenarios deja la tasa entre <b>{_pct(bajo, 2)}</b> y "
              f"<b>{_pct(alto, 2)}</b>.")
    return texto


def lectura_prima_tasa(primas: dict, prima_hoy: float, tipo: str, dias: int, r_bajo: float, r_alto: float) -> str:
    rango = abs(primas["alta"] - primas["baja"])
    rel = rango / prima_hoy if prima_hoy > 0 else 0.0
    sentido = ("sube cuando sube la tasa (una tasa alta abarata pagar el strike en el futuro)" if tipo == "call"
               else "baja cuando sube la tasa (el strike que se recibe vale menos hoy)")
    texto = (f"La {tipo} vale {_usd(prima_hoy)} con la tasa de hoy. Si la tasa promedio de los próximos {dias} días "
             f"queda entre {_pct(r_bajo, 2)} y {_pct(r_alto, 2)}, la prima se movería entre "
             f"<b>{_usd(min(primas.values()))}</b> y <b>{_usd(max(primas.values()))}</b>: un rango de {_usd(rango)} "
             f"({_pct(rel)} de la prima). La prima {sentido}. ")
    if rel < 0.05:
        texto += ("Es un efecto pequeño: en una opción de pocos meses, el riesgo de tasa importa mucho menos que el "
                  "precio de la acción o su volatilidad, así que usar una tasa fija en el árbol CRR es razonable.")
    else:
        texto += ("Es un efecto apreciable: conviene revisar la tasa usada en el árbol CRR si el vencimiento es largo.")
    return texto


def lectura_spread(a: str, b: str, ou, hoy: float, desde) -> str:
    """hoy: diferencia de log-rendimientos acumulados desde `desde`, ln(P_a/P_a0) − ln(P_b/P_b0)."""
    dif = np.expm1(hoy)
    texto = (f"Desde el {desde:%d/%m/%Y}, {a} acumula un rendimiento relativo <b>{_pct(abs(dif))} "
             f"{'mayor' if dif > 0 else 'menor'}</b> que {b}. Frente a su nivel de largo plazo, la diferencia de hoy está "
             f"a <b>{_num(ou.z_hoy, 1)} desviaciones</b>. ")
    if ou.evidencia == "fuerte":
        texto += (f"Hay evidencia clara de que la diferencia <b>vuelve</b> a su nivel: la mitad de una separación se cierra "
                  f"en unas <b>{_anios_o_dias(ou.vida_media)}</b>. ")
    elif ou.evidencia == "débil":
        texto += (f"Hay evidencia <b>débil</b> (solo al 10 % de significancia) de que la diferencia vuelve a su nivel, con "
                  f"una vida media de unas {_anios_o_dias(ou.vida_media)}. ")
    else:
        texto += ("No hay evidencia estadística de que la diferencia vuelva a un nivel fijo: las dos acciones pueden "
                  "separarse de forma duradera. ")
    if ou.evidencia != "ninguna" and abs(ou.z_hoy) >= 2:
        texto += (f"Como la separación de hoy es grande, el modelo espera que se cierre: {a if ou.z_hoy > 0 else b} "
                  "debería rendir menos que la otra en los próximos meses. ")
    if ou.evidencia != "ninguna":
        texto += ("Para la cartera, esto significa que las dos acciones del mismo sector diversifican poco entre sí: "
                  "si se separan, tienden a reencontrarse.")
    return texto


# ---------------------------------------------------------------------------
# Herding: comportamiento en manada
# ---------------------------------------------------------------------------
INTRO_HERDING = (
    "El VaR de la pestaña Riesgo usa las correlaciones históricas: supone que las acciones se acompañan tanto como "
    "lo hicieron en el pasado. Pero cuando hay miedo o euforia, los inversionistas <b>dejan de mirar su propia "
    "información y copian lo que hacen los demás</b>: todos venden (o compran) a la vez y las acciones se mueven "
    "juntas. El efecto manada responde: <b>¿hay señales de manada en nuestras acciones, y qué le pasa a nuestro "
    "riesgo cuando todos actúan igual?</b>")


def variables_herding(rho_hoy: float, rho_max: float, h_sel: float, g2: float) -> list[tuple[str, str, str, str, str]]:
    return [
        ("ρ̄", "Correlación promedio", _num(rho_hoy, 2),
         "Qué tanto se mueven juntas, en promedio, las 10 acciones (0 = cada una por su lado, 1 = todas igual).",
         f"Si sube, la diversificación se pierde. Su máximo en la historia fue {_num(rho_max, 2)}."),
        ("h", "Intensidad de manada", _num(h_sel, 2),
         "La parte del movimiento de cada acción que viene de copiar a la multitud y no de su propia información.",
         "Si sube, las correlaciones se acercan a 1 y el VaR de la cartera crece aunque ninguna acción sea más volátil."),
        ("γ₂", "Prueba CSAD", _num(g2, 2),
         "Mide si en los días de movimientos fuertes del mercado las acciones se parecen más de lo normal.",
         "Si es negativo y significativo, hay manada: en los días extremos todos hacen lo mismo."),
    ]


def lectura_correlacion_movil(serie, rho_cal: float, ventana: int) -> str:
    pico = serie.idxmax()
    valle = serie.idxmin()
    hoy = float(serie.iloc[-1])
    texto = (f"Con ventanas de {ventana} sesiones, la correlación promedio entre nuestras acciones fue en general "
             f"{_num(serie.median(), 2)}, pero no es estable: llegó a <b>{_num(serie.max(), 2)}</b> en la ventana que "
             f"termina el {pico:%d/%m/%Y} y bajó a {_num(serie.min(), 2)} el {valle:%d/%m/%Y}. Hoy está en "
             f"<b>{_num(hoy, 2)}</b>. ")
    texto += (f"El VaR de la pestaña Riesgo usa un promedio de {_num(rho_cal, 2)} (ventana de calibración). Cuando la "
              "correlación sube, la diversificación protege menos, y en los mercados eso suele ocurrir en momentos "
              "de estrés, justo cuando más se necesita.")
    return texto


def lectura_csad(c) -> str:
    texto = (f"Cada punto es un día: el movimiento promedio del mercado (eje horizontal) y qué tan distintas fueron "
             f"las acciones entre sí ese día (CSAD). La curva ajustada tiene γ₂ = {_num(c.g2, 2)} "
             f"(estadístico t = {_num(c.t_g2, 2)}). ")
    if c.hay_manada:
        texto += ("Es <b>negativo y significativo</b>: en los días de movimientos grandes, nuestras acciones se parecen "
                  "más de lo que deberían. Hay <b>evidencia de manada</b>.")
    elif c.g2 < 0:
        texto += ("Es negativo, pero no significativo: hay una señal débil de manada que no alcanza a ser concluyente.")
    else:
        texto += ("No es negativo: en los días de movimientos grandes, las acciones se separan <b>más</b>, no menos. "
                  "<b>No hay evidencia de manada</b> en el sentido de la prueba CSAD: en los días de movimientos "
                  "fuertes, nuestras acciones reaccionan de forma distinta entre sí. Aun así, la correlación sí sube en ciertos episodios, "
                  "así que vale la pena medir qué pasaría con el riesgo si los inversionistas se copiaran.")
    return texto


def lectura_var_manada(var0: float, var_h: float, var_pico: float, h_sel: float, h_pico: float, rho_pico: float,
                       var_max: float, capital: float, alpha: float, horizonte: int) -> str:
    texto = (f"Sin manada adicional (h = 0), el VaR normal al {_pct(alpha, 0)} y {horizonte} días es <b>{_usd(var0)}</b>, "
             "el mismo de la pestaña Riesgo. ")
    if h_sel > 0:
        texto += (f"Con h = {_num(h_sel, 2)}, sube a <b>{_usd(var_h)}</b> ({_pct(var_h / var0 - 1)} más). ")
    texto += (f"Si la correlación volviera a su pico histórico ({_num(rho_pico, 2)}, equivalente a h = {_num(h_pico, 2)}), "
              f"el VaR sería {_usd(var_pico)}. En el extremo, si todos se copiaran por completo (h = 1), sería "
              f"<b>{_usd(var_max)}</b> ({_pct(var_max / capital)} del capital): ya no habría diversificación y la cartera "
              "se comportaría como una sola acción. Ninguna volatilidad individual cambió; todo el aumento viene de que "
              "las acciones dejan de compensarse entre sí.")
    return texto


# ---------------------------------------------------------------------------
# Opciones reales
# ---------------------------------------------------------------------------
INTRO_OPCIONES_REALES = (
    "El VPN tradicional decide hoy, de una vez: invertir o no. Pero un proyecto real tiene <b>flexibilidad</b>: se "
    "puede esperar a ver cómo evoluciona el mercado, crecer si va bien o vender los activos si va mal. Esa "
    "flexibilidad funciona como una opción americana sobre el valor del proyecto y se valora con el <b>mismo árbol "
    "CRR</b> de la pestaña Opciones. Pregunta: <b>¿cuánto vale poder esperar, expandir o abandonar un proyecto "
    "petrolero con la volatilidad de nuestras acciones de energía?</b>")


def variables_opciones_reales(v0, inversion, sigma, r, delta, t, nombre_sigma) -> list[tuple[str, str, str, str, str]]:
    return [
        ("V₀", "Valor del proyecto", _usd(v0) + " M",
         "Valor presente de los flujos que generaría el proyecto. Hace el papel del precio de la acción.",
         "Si sube, el proyecto se vuelve más atractivo: vale más esperar para invertir y menos abandonar."),
        ("I", "Inversión", _usd(inversion) + " M", "Lo que cuesta hacer el proyecto. Hace el papel del strike.",
         "Si sube, baja el VPN; la opción de esperar gana importancia frente a invertir ya."),
        ("σ", "Volatilidad del proyecto", _pct(sigma),
         f"Incertidumbre sobre el valor del proyecto; se toma de {nombre_sigma}.",
         "Si sube, toda la flexibilidad vale más: hay más que ganar esperando a ver qué pasa."),
        ("δ", "Flujos que se pierden al esperar", _pct(delta),
         "Mientras se espera, el proyecto no produce. Hace el papel del dividendo q.",
         "Si sube, esperar cuesta más y conviene invertir antes."),
        ("r · T", "Tasa y plazo", f"{_pct(r, 2)} · {_num(t, 1)} años",
         "Tasa libre de riesgo del Tesoro y tiempo máximo para decidir.",
         "Si T sube, hay más tiempo para que se resuelva la incertidumbre y la flexibilidad vale más."),
    ]


def lectura_opciones_reales(res: dict, vpn: float) -> str:
    """res: {tipo: OpcionReal}."""
    esp = res["esperar"]
    texto = f"Invirtiendo hoy, el VPN estático es <b>{_usd(vpn)} M</b>"
    texto += (", así que con la regla tradicional el proyecto <b>se rechazaría</b>. " if vpn < 0
              else ", así que con la regla tradicional se aceptaría de inmediato. ")
    texto += (f"Con la opción de <b>esperar</b> hasta {_num(esp.tiempos[-1], 1)} años, el proyecto vale "
              f"<b>{_usd(esp.vpn_estrategico)} M</b>")
    if esp.decision[0][0] == "Invertir":
        texto += ": aun así conviene invertir hoy, porque esperar cuesta más (flujos perdidos) de lo que se gana. "
    else:
        texto += ": hoy lo óptimo es <b>esperar</b> y decidir con más información. "
    texto += (f"Ya hecho el proyecto, poder <b>expandirlo</b> vale {_usd(res['expandir'].opcion)} M, poder "
              f"<b>abandonarlo</b> vale {_usd(res['abandonar'].opcion)} M y tener las dos opciones vale "
              f"{_usd(res['expandir o abandonar'].opcion)} M, con lo que el VPN pasa de {_usd(vpn)} M a "
              f"<b>{_usd(res['expandir o abandonar'].vpn_estrategico)} M</b>. ")
    suma = res["expandir"].opcion + res["abandonar"].opcion
    if res["expandir o abandonar"].opcion < suma - 0.005:
        texto += ("Las dos juntas valen menos que la suma por separado, porque no se pueden usar a la vez: si el proyecto "
                  "va bien se expande y si va mal se abandona.")
    else:
        texto += ("Las dos juntas valen prácticamente la suma de cada una: se usan en escenarios opuestos (expandir si "
                  "va bien, abandonar si va mal), así que casi nunca compiten.")
    return texto


def lectura_mapa_decision(o, tipo: str) -> str:
    etiqueta = {"esperar": "Invertir", "expandir": "Expandir", "abandonar": "Abandonar"}.get(tipo)
    nodos = [(t, v, dd) for t, vs, ds in zip(o.tiempos, o.valores_v, o.decision) for v, dd in zip(vs, ds)]
    if tipo == "esperar":
        frontera = [min((v for t_, v, dd in nodos if t_ == t and dd == "Invertir"), default=None) for t in o.tiempos]
        f = [x for x in frontera if x is not None]
        if not f:
            return "Con estos valores nunca conviene invertir antes del plazo: la opción de esperar domina en todo el árbol."
        return (f"Cada punto es un posible valor futuro del proyecto. Al comienzo solo conviene invertir si el proyecto "
                f"llega a valer unos <b>{_usd(f[0])} M</b>; al final del plazo basta con que supere la inversión. Esa "
                "frontera desciende porque, a medida que se acaba el tiempo, esperar vale menos. Por encima de la "
                "frontera se invierte; por debajo, se espera.")
    if tipo == "expandir o abandonar":
        return ("Cada punto es un posible valor futuro del proyecto. En la zona alta conviene <b>expandir</b> y en la "
                "baja <b>abandonar</b>; en el medio se sigue operando sin cambios y se conserva la flexibilidad.")
    n_ej = sum(1 for _, _, dd in nodos if dd == etiqueta)
    return (f"Cada punto es un posible valor futuro del proyecto. En {n_ej} de los {len(nodos)} nodos del árbol conviene "
            f"<b>{etiqueta.lower()}</b>; en los demás, lo óptimo es seguir y mantener la opción abierta.")


def lectura_sensibilidad_or(tabla, sigmas_ref: dict) -> str:
    fila_baja, fila_alta = tabla.dropna().iloc[0], tabla.dropna().iloc[-1]
    texto = (f"Con volatilidad de {_pct(fila_baja['σ'], 0)}, la flexibilidad de esperar vale {_usd(fila_baja['esperar'])} M; "
             f"con {_pct(fila_alta['σ'], 0)}, <b>{_usd(fila_alta['esperar'])} M</b>. ")
    texto += ("A diferencia del VPN, que castiga la incertidumbre con una tasa de descuento más alta, aquí <b>más "
              "incertidumbre vale más</b>: la flexibilidad permite quedarse con lo bueno y evitar lo malo. ")
    if sigmas_ref:
        texto += "Las líneas verticales marcan " + _lista([f"{k} ({_pct(v)})" for k, v in sigmas_ref.items()]) + "."
    return texto
