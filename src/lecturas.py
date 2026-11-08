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
