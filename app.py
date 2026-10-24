"""App del Taller de Valoración de Activos (Streamlit).

Ejecutar:  streamlit run app.py

Esta capa SOLO maneja interfaz: lee entradas, llama a los módulos de src/
y muestra resultados. Ningún cálculo financiero debe vivir aquí.
"""
import time
from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

from src import calibracion, config, datos, ui

st.set_page_config(page_title="Quantia", layout="wide", initial_sidebar_state="collapsed")

# Portada de bienvenida: solo la primera vez que se abre la app en la sesión
if config.MOSTRAR_INTRO and not st.session_state.get("intro_vista"):
    if config.ESTILO_INTRO == "metal":
        # Cintas de oro líquido (WebGL). El CSS estira el recuadro a pantalla completa;
        # si no aplicara, el recuadro queda de 1 px y no tapa nada.
        st.markdown(ui.css_portada_metal(config.DURACION_INTRO), unsafe_allow_html=True)
        portada = ui.html_portada_metal(duracion=config.DURACION_INTRO)
        if hasattr(st, "iframe"):          # Streamlit reciente
            st.iframe(portada, height=1)
        else:                              # versiones anteriores
            components.html(portada, height=1)
    else:
        st.markdown(ui.html_intro(duracion=config.DURACION_INTRO, fuente=config.FUENTE_INTRO),
                    unsafe_allow_html=True)
    st.session_state["intro_vista"] = True
    # En esta primera pasada SOLO se dibuja la portada: nada de la app existe todavía,
    # así que no se puede asomar por debajo. Al terminar la animación se recarga la
    # app (st.rerun) y ya se dibuja completa, sin la portada.
    time.sleep(config.DURACION_INTRO + 0.2)
    st.rerun()

# Tema negro y dorado para toda la app
st.markdown(ui.css_global(), unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Dos fases:
#   1. "entradas": solo el formulario, a pantalla completa.
#   2. "resultados": los cálculos; el formulario pasa al panel lateral, que se
#      abre con la bolita dorada de la esquina inferior derecha.
# Los controles tienen `key`, así que conservan sus valores al cambiar de lugar.
# ---------------------------------------------------------------------------
res = st.session_state.get("res")
en_resultados = res is not None
COLUMNAS = 1 if en_resultados else 2      # en el panel lateral, campos en una columna

hoy = date.today()
c = config.CONTROL_CRR


def _pares():
    """Dos columnas en la pantalla de entradas; una sola dentro del panel lateral."""
    return st.columns(2) if COLUMNAS == 2 else (st.container(), st.container())


def seccion_universo():
    universo = st.data_editor(
        pd.DataFrame({
            "Ticker": list(config.TICKERS_SECTORES),
            "Sector": list(config.TICKERS_SECTORES.values()),
            "q": [0.0] * len(config.TICKERS_SECTORES),
        }),
        key="universo", num_rows="fixed", hide_index=True, width="stretch",
        column_config={
            "q": st.column_config.NumberColumn(
                "q manual", help="Valor de respaldo (decimal, continuo anual). Si está activo el cálculo "
                "con Yahoo, la app usa los dividendos reales.", min_value=0.0, max_value=0.5,
                step=0.001, format="%.4f"),
        },
    )
    st.markdown('<div class="q-nota">Puede corregir el sector si falla el metadato. La columna q manual '
                'solo se usa si no se calculan los dividendos con Yahoo.</div>', unsafe_allow_html=True)
    return universo


def seccion_periodo():
    a, b = _pares()
    with a:
        fecha_inicio = st.date_input("Fecha inicial", hoy - timedelta(days=int(365.25 * config.ANIOS_HISTORIA)),
                                     key="fecha_inicio")
    with b:
        fecha_fin = st.date_input("Último día con datos", hoy, key="fecha_fin")
    fuente = st.selectbox("Fuente", ["Yahoo Finance", "Respaldo del repositorio", "Subir CSV"], key="fuente",
                          help="Si Yahoo falla, use el respaldo guardado en data/raw o suba sus propios CSV.")
    csv_aj = csv_sin = None
    procedencia = ""
    if fuente == "Subir CSV":
        csv_aj = st.file_uploader("CSV precios ajustados", type="csv", key="csv_aj")
        csv_sin = st.file_uploader("CSV precios sin ajustar", type="csv", key="csv_sin")
        procedencia = st.text_input("Procedencia (fuente y fecha de descarga)", key="procedencia")
    guardar = fuente == "Yahoo Finance" and st.checkbox("Guardar descarga como respaldo en data/raw",
                                                        value=True, key="guardar")
    return dict(fecha_inicio=fecha_inicio, fecha_fin=fecha_fin, fuente=fuente, csv_aj=csv_aj,
                csv_sin=csv_sin, procedencia=procedencia, guardar=guardar)


def seccion_cartera():
    a, b = _pares()
    with a:
        capital = st.number_input("Capital (USD)", value=config.CAPITAL, step=1_000.0, key="capital")
        h_var = st.number_input("Horizonte VaR (sesiones)", value=config.HORIZONTE_VAR, min_value=1, key="h_var")
    with b:
        alpha = st.number_input("Confianza VaR", value=config.ALPHA, min_value=0.90, max_value=0.999,
                                format="%.3f", key="alpha")
        semilla = st.number_input("Semilla", value=config.SEMILLA, step=1, key="semilla")
    auto_q_r = st.checkbox("Calcular q y r con datos de Yahoo (recomendado)", value=True, key="auto_q_r",
                           help="q: dividendos de 12 meses / precio sin ajustar. r: letra del Tesoro a 13 "
                                "semanas (^IRX). Si falla, se usan los valores manuales.")
    r = st.number_input("Tasa libre de riesgo r manual (decimal, anual)", value=config.TASA_LIBRE_RIESGO,
                        format="%.4f", key="r_manual")
    return dict(capital=capital, alpha=alpha, h_var=h_var, semilla=semilla, auto_q_r=auto_q_r, r=r)


def seccion_opcion(tickers):
    a, b = _pares()
    with a:
        ticker_opcion = st.selectbox("Subyacente", tickers, key="ticker_opcion",
                                     index=tickers.index(config.TICKER_OPCION) if config.TICKER_OPCION in tickers else 0)
        tipo = st.radio("Tipo", ["call", "put"], horizontal=True, key="tipo_opcion")
    with b:
        n_pasos = st.number_input("Pasos del árbol N", value=config.N_PASOS_CRR, min_value=1, key="n_pasos")
        estilo = st.radio("Estilo", ["europea", "americana"], horizontal=True, key="estilo_opcion")
    st.markdown('<div class="q-nota">Árbol CRR · entrega 1: valores del caso de control. En la semana 2, '
                'K y T vendrán del contrato real y S0 del cierre sin ajustar.</div>', unsafe_allow_html=True)
    a, b = _pares()
    with a:
        s0 = st.number_input("Spot S0", value=c["S0"], min_value=0.01, key="crr_s0")
        t = st.number_input("Vencimiento T (años)", value=c["T"], min_value=0.0, format="%.4f", key="crr_t")
        sigma = st.number_input("Volatilidad σ (decimal)", value=c["sigma"], min_value=0.0001, format="%.4f",
                                key="crr_sigma")
    with b:
        k = st.number_input("Strike K", value=c["K"], min_value=0.01, key="crr_k")
        r_arbol = st.number_input("Tasa r del árbol (decimal)", value=c["r"], format="%.4f", key="crr_r")
        q_arbol = st.number_input("Dividendo q del subyacente", value=c["q"], min_value=0.0, format="%.4f",
                                  key="crr_q")
    return dict(ticker=ticker_opcion, tipo=tipo, estilo=estilo, n=n_pasos, s0=s0, k=k, t=t,
                r=r_arbol, sigma=sigma, q=q_arbol)


def seccion_estrategias():
    a, b = _pares()
    with a:
        presupuesto_v = st.number_input("Presupuesto de riesgo V (USD)", value=config.PRESUPUESTO_RIESGO,
                                        min_value=0.0, step=500.0, key="presupuesto_v",
                                        help="VaR máximo permitido en la estrategia de presupuesto de riesgo (Paso 12).")
    with b:
        h_rolling = st.number_input("Horizonte H del rolling (sesiones)", value=config.H_ROLLING, min_value=1,
                                    key="h_rolling", help="Sesiones que se pronostican en el rolling del GBM (Paso 13).")
    return dict(presupuesto_v=presupuesto_v, h_rolling=h_rolling)


# ---------------------------------------------------------------------------
# Paso 1. Entradas: pantalla completa (fase 1) o panel lateral (fase 2)
# ---------------------------------------------------------------------------
if not en_resultados:
    st.markdown(ui.html_franja_entradas(), unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1.05, 1, 1], gap="medium")
    with col1:
        with st.container(key="tarjeta_universo"):
            st.markdown(ui.titulo_tarjeta("Acciones y sectores"), unsafe_allow_html=True)
            universo = seccion_universo()
    tickers = list(universo["Ticker"])
    with col2:
        with st.container(key="tarjeta_periodo"):
            st.markdown(ui.titulo_tarjeta("Período de datos"), unsafe_allow_html=True)
            periodo = seccion_periodo()
        with st.container(key="tarjeta_cartera"):
            st.markdown(ui.titulo_tarjeta("Cartera y riesgo"), unsafe_allow_html=True)
            cartera = seccion_cartera()
    with col3:
        with st.container(key="tarjeta_opcion"):
            st.markdown(ui.titulo_tarjeta("Opción"), unsafe_allow_html=True)
            opcion = seccion_opcion(tickers)
        with st.container(key="tarjeta_estrategias"):
            st.markdown(ui.titulo_tarjeta("Estrategias"), unsafe_allow_html=True)
            estrategias = seccion_estrategias()
    pie_izq, pie_der = st.columns([3, 1])
    with pie_izq:
        n_sect = len(set(universo["Sector"]))
        st.markdown(f'<div class="q-resumen">{len(tickers)} acciones · {n_sect} sectores · '
                    f'USD {cartera["capital"]:,.0f}</div>'.replace(",", "."), unsafe_allow_html=True)
    with pie_der:
        ejecutar = st.button("◆  Ejecutar análisis", type="primary", width="stretch", key="btn_ejecutar")
else:
    with st.sidebar:
        st.markdown('<div class="q-panel-cab"><div class="q-eyebrow">Entradas</div>'
                    '<div class="t">Ajusta y vuelve a calcular</div></div>', unsafe_allow_html=True)
        ejecutar = st.button("◆  Volver a ejecutar", type="primary", width="stretch", key="btn_reejecutar")
        with st.expander("Acciones y sectores"):
            universo = seccion_universo()
        tickers = list(universo["Ticker"])
        with st.expander("Período de datos", expanded=True):
            periodo = seccion_periodo()
        with st.expander("Cartera y riesgo", expanded=True):
            cartera = seccion_cartera()
        with st.expander("Opción"):
            opcion = seccion_opcion(tickers)
        with st.expander("Estrategias"):
            estrategias = seccion_estrategias()

# ---------------------------------------------------------------------------
# Ejecución: obtiene datos y calcula; los resultados quedan en session_state
# ---------------------------------------------------------------------------
sectores = dict(zip(universo["Ticker"], universo["Sector"]))
q_manual = pd.Series(universo["q"].to_numpy(), index=tickers, dtype=float).fillna(0.0)
fecha_inicio, fecha_fin, fuente = periodo["fecha_inicio"], periodo["fecha_fin"], periodo["fuente"]
auto_q_r, r = cartera["auto_q_r"], cartera["r"]


def obtener_crudos():
    if fuente == "Yahoo Finance":
        crudos = datos.descargar_yahoo(tickers, fecha_inicio, fecha_fin)
        crudos.meta["sectores_yahoo"] = datos.sectores_yahoo(tickers)
        if auto_q_r:
            crudos.meta["dividendos"] = datos.dividendos_yahoo(tickers, crudos.sin_ajustar)
            crudos.meta["tasa_libre_riesgo"] = datos.tasa_libre_riesgo_yahoo(fecha_fin)
        if periodo["guardar"]:
            datos.guardar_respaldo(crudos)
        return crudos
    if fuente == "Respaldo del repositorio":
        if not datos.existe_respaldo():
            raise RuntimeError("No hay respaldo en data/raw. Descargue primero desde Yahoo Finance.")
        return datos.cargar_respaldo()
    if periodo["csv_aj"] is None or periodo["csv_sin"] is None:
        raise RuntimeError("Suba los dos archivos CSV (ajustados y sin ajustar).")
    return datos.cargar_csv(periodo["csv_aj"], periodo["csv_sin"], periodo["procedencia"])


def elegir_q_y_r(meta: dict) -> tuple[pd.Series, pd.Series, float, str]:
    """q y r que se usan: los de Yahoo si se pidieron y están disponibles; si no, los manuales."""
    divs = meta.get("dividendos", {}) if auto_q_r else {}
    q_usado = q_manual.copy()
    fuente_q = pd.Series("manual", index=tickers)
    for t in tickers:
        if divs.get(t, {}).get("q") is not None:
            q_usado[t] = divs[t]["q"]
            fuente_q[t] = "Yahoo (dividendos 12 meses)"
    tasa = meta.get("tasa_libre_riesgo") if auto_q_r else None
    if tasa:
        return q_usado, fuente_q, tasa["r"], f"{tasa['fuente']}, {tasa['fecha']}: {tasa['irx_porcentaje']:.2f} %"
    return q_usado, fuente_q, float(r), "manual"


if ejecutar:
    try:
        with st.spinner("Obteniendo y validando datos..."):
            crudos = obtener_crudos()
            limpio = datos.limpiar(crudos, tickers)
            ventanas = datos.separar_ventanas(limpio.ajustados, config.N_CALIBRACION, config.N_EVALUACION)
            precios_cal = ventanas["calibracion"]
            q_usado, fuente_q, r_usado, fuente_r = elegir_q_y_r(crudos.meta)
            st.session_state["res"] = {
                "crudos": crudos,
                "limpio": limpio,
                "ventanas": ventanas,
                "tickers": tickers,
                "q": q_usado, "fuente_q": fuente_q, "r": r_usado, "fuente_r": fuente_r,
                "parametros": calibracion.calibrar(precios_cal, q_usado, config.D),
                "cov_diaria": calibracion.covarianza_diaria(precios_cal, "simple"),
                "corr": calibracion.correlacion(precios_cal),
                "sectores": sectores,
            }
        st.session_state.pop("error", None)
        calculo_ok = True
    except Exception as e:
        st.session_state["error"] = f"No se pudo completar el análisis: {e}"
        calculo_ok = False
    if calculo_ok:
        st.rerun()   # pasa a la fase de resultados (o los actualiza); va fuera del try a propósito

if st.session_state.get("error"):
    st.error(st.session_state["error"])

if not en_resultados:
    st.stop()

# ---------------------------------------------------------------------------
# Salidas
# ---------------------------------------------------------------------------
limpio, ventanas, meta = res["limpio"], res["ventanas"], res["crudos"].meta
tickers = res.get("tickers", tickers)
resumen = limpio.resumen
consulta = str(meta.get("consulta", meta.get("cargado_en", "")))[:10]
st.markdown(ui.html_franja_resultados([
    f"{meta.get('fuente', 'Datos')} · {consulta}",
    f"r = {res['r']:.2%}".replace(".", ","),
    f"{resumen['retornos_conjuntos']:,} retornos".replace(",", "."),
]), unsafe_allow_html=True)

tabs = st.tabs(["Datos", "GBM", "Opciones", "Riesgo", "Asignación", "Evaluación"])

with tabs[0]:
    cal, ev = ventanas["calibracion"], ventanas["evaluacion"]
    st.markdown(ui.html_kpis([
        ("Retornos conjuntos", f"{resumen['retornos_conjuntos']:,}".replace(",", "."),
         f"mínimo requerido {config.MIN_RETORNOS}"),
        ("Sesiones descartadas", str(resumen["sesiones_descartadas"]),
         "sin faltantes ni duplicados" if resumen["sesiones_descartadas"] == 0 and resumen["duplicados_eliminados"] == 0
         else f"{resumen['duplicados_eliminados']} duplicados eliminados"),
        ("Calibración · evaluación", f"{len(cal) - 1} · {len(ev) - 1}",
         f"{ui.mes_anio(cal.index[1])} – {ui.mes_anio(ev.index[-1])}"),
        ("Tasa libre de riesgo", f"{res['r']:.2%}".replace(".", ","),
         "^IRX 13 semanas, continua" if res["fuente_r"] != "manual" else "valor manual"),
    ]), unsafe_allow_html=True)

    g1, g2 = st.columns([1.5, 1], gap="medium")
    with g1:
        with st.container(key="tarjeta_g_precios"):
            st.markdown(ui.titulo_tarjeta("Precios normalizados · base 100"), unsafe_allow_html=True)
            norm = calibracion.normalizar(limpio.ajustados)
            st.plotly_chart(ui.estilo_fig(px.line(norm, labels={"value": "Índice", "Fecha": "", "Ticker": ""}), 380),
                            width="stretch")
    with g2:
        with st.container(key="tarjeta_g_corr"):
            st.markdown(ui.titulo_tarjeta("Correlaciones · log-retornos"), unsafe_allow_html=True)
            fig_c = px.imshow(res["corr"], text_auto=".2f", zmin=-1, zmax=1, aspect="auto",
                              color_continuous_scale=ui.ESCALA_CORR)
            fig_c.update_layout(coloraxis_showscale=False)
            fig_c.update_traces(textfont=dict(size=10, color=ui.TEXTO))
            st.plotly_chart(ui.estilo_fig(fig_c, 380), width="stretch")

    with st.container(key="tarjeta_universo_res"):
        st.markdown(ui.titulo_tarjeta("Universo y procedencia"), unsafe_allow_html=True)
        sect_yahoo = meta.get("sectores_yahoo", {})
        st.dataframe(
            pd.DataFrame({
                "Ticker": tickers,
                "Sector usado": [res["sectores"][t] for t in tickers],
                "Sector Yahoo": [sect_yahoo.get(t, "—") or "no disponible" for t in tickers],
                "Fuente": meta.get("fuente", "—"),
                "Fecha de consulta": meta.get("consulta", meta.get("cargado_en", "—")),
            }),
            hide_index=True, width="stretch",
        )
        if "procedencia" in meta:
            st.caption(f"Procedencia declarada: {meta['procedencia']}")

    with st.container(key="tarjeta_calidad"):
        st.markdown(ui.titulo_tarjeta("Calidad de los datos"), unsafe_allow_html=True)
        st.write(
            f"Período limpio: **{resumen['desde']}** a **{resumen['hasta']}** · "
            f"{resumen['fechas_originales']} fechas originales. "
            f"Calibración inicial: {len(cal) - 1} retornos ({cal.index[1].date()} a {cal.index[-1].date()}). "
            f"Evaluación: {len(ev) - 1} retornos ({ev.index[1].date()} a {ev.index[-1].date()}). "
            f"Retornos más antiguos no usados en estas ventanas: {ventanas['retornos_no_usados']}."
        )
        st.dataframe(limpio.calidad_por_ticker, width="stretch")
        if resumen["fechas_descartadas"]:
            with st.expander(f"Ver las {resumen['sesiones_descartadas']} fechas descartadas"):
                st.write(", ".join(resumen["fechas_descartadas"]))
        st.markdown('<div class="q-nota">Se usan solo sesiones en las que las 10 acciones tienen precio válido '
                    '(ajustado y sin ajustar). Los faltantes no se rellenan: no se inventan retornos cero.</div>',
                    unsafe_allow_html=True)

    with st.container(key="tarjeta_hist"):
        st.markdown(ui.titulo_tarjeta("Distribución de log-retornos diarios · ventana de calibración"),
                    unsafe_allow_html=True)
        x_cal = calibracion.log_retornos(cal)
        largo = x_cal.melt(var_name="Ticker", value_name="Log-retorno")
        fig_h = px.histogram(largo, x="Log-retorno", facet_col="Ticker", facet_col_wrap=5, nbins=60,
                             color_discrete_sequence=[ui.ORO])
        fig_h.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1], font=dict(color=ui.ORO_CLARO)))
        fig_h.update_xaxes(tickformat=".0%", title_text="")
        fig_h.update_yaxes(title_text="")
        fig_h.update_traces(marker_line_width=0, opacity=.85)
        st.plotly_chart(ui.estilo_fig(fig_h, 480), width="stretch")

    with st.container(key="tarjeta_q_r"):
        st.markdown(ui.titulo_tarjeta("Dividendos q y tasa libre de riesgo r"), unsafe_allow_html=True)
        divs = meta.get("dividendos", {})
        st.dataframe(
            pd.DataFrame({
                "Ticker": tickers,
                "Dividendos 12 meses (USD)": [divs.get(t, {}).get("dividendos_12m") for t in tickers],
                "Precio sin ajustar (USD)": [divs.get(t, {}).get("precio") for t in tickers],
                "q continua": [res["q"][t] for t in tickers],
                "Fuente de q": [res["fuente_q"][t] for t in tickers],
            }).style.format({
                "Dividendos 12 meses (USD)": "{:.2f}", "Precio sin ajustar (USD)": "{:.2f}", "q continua": "{:.2%}",
            }, na_rep="—"),
            hide_index=True, width="stretch",
        )
        st.write(f"**r continua usada: {res['r']:.2%}**  (fuente: {res['fuente_r']})")
        st.markdown('<div class="q-nota">q = ln(1 + D₁₂/S), con D₁₂ los dividendos pagados en los 365 días previos '
                    'a la última fecha y S el último cierre sin ajustar. r = ln(1 + IRX/100), con IRX el rendimiento '
                    'anual de la letra del Tesoro a 13 semanas. Suponen que el dividendo de los próximos 12 meses será '
                    'similar al de los últimos 12 y que se reparte de forma continua.</div>', unsafe_allow_html=True)

    with st.container(key="tarjeta_param"):
        st.markdown(ui.titulo_tarjeta("Parámetros calibrados · anuales"), unsafe_allow_html=True)
        st.dataframe(
            res["parametros"].style.format({
                "Media diaria x̄": "{:.5f}", "Desv. diaria s": "{:.5f}", "σ anual": "{:.2%}",
                "μ total anual": "{:.2%}", "q": "{:.2%}", "a = μ − q": "{:.2%}",
            }),
            width="stretch",
        )
        st.markdown('<div class="q-nota">σ = √252 · s (divisor n − 1);  μ total = 252 · x̄ + ½σ²;  '
                    'a = μ total − q (spot exdividendo). Calibración con los 504 retornos previos a la ventana '
                    'de evaluación.</div>', unsafe_allow_html=True)
        with st.expander("Covarianza diaria de retornos simples Σ_R (n − 1)"):
            st.dataframe(res["cov_diaria"].style.format("{:.6f}"), width="stretch")

    with st.container(key="tarjeta_descargas"):
        st.markdown(ui.titulo_tarjeta("Descargas y versiones"), unsafe_allow_html=True)
        d1, d2, d3 = st.columns(3)
        d1.download_button("Precios ajustados (CSV)", datos.a_csv_bytes(limpio.ajustados),
                           "precios_ajustados_limpios.csv", width="stretch")
        d2.download_button("Precios sin ajustar (CSV)", datos.a_csv_bytes(limpio.sin_ajustar),
                           "precios_sin_ajustar_limpios.csv", width="stretch")
        d3.download_button("Parámetros (CSV)", datos.a_csv_bytes(res["parametros"]),
                           "parametros_calibracion.csv", width="stretch")
        with st.expander("Metadatos de la descarga y versiones de bibliotecas"):
            st.json(meta)

for i, texto in [(1, "Trayectorias GBM y cuantiles · entrega 2"),
                 (2, "Árbol CRR (entrega 1) y comparación con el mercado (entrega 2)"),
                 (3, "VaR y ES histórico, normal y Monte Carlo · entrega 2"),
                 (4, "Asignación por mínimo VaR y estrategias · entrega 3"),
                 (5, "Rolling del GBM, RMSE y MAPE · entrega 2")]:
    with tabs[i]:
        with st.container(key=f"tarjeta_pendiente_{i}"):
            st.markdown(ui.titulo_tarjeta("Pendiente"), unsafe_allow_html=True)
            st.markdown(f'<div class="q-nota">{texto}</div>', unsafe_allow_html=True)
