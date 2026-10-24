"""App del Taller de Valoración de Activos (Streamlit).

Ejecutar:  streamlit run app.py

Esta capa SOLO maneja interfaz: lee entradas, llama a los módulos de src/
y muestra resultados. Ningún cálculo financiero debe vivir aquí.
"""
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from src import calibracion, config, crr, datos, gbm, lecturas, opciones, ui

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
            # Cadena de opciones del subyacente (Paso 5). Si falla, el resto del análisis sigue.
            cadena, meta_cadena, error_cadena = None, None, None
            try:
                if fuente == "Yahoo Finance":
                    cadena, meta_cadena = opciones.descargar_cadena(opcion["ticker"])
                    opciones.guardar_cadena(cadena, meta_cadena)
                else:
                    cargada = opciones.cargar_cadena(opcion["ticker"])
                    if cargada is None:
                        error_cadena = "No hay una cadena de opciones guardada en data/raw para este subyacente."
                    else:
                        cadena, meta_cadena = cargada
            except Exception as e_op:
                error_cadena = f"No se pudo descargar la cadena de opciones: {e_op}"
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
                "ticker_opcion": opcion["ticker"],
                "cadena": cadena, "meta_cadena": meta_cadena, "error_cadena": error_cadena,
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

    g1 = st.container()
    with g1:
        with st.container(key="tarjeta_g_precios"):
            st.markdown(ui.titulo_tarjeta("Precios normalizados · base 100"), unsafe_allow_html=True)
            norm = calibracion.normalizar(limpio.ajustados)
            fig_p = px.line(norm, labels={"value": "Índice", "Fecha": "", "Ticker": "", "variable": ""},
                            color_discrete_sequence=ui.PALETA)
            fig_p.update_layout(legend_title_text="")
            st.plotly_chart(ui.estilo_fig(fig_p, 400), width="stretch", config=ui.PLOTLY_CONFIG)
    sect = res["sectores"]
    g2, g3 = st.columns([1.15, 1], gap="medium")
    with g2:
        with st.container(key="tarjeta_g_corr"):
            st.markdown(ui.titulo_tarjeta("Correlaciones · agrupadas por sector"), unsafe_allow_html=True)
            orden = lecturas.orden_por_sector(tickers, sect)
            corr_o = res["corr"].loc[orden, orden]
            fig_c = px.imshow(corr_o, text_auto=".2f", zmin=-1, zmax=1, aspect="auto",
                              color_continuous_scale=ui.ESCALA_CORR)
            fig_c.update_layout(coloraxis_colorbar=dict(title="", thickness=10, tickfont=dict(color=ui.TENUE)))
            fig_c.update_traces(textfont=dict(size=10, color=ui.TEXTO))
            # Líneas que separan los bloques de cada sector
            limites = [i for i in range(1, len(orden)) if sect[orden[i]] != sect[orden[i - 1]]]
            for i in limites:
                fig_c.add_hline(y=i - 0.5, line_color=ui.ORO, line_width=1, opacity=.6)
                fig_c.add_vline(x=i - 0.5, line_color=ui.ORO, line_width=1, opacity=.6)
            st.plotly_chart(ui.estilo_fig(fig_c, 470), width="stretch", config=ui.PLOTLY_CONFIG)

    with g3, st.container(key="tarjeta_corr_lectura"):
        st.markdown(ui.titulo_tarjeta("¿Qué acciones se mueven juntas?"), unsafe_allow_html=True)
        st.markdown(ui.html_lectura(lecturas.lectura_correlaciones(res["corr"], sect)), unsafe_allow_html=True)
        st.dataframe(lecturas.pares_correlacion(res["corr"], sect).style.format({"Correlación": "{:.2f}"}),
                     hide_index=True, width="stretch")
        st.markdown('<div class="q-nota">La correlación va de −1 a 1: cerca de 1, las dos acciones suben y bajan '
                    'al mismo tiempo; cerca de 0, se mueven de forma independiente; negativa, tienden a moverse '
                    'en direcciones opuestas.</div>', unsafe_allow_html=True)

    with st.container(key="tarjeta_universo_res"):
        st.markdown(ui.titulo_tarjeta("Universo y procedencia"), unsafe_allow_html=True)
        sect_yahoo = meta.get("sectores_yahoo", {})
        st.dataframe(
            pd.DataFrame({
                "Ticker": tickers,
                "Sector usado": [res["sectores"][t] for t in tickers],
                "Sector Yahoo": [sect_yahoo.get(t, "—") or "no disponible" for t in tickers],
                "Fuente": meta.get("fuente", "—"),
                "Fecha de consulta": str(meta.get("consulta", meta.get("cargado_en", "—"))).replace("T", " "),
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
        st.dataframe(limpio.calidad_por_ticker.style.format(
            {"Primera fecha": "{:%Y-%m-%d}", "Última fecha": "{:%Y-%m-%d}"}), width="stretch")
        if resumen["fechas_descartadas"]:
            with st.expander(f"Ver las {resumen['sesiones_descartadas']} fechas descartadas"):
                st.write(", ".join(resumen["fechas_descartadas"]))
        st.markdown('<div class="q-nota">Se usan solo sesiones en las que las 10 acciones tienen precio válido '
                    '(ajustado y sin ajustar). Los faltantes no se rellenan: no se inventan retornos cero.</div>',
                    unsafe_allow_html=True)

    with st.container(key="tarjeta_hist"):
        st.markdown(ui.titulo_tarjeta("¿Cuánto se mueve cada acción en un día? · ventana de calibración"),
                    unsafe_allow_html=True)
        x_cal = calibracion.log_retornos(cal)
        largo = x_cal.melt(var_name="Ticker", value_name="Log-retorno")
        largo["Sector"] = largo["Ticker"].map(sect)
        colores = ui.colores_sector([sect[t] for t in tickers])
        fig_b = px.box(largo, x="Ticker", y="Log-retorno", color="Sector", color_discrete_map=colores,
                       category_orders={"Ticker": lecturas.orden_por_sector(tickers, sect)})
        fig_b.update_traces(marker=dict(size=3, opacity=.6), line=dict(width=1.2))
        fig_b.update_yaxes(tickformat=".0%", title_text="Log-retorno diario")
        fig_b.update_xaxes(title_text="")
        fig_b.update_layout(legend=dict(orientation="h", y=1.08, x=0, title=""), boxmode="overlay")
        st.plotly_chart(ui.estilo_fig(fig_b, 400), width="stretch", config=ui.PLOTLY_CONFIG)
        st.markdown(ui.html_lectura(lecturas.lectura_dispersion(x_cal)), unsafe_allow_html=True)

    with st.container(key="tarjeta_hist_detalle"):
        st.markdown(ui.titulo_tarjeta("Distribución detallada frente a la curva normal del GBM"),
                    unsafe_allow_html=True)
        h1, h2 = st.columns([1.3, 1], gap="medium")
        with h1:
            elegido = st.selectbox("Acción", tickers, key="hist_ticker")
            serie = x_cal[elegido]
            fig_d = px.histogram(serie, nbins=60, histnorm="probability density",
                                 color_discrete_sequence=[ui.ORO])
            fig_d.update_traces(marker_line_width=0, opacity=.8, name="Datos reales")
            malla = np.linspace(serie.min(), serie.max(), 300)
            mu_d, s_d = serie.mean(), serie.std(ddof=1)
            curva = np.exp(-0.5 * ((malla - mu_d) / s_d) ** 2) / (s_d * np.sqrt(2 * np.pi))
            fig_d.add_scatter(x=malla, y=curva, mode="lines", name="Normal (supuesto del GBM)",
                              line=dict(color=ui.TEXTO, width=2, dash="dot"))
            fig_d.update_xaxes(tickformat=".0%", title_text="Log-retorno diario")
            fig_d.update_yaxes(title_text="Densidad")
            fig_d.update_layout(legend=dict(orientation="h", y=1.1, x=0, title=""), bargap=0.02)
            st.plotly_chart(ui.estilo_fig(fig_d, 360), width="stretch", config=ui.PLOTLY_CONFIG)
        with h2:
            st.markdown(ui.html_lectura(lecturas.lectura_histograma(serie, elegido)), unsafe_allow_html=True)
            st.markdown('<div class="q-nota">El GBM supone que los log-retornos siguen una curva normal (línea '
                        'punteada). Si las barras se salen de la curva en los extremos, hay más días de '
                        'movimientos fuertes de los que el modelo espera.</div>', unsafe_allow_html=True)
        st.dataframe(
            lecturas.resumen_distribucion(x_cal).style.format({
                "Mejor día": "{:.2%}", "Peor día": "{:.2%}", "Esperados si fuera normal": "{:.1f}",
            }),
            width="stretch",
        )

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
        st.markdown(ui.titulo_tarjeta("Parámetros calibrados · riesgo frente a retorno"), unsafe_allow_html=True)
        par = res["parametros"].reset_index().rename(columns={"index": "Ticker"})
        par = par.rename(columns={par.columns[0]: "Ticker"})
        par["Sector"] = par["Ticker"].map(sect)
        fig_rr = px.scatter(par, x="σ anual", y="μ total anual", color="Sector", text="Ticker",
                            color_discrete_map=ui.colores_sector([sect[t] for t in tickers]))
        fig_rr.update_traces(marker=dict(size=14, line=dict(width=1, color="#070706")),
                             textposition="top center", textfont=dict(color=ui.TEXTO, size=11))
        fig_rr.update_xaxes(tickformat=".0%", title_text="Volatilidad anual σ (riesgo)")
        fig_rr.update_yaxes(tickformat=".0%", title_text="Retorno total anual μ")
        fig_rr.add_hline(y=0, line_color="#3a3224", line_width=1)
        fig_rr.update_layout(legend=dict(orientation="h", y=1.1, x=0, title=""))
        st.plotly_chart(ui.estilo_fig(fig_rr, 420), width="stretch", config=ui.PLOTLY_CONFIG)
        st.markdown(ui.html_lectura(lecturas.lectura_riesgo_retorno(res["parametros"])), unsafe_allow_html=True)
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

# ---------------------------------------------------------------------------
# GBM (Paso 4): proyección del precio bajo la medida física P
# Parámetros calibrados con los últimos 504 retornos hasta hoy (decisión D15);
# S0 = último cierre sin ajustar; drift a = μ total − q.
# ---------------------------------------------------------------------------
with tabs[1]:
    precios_hoy = limpio.ajustados.iloc[-(config.N_CALIBRACION + 1):]
    par_hoy = calibracion.calibrar(precios_hoy, res["q"], config.D)
    s0_hoy = limpio.sin_ajustar.iloc[-1]
    semilla_base = int(st.session_state.get("semilla", config.SEMILLA))

    def simular(ticker: str, sesiones: int):
        a_t, sig_t = float(par_hoy.loc[ticker, "a = μ − q"]), float(par_hoy.loc[ticker, "σ anual"])
        tr = gbm.simular_gbm(float(s0_hoy[ticker]), a_t, sig_t, sesiones, config.N_TRAYECTORIAS_GBM,
                             semilla_base + tickers.index(ticker))
        return tr, gbm.resumir(tr, a_t)

    c1, c2, _ = st.columns([1, 1, 2])
    with c1:
        t_gbm = st.selectbox("Acción", tickers, key="gbm_ticker",
                             index=tickers.index(config.TICKER_OPCION) if config.TICKER_OPCION in tickers else 0)
    with c2:
        horizonte = st.selectbox("Horizonte", list(gbm.HORIZONTES), index=2, key="gbm_horizonte")
    sesiones = gbm.HORIZONTES[horizonte]
    tray, rg = simular(t_gbm, sesiones)
    bajo, alto = rg.rango

    st.markdown(ui.html_kpis([
        ("Precio hoy", lecturas._usd(rg.s0), f"último cierre · {limpio.sin_ajustar.index[-1]:%Y-%m-%d}"),
        ("Precio más probable", lecturas._usd(rg.mediana), f"mediana en {horizonte}"),
        ("Rango probable", f"{bajo:,.0f} – {alto:,.0f}".replace(",", "."), "80 % de las simulaciones (P10–P90)"),
        ("Probabilidad de subir", f"{rg.prob_subir:.0%}".replace(".", ","), "terminar por encima del precio de hoy"),
    ]), unsafe_allow_html=True)

    with st.container(key="tarjeta_gbm_abanico"):
        st.markdown(ui.titulo_tarjeta(f"Caminos posibles de {t_gbm} · {config.N_TRAYECTORIAS_GBM:,} simulaciones"
                                      .replace(",", ".")), unsafe_allow_html=True)
        fechas = pd.bdate_range(limpio.sin_ajustar.index[-1], periods=sesiones + 1)
        fig_g = go.Figure()
        # 40 caminos de ejemplo en una sola traza (separados por None) para que el gráfico sea liviano
        xs, ys = [], []
        for fila in tray[:40]:
            xs += list(fechas) + [None]
            ys += list(fila) + [None]
        fig_g.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Caminos de ejemplo (40)",
                                   line=dict(color="rgba(232,226,212,.18)", width=1), hoverinfo="skip"))
        fig_g.add_trace(go.Scatter(x=fechas, y=rg.p90, mode="lines", line=dict(width=0), showlegend=False,
                                   hoverinfo="skip"))
        fig_g.add_trace(go.Scatter(x=fechas, y=rg.p10, mode="lines", line=dict(width=0), fill="tonexty",
                                   fillcolor="rgba(217,181,106,.22)", name="Rango probable (P10–P90)",
                                   hovertemplate="P10: %{y:.2f}<extra></extra>"))
        fig_g.add_trace(go.Scatter(x=fechas, y=rg.p50, mode="lines", name="Mediana",
                                   line=dict(color=ui.ORO_CLARO, width=3),
                                   hovertemplate="Mediana: %{y:.2f}<extra></extra>"))
        fig_g.add_hline(y=rg.s0, line_color=ui.TENUE, line_dash="dot", line_width=1,
                        annotation_text="Precio hoy", annotation_font_color=ui.TENUE)
        fig_g.update_yaxes(title_text="Precio (USD)")
        fig_g.update_layout(legend=dict(orientation="h", y=1.08, x=0), hovermode="x unified")
        st.plotly_chart(ui.estilo_fig(fig_g, 460), width="stretch", config=ui.PLOTLY_CONFIG)
        st.markdown(ui.html_lectura(lecturas.lectura_gbm(t_gbm, horizonte, rg)), unsafe_allow_html=True)

    d1, d2 = st.columns([1.3, 1], gap="medium")
    with d1, st.container(key="tarjeta_gbm_final"):
        st.markdown(ui.titulo_tarjeta(f"Distribución de precios al final de {horizonte}"), unsafe_allow_html=True)
        fig_f = px.histogram(x=rg.finales, nbins=70, color_discrete_sequence=[ui.ORO])
        fig_f.update_traces(marker_line_width=0, opacity=.85, hovertemplate="USD %{x:.0f}: %{y} simulaciones<extra></extra>")
        for valor, texto, color in [(rg.s0, "Hoy", ui.TENUE), (bajo, "P10", ui.ORO_CLARO),
                                    (rg.mediana, "Mediana", "#ffffff"), (alto, "P90", ui.ORO_CLARO)]:
            fig_f.add_vline(x=valor, line_color=color, line_dash="dot", line_width=1.5,
                            annotation_text=texto, annotation_font_color=color, annotation_position="top")
        fig_f.update_xaxes(title_text="Precio final (USD)")
        fig_f.update_yaxes(title_text="Número de simulaciones")
        fig_f.update_layout(bargap=0.02, showlegend=False)
        st.plotly_chart(ui.estilo_fig(fig_f, 360), width="stretch", config=ui.PLOTLY_CONFIG)
        st.markdown('<div class="q-nota">La cola larga hacia la derecha es normal en el GBM: el precio no puede '
                    'caer por debajo de cero, pero sí puede subir mucho.</div>', unsafe_allow_html=True)
    with d2, st.container(key="tarjeta_gbm_checks"):
        st.markdown(ui.titulo_tarjeta("Comprobaciones del modelo"), unsafe_allow_html=True)
        st.markdown(ui.html_checklist(lecturas.comprobaciones_gbm(rg)), unsafe_allow_html=True)
        st.markdown(f'<div class="q-nota" style="margin-top:12px">Parámetros usados para {t_gbm}: drift '
                    f'a = {par_hoy.loc[t_gbm, "a = μ − q"]:.2%}, volatilidad σ = {par_hoy.loc[t_gbm, "σ anual"]:.2%}, '
                    f'calibrados con los últimos {config.N_CALIBRACION} retornos ({precios_hoy.index[1]:%Y-%m-%d} a '
                    f'{precios_hoy.index[-1]:%Y-%m-%d}). Semilla {semilla_base + tickers.index(t_gbm)}.</div>',
                    unsafe_allow_html=True)

    with st.container(key="tarjeta_gbm_tabla"):
        st.markdown(ui.titulo_tarjeta(f"Las 10 acciones en {horizonte}"), unsafe_allow_html=True)
        filas = []
        for t in tickers:
            _, r_t = simular(t, sesiones)
            filas.append({"Ticker": t, "Sector": sect[t], "Precio hoy": r_t.s0,
                          "Mediana": r_t.mediana, "P10": r_t.rango[0], "P90": r_t.rango[1],
                          "Prob. de subir": r_t.prob_subir,
                          "Drift a": par_hoy.loc[t, "a = μ − q"], "Volatilidad σ": par_hoy.loc[t, "σ anual"]})
        st.dataframe(pd.DataFrame(filas).style.format({
            "Precio hoy": "{:.2f}", "Mediana": "{:.2f}", "P10": "{:.2f}", "P90": "{:.2f}",
            "Prob. de subir": "{:.0%}", "Drift a": "{:.2%}", "Volatilidad σ": "{:.2%}",
        }), hide_index=True, width="stretch")
        st.markdown('<div class="q-nota">P10 y P90: el 10 % de las simulaciones termina por debajo de P10 y el 10 % '
                    'por encima de P90. Son proyecciones del modelo, no predicciones seguras.</div>',
                    unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Opciones (Pasos 5, 6, 7, 8 y 9)
# ---------------------------------------------------------------------------
with tabs[2]:
    # ---------- Parte 1: calculadora CRR (Paso 8) ----------
    o = opcion
    with st.container(key="tarjeta_crr"):
        st.markdown(ui.titulo_tarjeta("Calculadora · árbol binomial CRR"), unsafe_allow_html=True)
        st.markdown(f'<div class="q-nota">Usa los valores del panel de entradas (sección Opción): S0 = {o["s0"]:.2f}, '
                    f'K = {o["k"]:.2f}, T = {o["t"]:.2f} años, r = {o["r"]:.2%}, q = {o["q"]:.2%}, σ = {o["sigma"]:.2%}, '
                    f'N = {int(o["n"])} pasos, {o["tipo"]}. Por defecto es el caso de control del instructivo.</div>',
                    unsafe_allow_html=True)
        try:
            v_eu = crr.precio_crr(o["s0"], o["k"], o["t"], o["r"], o["q"], o["sigma"], int(o["n"]), o["tipo"], "europea")
            v_am = crr.precio_crr(o["s0"], o["k"], o["t"], o["r"], o["q"], o["sigma"], int(o["n"]), o["tipo"], "americana")
            par_c = crr.parametros_crr(o["t"], int(o["n"]), o["r"], o["q"], o["sigma"]) if o["t"] > 0 else None
            error_crr = None
        except ValueError as e_crr:
            error_crr = str(e_crr)
        if error_crr:
            st.error(f"Revise las entradas del árbol: {error_crr}")
        else:
            st.markdown(ui.html_kpis([
                ("Opción europea", lecturas._usd(v_eu), f"{o['tipo']} · solo se ejerce al vencimiento"),
                ("Opción americana", lecturas._usd(v_am), f"{o['tipo']} · se puede ejercer antes"),
                ("Valor del ejercicio anticipado", lecturas._usd(v_am - v_eu), "americana − europea"),
                ("Probabilidad neutral p^Q", f"{par_c.p:.4f}".replace(".", ",") if par_c else "—",
                 f"u = {par_c.u:.4f} · d = {par_c.d:.4f}".replace(".", ",") if par_c else "T = 0: se paga el payoff"),
            ]), unsafe_allow_html=True)
            st.markdown(ui.html_lectura(lecturas.lectura_crr(v_eu, v_am, o["tipo"])), unsafe_allow_html=True)

            a1, a2 = st.columns(2, gap="medium")
            with a1:
                estilo_dib = o["estilo"]
                st.markdown(f'<div class="q-titulo" style="margin-top:6px">Árbol de 3 pasos · {o["tipo"]} {estilo_dib}</div>',
                            unsafe_allow_html=True)
                if o["t"] > 0:
                    arb = crr.arbol_crr(o["s0"], o["k"], o["t"], o["r"], o["q"], o["sigma"], 3, o["tipo"], estilo_dib)
                    fig_t = go.Figure()
                    for m in range(3):
                        for j in range(m + 1):
                            for j2 in (j, j + 1):
                                fig_t.add_trace(go.Scatter(x=[m, m + 1], y=[2 * j - m, 2 * j2 - m - 1], mode="lines",
                                                           line=dict(color="#3a3224", width=1.5), hoverinfo="skip",
                                                           showlegend=False))
                    xs, ys, txt, col = [], [], [], []
                    for m in range(4):
                        for j in range(m + 1):
                            xs.append(m); ys.append(2 * j - m)
                            txt.append(f"S={arb.precios[m][j]:.2f}<br>V={arb.valores[m][j]:.2f}")
                            col.append(ui.ORO_CLARO if arb.ejercicio[m][j] and m < 3 else
                                       (ui.ORO if m < 3 else ui.TENUE))
                    fig_t.add_trace(go.Scatter(x=xs, y=ys, mode="markers+text", text=txt, textposition="top center",
                                               textfont=dict(size=11, color=ui.TEXTO),
                                               marker=dict(size=14, color=col, line=dict(color="#070706", width=1)),
                                               hoverinfo="skip", showlegend=False))
                    fig_t.update_xaxes(tickvals=[0, 1, 2, 3], ticktext=["Hoy", "Paso 1", "Paso 2", "Vencimiento"],
                                       range=[-0.4, 3.4], showgrid=False)
                    fig_t.update_yaxes(visible=False, range=[-4, 4.6])
                    st.plotly_chart(ui.estilo_fig(fig_t, 420), width="stretch", config=ui.PLOTLY_CONFIG)
                    st.markdown(f'<div class="q-nota">S: precio de la acción en el nodo. V: valor de la opción. En cada '
                                f'paso el precio sube ×{arb.parametros.u:.4f} o baja ×{arb.parametros.d:.4f}; el valor se '
                                f'calcula de derecha a izquierda con p^Q = {arb.parametros.p:.4f}.'
                                f'{" Los nodos más claros son los que conviene ejercer antes." if estilo_dib == "americana" else ""}'
                                '</div>', unsafe_allow_html=True)
            with a2:
                st.markdown('<div class="q-titulo" style="margin-top:6px">Convergencia al aumentar los pasos N</div>',
                            unsafe_allow_html=True)
                if o["t"] > 0:
                    conv_eu = crr.convergencia(o["s0"], o["k"], o["t"], o["r"], o["q"], o["sigma"], o["tipo"], "europea")
                    conv_am = crr.convergencia(o["s0"], o["k"], o["t"], o["r"], o["q"], o["sigma"], o["tipo"], "americana")
                    bs = crr.black_scholes(o["s0"], o["k"], o["t"], o["r"], o["q"], o["sigma"], o["tipo"])
                    fig_cv = go.Figure()
                    fig_cv.add_trace(go.Scatter(x=[n for n, _ in conv_eu], y=[v for _, v in conv_eu], mode="lines+markers",
                                                name="Europea", line=dict(color=ui.ORO_CLARO, width=2)))
                    fig_cv.add_trace(go.Scatter(x=[n for n, _ in conv_am], y=[v for _, v in conv_am], mode="lines+markers",
                                                name="Americana", line=dict(color=ui.PALETA[4], width=2)))
                    fig_cv.add_hline(y=bs, line_color=ui.TENUE, line_dash="dot",
                                     annotation_text=f"Valor exacto europea (Black-Scholes) {bs:.4f}",
                                     annotation_font_color=ui.TENUE)
                    fig_cv.update_xaxes(type="log", title_text="Pasos del árbol N (escala logarítmica)")
                    fig_cv.update_yaxes(title_text="Prima (USD)")
                    fig_cv.update_layout(legend=dict(orientation="h", y=1.1, x=0))
                    st.plotly_chart(ui.estilo_fig(fig_cv, 420), width="stretch", config=ui.PLOTLY_CONFIG)
                    st.markdown('<div class="q-nota">Con pocos pasos el árbol oscila; con más pasos se estabiliza en el '
                                'valor exacto.</div>', unsafe_allow_html=True)

            c_ctrl = config.CONTROL_CRR
            v_ctrl = crr.precio_crr(c_ctrl["S0"], c_ctrl["K"], c_ctrl["T"], c_ctrl["r"], c_ctrl["q"], c_ctrl["sigma"],
                                    1000, "call", "europea")
            st.markdown(ui.html_checklist([
                (abs(v_ctrl - config.CONTROL_CRR_VALOR) < 0.005,
                 f"Control del instructivo: call europea con S0 = K = 100, T = 1, r = 5 %, q = 0, σ = 20 % da "
                 f"{v_ctrl:.4f} con N = 1.000 (objetivo 10,4506)."),
                (par_c is None or 0 < par_c.p < 1, "La probabilidad neutral p^Q está entre 0 y 1."),
                (v_am >= v_eu - 1e-9, "La americana vale al menos lo mismo que la europea."),
            ]), unsafe_allow_html=True)

    # ---------- Parte 2: contratos reales (Pasos 5, 6 y 7) ----------
    t_op = res.get("ticker_opcion", config.TICKER_OPCION)
    cadena, meta_cad = res.get("cadena"), res.get("meta_cadena") or {}
    if cadena is None:
        with st.container(key="tarjeta_sin_cadena"):
            st.markdown(ui.titulo_tarjeta(f"Contratos de mercado de {t_op}"), unsafe_allow_html=True)
            st.markdown(ui.html_lectura(res.get("error_cadena") or "Ejecute el análisis con la fuente Yahoo Finance "
                                        "para descargar la cadena de opciones."), unsafe_allow_html=True)
    else:
        tipo_op = o["tipo"]
        s0_op = float(limpio.sin_ajustar[t_op].iloc[-1])
        a_op, sig_op = float(par_hoy.loc[t_op, "a = μ − q"]), float(par_hoy.loc[t_op, "σ anual"])
        q_op, r_op = float(res["q"][t_op]), float(res["r"])
        hoy_op = pd.Timestamp(str(meta_cad.get("consulta", limpio.sin_ajustar.index[-1]))[:10]).date()
        cands = opciones.candidatos(cadena, tipo_op, s0_op, a_op, sig_op, hoy_op)

        with st.container(key="tarjeta_candidatos"):
            st.markdown(ui.titulo_tarjeta(f"Contratos candidatos · {tipo_op}s de {t_op}"), unsafe_allow_html=True)
            st.markdown(f'<div class="q-nota">Consulta: {str(meta_cad.get("consulta", "—")).replace("T", " ")} · '
                        f'{meta_cad.get("fuente", "")} (las cotizaciones pueden ser diferidas). Precio de {t_op}: '
                        f'{lecturas._usd(s0_op)} (cierre sin ajustar). Para cada vencimiento se eligen los strikes '
                        'reales más cercanos a los precios proyectados P25, P50 y P75 del GBM (Ec. 4).</div>',
                        unsafe_allow_html=True)
            if cands.empty:
                st.warning("No se encontraron candidatos en la cadena descargada.")
            else:
                vista = cands[["Vencimiento", "Días", "Cuantil", "Precio proyectado", "Strike", "Bid", "Ask",
                               "Spread %", "Volumen", "Interés abierto", "Vol. implícita"]]
                st.dataframe(vista.style.format({
                    "Precio proyectado": "{:.2f}", "Strike": "{:.2f}", "Bid": "{:.2f}", "Ask": "{:.2f}",
                    "Spread %": "{:.1%}", "Volumen": "{:,.0f}", "Interés abierto": "{:,.0f}", "Vol. implícita": "{:.1%}",
                }, na_rep="—"), hide_index=True, width="stretch")
                n_inval = int((~cands["Cotización válida"]).sum())
                if n_inval:
                    st.markdown(f'<div class="q-nota">{n_inval} de {len(cands)} contratos no tienen bid/ask utilizable '
                                'en esta consulta (por ejemplo, en cero con el mercado cerrado).</div>',
                                unsafe_allow_html=True)

        if not cands.empty:
            idx_sug, motivo = opciones.sugerir(cands)
            etiquetas = [f"{r_['Vencimiento']} · strike {r_['Strike']:.2f} · {r_['Cuantil']}" for _, r_ in cands.iterrows()]
            with st.container(key="tarjeta_eleccion"):
                st.markdown(ui.titulo_tarjeta("Contrato elegido"), unsafe_allow_html=True)
                e1, e2 = st.columns([1, 1.4], gap="medium")
                with e1:
                    elegido_i = st.selectbox("Contrato", list(range(len(cands))), index=idx_sug,
                                             format_func=lambda i: etiquetas[i] + ("  ★ sugerido" if i == idx_sug else ""),
                                             key="contrato_elegido")
                with e2:
                    st.markdown(ui.html_lectura(f"<b>Sugerido:</b> {etiquetas[idx_sug]}. {motivo}"), unsafe_allow_html=True)
                c_el = cands.loc[elegido_i]
                prima = c_el["Ask"] if pd.notna(c_el["Ask"]) and c_el["Ask"] > 0 else c_el["Último precio"]
                probs = opciones.probabilidades(s0_op, a_op, sig_op, c_el["Strike"], c_el["T GBM"], c_el["T valoración"],
                                                r_op, tipo_op, float(prima), semilla=semilla_base)
                st.markdown(ui.html_kpis([
                    ("Prima a pagar (ask)", lecturas._usd(prima), "por acción · ×100 por contrato"),
                    ("Probabilidad de ejercicio", f"{probs['p_ejercicio']:.0%}",
                     f"{'S_T > K' if tipo_op == 'call' else 'S_T < K'} · fórmula exacta {probs['p_ejercicio_exacta']:.0%}"),
                    ("Probabilidad de ganancia", f"{probs['p_beneficio']:.0%}", "Π_T > 0 al vencimiento (Ec. 5)"),
                    ("Punto de equilibrio", lecturas._usd(probs["punto_equilibrio"]), "precio que debe alcanzar la acción"),
                ]), unsafe_allow_html=True)
                st.markdown(ui.html_lectura(lecturas.lectura_contrato(c_el, probs, tipo_op, t_op)), unsafe_allow_html=True)

                g_a, g_b = st.columns([1.4, 1], gap="medium")
                with g_a:
                    st.markdown('<div class="q-titulo" style="margin-top:6px">Ganancia o pérdida al vencimiento</div>',
                                unsafe_allow_html=True)
                    lo, hi = np.percentile(probs["st"], [0.5, 99.5])
                    malla = np.linspace(lo, hi, 300)
                    ganancia = crr.payoff(malla, c_el["Strike"], tipo_op) - probs["costo_total"]
                    fig_pay = go.Figure()
                    fig_pay.add_trace(go.Histogram(x=probs["st"], nbinsx=70, histnorm="probability density",
                                                   marker_color="rgba(217,181,106,.25)", name="Precios simulados (GBM)",
                                                   yaxis="y2", hoverinfo="skip"))
                    fig_pay.add_trace(go.Scatter(x=malla, y=ganancia, mode="lines", name="Ganancia por acción",
                                                 line=dict(color=ui.ORO_CLARO, width=3),
                                                 hovertemplate="Si termina en %{x:.0f}: %{y:.2f} USD<extra></extra>"))
                    fig_pay.add_hline(y=0, line_color="#3a3224")
                    fig_pay.add_vline(x=c_el["Strike"], line_dash="dot", line_color=ui.TENUE, annotation_text="Strike",
                                      annotation_font_color=ui.TENUE)
                    fig_pay.add_vline(x=probs["punto_equilibrio"], line_dash="dot", line_color=ui.ORO,
                                      annotation_text="Equilibrio", annotation_font_color=ui.ORO,
                                      annotation_position="bottom right")
                    fig_pay.update_layout(yaxis2=dict(overlaying="y", side="right", visible=False),
                                          legend=dict(orientation="h", y=1.1, x=0), bargap=0)
                    fig_pay.update_xaxes(title_text=f"Precio de {t_op} al vencimiento (USD)")
                    fig_pay.update_yaxes(title_text="Ganancia por acción (USD)")
                    st.plotly_chart(ui.estilo_fig(fig_pay, 400), width="stretch", config=ui.PLOTLY_CONFIG)
                with g_b:
                    st.markdown('<div class="q-titulo" style="margin-top:6px">¿Qué pasa si cambian a y σ?</div>',
                                unsafe_allow_html=True)
                    sens = opciones.sensibilidad(s0_op, a_op, sig_op, c_el["Strike"], c_el["T GBM"], c_el["T valoración"],
                                                 r_op, tipo_op, float(prima), semilla=semilla_base)
                    st.dataframe(sens.style.format({"Drift a": "{:.1%}", "Volatilidad σ": "{:.1%}",
                                                    "P(ejercicio)": "{:.0%}", "P(beneficio)": "{:.0%}"}),
                                 hide_index=True, width="stretch")
                    st.markdown(ui.html_lectura(lecturas.lectura_sensibilidad(sens)), unsafe_allow_html=True)

            # ---------- Parte 3: teórico vs. mercado (Pasos 7 y 9) ----------
            with st.container(key="tarjeta_mercado"):
                st.markdown(ui.titulo_tarjeta("Precio teórico frente al mercado"), unsafe_allow_html=True)
                n_val = max(int(o["n"]), 200)
                v_am_m = crr.precio_crr(s0_op, c_el["Strike"], c_el["T valoración"], r_op, q_op, sig_op, n_val,
                                        tipo_op, "americana")
                v_eu_m = crr.precio_crr(s0_op, c_el["Strike"], c_el["T valoración"], r_op, q_op, sig_op, n_val,
                                        tipo_op, "europea")
                cmp = opciones.comparar(v_am_m, c_el["Bid"], c_el["Ask"])
                st.markdown(ui.html_kpis([
                    ("CRR americana", lecturas._usd(v_am_m), "estilo del contrato real"),
                    ("CRR europea", lecturas._usd(v_eu_m), "referencia"),
                    ("Precio medio del mercado", lecturas._usd(cmp["medio"]) if cmp["valida"] else "pendiente",
                     f"bid {c_el['Bid']:.2f} · ask {c_el['Ask']:.2f}" if cmp["valida"] else "bid/ask no utilizable"),
                    ("Diferencia δ", lecturas._usd(cmp["delta"]) if cmp["valida"] else "—",
                     f"δ relativo {cmp['delta_rel']:.1%}".replace(".", ",") if cmp["valida"] else "comparación pendiente"),
                ]), unsafe_allow_html=True)
                m1, m2 = st.columns([1.4, 1], gap="medium")
                with m1:
                    fig_m = go.Figure()
                    if cmp["valida"]:
                        fig_m.add_trace(go.Scatter(x=[c_el["Bid"], c_el["Ask"]], y=["Mercado"] * 2, mode="lines",
                                                   line=dict(color=ui.ORO, width=18), name="Rango bid – ask",
                                                   hovertemplate="%{x:.2f}<extra></extra>"))
                        fig_m.add_trace(go.Scatter(x=[cmp["medio"]], y=["Mercado"], mode="markers", name="Precio medio",
                                                   marker=dict(color="#070706", size=10, line=dict(color=ui.TEXTO, width=2))))
                    fig_m.add_trace(go.Scatter(x=[v_am_m], y=["Modelo"], mode="markers", name="CRR americana",
                                               marker=dict(symbol="diamond", size=16, color=ui.ORO_CLARO)))
                    fig_m.add_trace(go.Scatter(x=[v_eu_m], y=["Modelo"], mode="markers", name="CRR europea",
                                               marker=dict(symbol="circle-open", size=12, color=ui.TEXTO)))
                    fig_m.update_xaxes(title_text="Prima por acción (USD)")
                    fig_m.update_layout(legend=dict(orientation="h", y=1.25, x=0))
                    st.plotly_chart(ui.estilo_fig(fig_m, 260), width="stretch", config=ui.PLOTLY_CONFIG)
                with m2:
                    st.markdown(ui.html_lectura(lecturas.lectura_mercado(cmp, v_am_m, c_el["Bid"], c_el["Ask"],
                                                                        c_el["Vol. implícita"], sig_op)),
                                unsafe_allow_html=True)
                st.markdown('<div class="q-titulo" style="margin-top:8px">Ficha del contrato</div>', unsafe_allow_html=True)
                st.dataframe(pd.DataFrame([{
                    "Identificador": c_el["Contrato"], "Tipo": tipo_op, "Strike": c_el["Strike"],
                    "Vencimiento": c_el["Vencimiento"], "Estilo": "americano", "Spot S0": s0_op,
                    "Bid": c_el["Bid"], "Ask": c_el["Ask"], "Volumen": c_el["Volumen"],
                    "Interés abierto": c_el["Interés abierto"],
                    "Hora de consulta": str(meta_cad.get("consulta", "—")).replace("T", " "),
                }]).style.format({"Strike": "{:.2f}", "Spot S0": "{:.2f}", "Bid": "{:.2f}", "Ask": "{:.2f}",
                                  "Volumen": "{:,.0f}", "Interés abierto": "{:,.0f}"}, na_rep="—"),
                    hide_index=True, width="stretch")
                st.markdown(f'<div class="q-nota">Valoración bajo la medida neutral al riesgo Q (drift r − q): '
                            f'r = {r_op:.2%}, q = {q_op:.2%}, σ histórica = {sig_op:.2%}, T = {c_el["Días"]} días / 365 = '
                            f'{c_el["T valoración"]:.4f} años, N = {n_val} pasos. Las probabilidades de arriba usan la '
                            f'medida física P (drift a = {a_op:.2%}) y {c_el["Sesiones"]} sesiones '
                            f'(T GBM = {c_el["T GBM"]:.4f} años).</div>', unsafe_allow_html=True)

for i, texto in [(3, "VaR y ES histórico, normal y Monte Carlo · entrega 2"),
                 (4, "Asignación por mínimo VaR y estrategias · entrega 3"),
                 (5, "Rolling del GBM, RMSE y MAPE · entrega 2")]:
    with tabs[i]:
        with st.container(key=f"tarjeta_pendiente_{i}"):
            st.markdown(ui.titulo_tarjeta("Pendiente"), unsafe_allow_html=True)
            st.markdown(f'<div class="q-nota">{texto}</div>', unsafe_allow_html=True)
