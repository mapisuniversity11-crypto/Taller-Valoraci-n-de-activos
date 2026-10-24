"""App del Taller de Valoración de Activos (Streamlit).

Ejecutar:  streamlit run app.py

Esta capa SOLO maneja interfaz: lee entradas, llama a los módulos de src/
y muestra resultados. Ningún cálculo financiero debe vivir aquí.
"""
from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

from src import calibracion, config, datos, ui

st.set_page_config(page_title="Quantia", layout="wide")

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

st.title("Quantia")
st.caption(
    "Valoración de opciones y riesgo de portafolio — modelo binomial, GBM, opciones y asignación "
    "sectorial por VaR. Universidad Externado de Colombia."
)

# ---------------------------------------------------------------------------
# Paso 1. Entradas (barra lateral)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Entradas")
    ejecutar = st.button("Ejecutar análisis", type="primary", width="stretch")
    st.caption("Ajuste las entradas de abajo y presione el botón para recalcular todo.")

    st.subheader("Acciones, sectores y dividendos")
    universo = st.data_editor(
        pd.DataFrame(
            {
                "Ticker": list(config.TICKERS_SECTORES),
                "Sector": list(config.TICKERS_SECTORES.values()),
                "q": [0.0] * len(config.TICKERS_SECTORES),
            }
        ),
        num_rows="fixed",
        hide_index=True,
        column_config={
            "q": st.column_config.NumberColumn(
                "q manual", help="Valor de respaldo (decimal, continuo anual). Si la casilla de Yahoo está "
                "activa, la app calcula q con dividendos reales y los muestra en la pestaña Datos.",
                min_value=0.0, max_value=0.5, step=0.001, format="%.4f",
            ),
        },
    )
    st.caption(
        "Puede corregir el sector manualmente si falla el metadato. La columna q manual solo se usa "
        "si no se calculan los dividendos con Yahoo; los q reales se ven en la pestaña Datos."
    )

    st.subheader("Fechas")
    hoy = date.today()
    fecha_fin = st.date_input("Último día con datos", hoy)
    fecha_inicio = st.date_input(
        "Fecha inicial", hoy - timedelta(days=int(365.25 * config.ANIOS_HISTORIA))
    )
    st.caption("En yfinance `end` es exclusiva: se descarga hasta fecha_fin + 1 día.")

    st.subheader("Fuente de datos")
    fuente = st.radio(
        "Origen de los precios",
        ["Yahoo Finance", "Respaldo del repositorio", "Subir CSV"],
        help="Si Yahoo falla, use el respaldo guardado en data/raw o suba sus propios CSV.",
    )
    csv_aj = csv_sin = None
    procedencia = ""
    if fuente == "Subir CSV":
        csv_aj = st.file_uploader("CSV precios ajustados", type="csv")
        csv_sin = st.file_uploader("CSV precios sin ajustar", type="csv")
        procedencia = st.text_input("Procedencia (fuente y fecha de descarga)")
        st.caption("Formato: primera columna Fecha y una columna por ticker.")
    guardar = fuente == "Yahoo Finance" and st.checkbox(
        "Guardar descarga como respaldo en data/raw", value=True
    )

    st.subheader("Cartera y riesgo")
    capital = st.number_input("Capital (USD)", value=config.CAPITAL, step=1_000.0)
    alpha = st.number_input("Confianza VaR", value=config.ALPHA, min_value=0.90, max_value=0.999, format="%.3f")
    h_var = st.number_input("Horizonte VaR (sesiones)", value=config.HORIZONTE_VAR, min_value=1)
    auto_q_r = st.checkbox(
        "Calcular q y r con datos de Yahoo (recomendado)", value=True,
        help="q: dividendos de los últimos 12 meses / precio sin ajustar. r: letra del Tesoro a 13 semanas (^IRX). "
             "Si falla, se usan los valores manuales.",
    )
    r = st.number_input("Tasa libre de riesgo r manual (decimal, anual)", value=config.TASA_LIBRE_RIESGO, format="%.4f")
    semilla = st.number_input("Semilla", value=config.SEMILLA, step=1)

    st.subheader("Estrategias y evaluación")
    presupuesto_v = st.number_input(
        "Presupuesto de riesgo V (USD)", value=config.PRESUPUESTO_RIESGO, min_value=0.0, step=500.0,
        help="VaR máximo permitido para la estrategia de presupuesto de riesgo (Paso 12).",
    )
    h_rolling = st.number_input(
        "Horizonte H del rolling (sesiones)", value=config.H_ROLLING, min_value=1,
        help="Sesiones hacia adelante que se pronostican en la evaluación del GBM (Paso 13).",
    )

    st.subheader("Opción")
    ticker_opcion = st.selectbox(
        "Acción subyacente", list(universo["Ticker"]),
        index=list(universo["Ticker"]).index(config.TICKER_OPCION),
    )
    tipo_opcion = st.radio("Tipo", ["call", "put"], horizontal=True)
    estilo_opcion = st.radio("Estilo", ["europea", "americana"], horizontal=True)

    st.markdown("**Árbol CRR**")
    st.caption(
        "Entrega 1: valores del caso de control. En la semana 2, K y T se toman "
        "del contrato real elegido y S0 del cierre sin ajustar."
    )
    c = config.CONTROL_CRR
    crr_s0 = st.number_input("Spot S0", value=c["S0"], min_value=0.01)
    crr_k = st.number_input("Strike K", value=c["K"], min_value=0.01)
    crr_t = st.number_input("Vencimiento T (años)", value=c["T"], min_value=0.0, format="%.4f")
    crr_r = st.number_input("Tasa r del árbol (decimal)", value=c["r"], format="%.4f")
    crr_sigma = st.number_input("Volatilidad σ (decimal)", value=c["sigma"], min_value=0.0001, format="%.4f")
    crr_q = st.number_input("Dividendo q del subyacente", value=c["q"], min_value=0.0, format="%.4f")
    n_pasos = st.number_input("Pasos del árbol N", value=config.N_PASOS_CRR, min_value=1)


# ---------------------------------------------------------------------------
# Ejecución: obtiene datos y calcula; los resultados quedan en session_state
# para que no se pierdan al mover otro control de la app.
# ---------------------------------------------------------------------------
tickers = list(universo["Ticker"])
sectores = dict(zip(universo["Ticker"], universo["Sector"]))
q_manual = pd.Series(universo["q"].to_numpy(), index=tickers, dtype=float).fillna(0.0)


def obtener_crudos():
    if fuente == "Yahoo Finance":
        crudos = datos.descargar_yahoo(tickers, fecha_inicio, fecha_fin)
        crudos.meta["sectores_yahoo"] = datos.sectores_yahoo(tickers)
        if auto_q_r:
            crudos.meta["dividendos"] = datos.dividendos_yahoo(tickers, crudos.sin_ajustar)
            crudos.meta["tasa_libre_riesgo"] = datos.tasa_libre_riesgo_yahoo(fecha_fin)
        if guardar:
            datos.guardar_respaldo(crudos)
        return crudos
    if fuente == "Respaldo del repositorio":
        if not datos.existe_respaldo():
            raise RuntimeError("No hay respaldo en data/raw. Descargue primero desde Yahoo Finance.")
        return datos.cargar_respaldo()
    if csv_aj is None or csv_sin is None:
        raise RuntimeError("Suba los dos archivos CSV (ajustados y sin ajustar).")
    return datos.cargar_csv(csv_aj, csv_sin, procedencia)


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
                "q": q_usado, "fuente_q": fuente_q, "r": r_usado, "fuente_r": fuente_r,
                "parametros": calibracion.calibrar(precios_cal, q_usado, config.D),
                "cov_diaria": calibracion.covarianza_diaria(precios_cal, "simple"),
                "corr": calibracion.correlacion(precios_cal),
                "sectores": sectores,
            }
    except Exception as e:
        st.session_state.pop("res", None)
        st.error(f"No se pudo completar el análisis: {e}")

# ---------------------------------------------------------------------------
# Salidas
# ---------------------------------------------------------------------------
tabs = st.tabs(["Datos", "GBM", "Opciones", "Riesgo", "Asignación", "Evaluación"])
res = st.session_state.get("res")

if res is None:
    st.info("Configure las entradas y presione **Ejecutar análisis**.")
    st.stop()

with tabs[0]:
    limpio, ventanas, meta = res["limpio"], res["ventanas"], res["crudos"].meta
    resumen = limpio.resumen

    st.subheader("Universo y procedencia")
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

    st.subheader("Calidad de los datos")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fechas originales", resumen["fechas_originales"])
    c2.metric("Sesiones descartadas", resumen["sesiones_descartadas"])
    c3.metric("Retornos conjuntos", resumen["retornos_conjuntos"])
    c4.metric("Duplicados eliminados", resumen["duplicados_eliminados"])
    st.write(
        f"Período limpio: **{resumen['desde']}** a **{resumen['hasta']}**. "
        f"Calibración inicial: {len(ventanas['calibracion']) - 1} retornos "
        f"({ventanas['calibracion'].index[1].date()} a {ventanas['calibracion'].index[-1].date()}). "
        f"Evaluación: {len(ventanas['evaluacion']) - 1} retornos "
        f"({ventanas['evaluacion'].index[1].date()} a {ventanas['evaluacion'].index[-1].date()}). "
        f"Retornos más antiguos no usados en estas ventanas: {ventanas['retornos_no_usados']}."
    )
    st.dataframe(limpio.calidad_por_ticker, width="stretch")
    if resumen["fechas_descartadas"]:
        with st.expander(f"Ver las {resumen['sesiones_descartadas']} fechas descartadas"):
            st.write(", ".join(resumen["fechas_descartadas"]))
    st.caption(
        "Se usan solo sesiones en las que las 10 acciones tienen precio válido (ajustado y sin ajustar). "
        "Los faltantes no se rellenan: no se inventan retornos cero."
    )

    st.subheader("Precios normalizados (base 100)")
    norm = calibracion.normalizar(limpio.ajustados)
    st.plotly_chart(
        px.line(norm, labels={"value": "Índice (base 100)", "Fecha": "", "Ticker": "Ticker"}),
        width="stretch",
    )

    st.subheader("Distribución de log-retornos diarios (ventana de calibración)")
    x_cal = calibracion.log_retornos(ventanas["calibracion"])
    largo = x_cal.melt(var_name="Ticker", value_name="Log-retorno")
    fig_h = px.histogram(largo, x="Log-retorno", facet_col="Ticker", facet_col_wrap=5, nbins=60, height=480)
    fig_h.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig_h.update_xaxes(tickformat=".0%")
    st.plotly_chart(fig_h, width="stretch")

    st.subheader("Correlaciones (log-retornos, ventana de calibración)")
    st.plotly_chart(
        px.imshow(res["corr"], text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r", aspect="auto"),
        width="stretch",
    )

    st.subheader("Dividendos q y tasa libre de riesgo r")
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
    st.caption(
        "q = ln(1 + D₁₂/S), con D₁₂ los dividendos pagados en los 365 días previos a la última fecha y S el "
        "último cierre sin ajustar. r = ln(1 + IRX/100), con IRX el rendimiento anual de la letra del Tesoro "
        "a 13 semanas. Son aproximaciones: suponen que el dividendo de los próximos 12 meses será similar al "
        "de los últimos 12 y que se reparte de forma continua."
    )

    st.subheader("Parámetros calibrados (anuales, ventana de calibración)")
    st.dataframe(
        res["parametros"].style.format({
            "Media diaria x̄": "{:.5f}", "Desv. diaria s": "{:.5f}", "σ anual": "{:.2%}",
            "μ total anual": "{:.2%}", "q": "{:.2%}", "a = μ − q": "{:.2%}",
        }),
        width="stretch",
    )
    st.caption(
        "σ = √252 · s (divisor n − 1);  μ total = 252 · x̄ + ½σ²;  a = μ total − q (spot exdividendo). "
        "Calibración con los 504 retornos previos a la ventana de evaluación."
    )
    with st.expander("Covarianza diaria de retornos simples Σ_R (n − 1)"):
        st.dataframe(res["cov_diaria"].style.format("{:.6f}"), width="stretch")

    st.subheader("Descargas y versiones")
    d1, d2, d3 = st.columns(3)
    d1.download_button("Precios ajustados (CSV)", datos.a_csv_bytes(limpio.ajustados), "precios_ajustados_limpios.csv")
    d2.download_button("Precios sin ajustar (CSV)", datos.a_csv_bytes(limpio.sin_ajustar), "precios_sin_ajustar_limpios.csv")
    d3.download_button("Parámetros (CSV)", datos.a_csv_bytes(res["parametros"]), "parametros_calibracion.csv")
    with st.expander("Metadatos de la descarga y versiones de bibliotecas"):
        st.json(meta)

with tabs[1]:
    st.write("Pendiente (entrega 2): trayectorias GBM y cuantiles.")

with tabs[2]:
    st.write("Pendiente: árbol CRR (entrega 1) y comparación con mercado (entrega 2).")

with tabs[3]:
    st.write("Pendiente (entrega 2): VaR/ES histórico, normal y Monte Carlo.")

with tabs[4]:
    st.write("Pendiente (entrega 3): asignación por mínimo VaR y estrategias.")

with tabs[5]:
    st.write("Pendiente (entrega 2): rolling del GBM, RMSE y MAPE.")
