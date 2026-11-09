"""App del Taller de Valoración de Activos (Streamlit).

Ejecutar:  streamlit run app.py

Esta capa SOLO maneja interfaz: lee entradas, llama a los módulos de src/
y muestra resultados. Ningún cálculo financiero debe vivir aquí.

Organización:
  app.py     portada, panel de entradas, ejecución del análisis y orden de las pestañas
  vistas/    un archivo por pestaña (datos, gbm, opciones, riesgo, asignacion, evaluacion,
             modelos) y el botón de exportación
  src/       los cálculos (sin Streamlit), con pruebas en tests/
"""
import time
from datetime import date, timedelta

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import vistas
from src import asignacion, calibracion, config, datos, opciones, ui

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
    costo = st.number_input("Costo de transacción (% del monto negociado)", value=config.COSTO_TRANSACCION * 100,
                            min_value=0.0, max_value=5.0, step=0.05, format="%.2f", key="costo_pct") / 100
    return dict(capital=capital, alpha=alpha, h_var=h_var, semilla=semilla, auto_q_r=auto_q_r, r=r, costo=costo)


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


def obtener_crudos(fuente: str):
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


def _fecha_respaldo(meta: dict) -> str:
    return str(meta.get("consulta", meta.get("cargado_en", "fecha desconocida")))[:10]


def preparar_datos(fuente: str):
    """Descarga (o carga) y limpia. Si Yahoo Finance falla y hay respaldo en data/raw, usa el respaldo
    automáticamente y devuelve un aviso, para que la app nunca se quede en un error (D25)."""
    try:
        crudos = obtener_crudos(fuente)
        limpio = datos.limpiar(crudos, tickers)
        ventanas = datos.separar_ventanas(limpio.ajustados, config.N_CALIBRACION, config.N_EVALUACION)
        return crudos, limpio, ventanas, None
    except Exception as e_yahoo:
        if fuente != "Yahoo Finance" or not datos.existe_respaldo():
            raise
        crudos = datos.cargar_respaldo()
        limpio = datos.limpiar(crudos, tickers)
        ventanas = datos.separar_ventanas(limpio.ajustados, config.N_CALIBRACION, config.N_EVALUACION)
        aviso = (f"Yahoo Finance no respondió, así que se usaron los datos de respaldo guardados el "
                 f"{_fecha_respaldo(crudos.meta)} (data/raw). Detalle: {str(e_yahoo)[:160]}")
        return crudos, limpio, ventanas, aviso


if ejecutar:
    try:
        with st.spinner("Obteniendo y validando datos..."):
            crudos, limpio, ventanas, aviso_fuente = preparar_datos(fuente)
            precios_cal = ventanas["calibracion"]
            q_usado, fuente_q, r_usado, fuente_r = elegir_q_y_r(crudos.meta)
            # Cadena de opciones del subyacente (Paso 5). Si falla, el resto del análisis sigue.
            cadena, meta_cadena, error_cadena = None, None, None
            descargar_op = fuente == "Yahoo Finance" and aviso_fuente is None
            try:
                if descargar_op:
                    cadena, meta_cadena = opciones.descargar_cadena(opcion["ticker"])
                    opciones.guardar_cadena(cadena, meta_cadena)
            except Exception as e_op:
                error_cadena = f"No se pudo descargar la cadena de opciones: {e_op}"
            if cadena is None:
                # Sin descarga (respaldo o CSV) o si la descarga falló: cadena guardada en data/raw
                cargada = opciones.cargar_cadena(opcion["ticker"])
                if cargada is not None:
                    cadena, meta_cadena = cargada
                    if descargar_op:
                        fecha_c = str(meta_cadena.get("consulta", "fecha desconocida"))[:10]
                        aviso_cadena = (f"Yahoo Finance no entregó la cadena de opciones; se usó la guardada el "
                                        f"{fecha_c}.")
                        aviso_fuente = f"{aviso_fuente} {aviso_cadena}" if aviso_fuente else aviso_cadena
                    error_cadena = None
                elif error_cadena is None:
                    error_cadena = "No hay una cadena de opciones guardada en data/raw para este subyacente."
            # Histórico de la tasa del Tesoro (^IRX) para el modelo de Vasicek (pestaña Modelos).
            # Si no se puede descargar, se usa el guardado en data/raw; si tampoco hay, esa parte se omite.
            irx = None
            if descargar_op:
                irx = datos.historial_irx_yahoo(fecha_inicio, fecha_fin)
                if irx is not None and periodo["guardar"]:
                    datos.guardar_irx(irx)
            if irx is None:
                irx = datos.cargar_irx()
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
                "aviso_fuente": aviso_fuente,
                "irx": irx,
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

if res.get("aviso_fuente"):
    st.warning(res["aviso_fuente"], icon="⚠️")

# El botón de exportación se dibuja aquí, pero se llena al final, cuando ya están todos los resultados.
zona_exportar = st.empty()

# ---------------------------------------------------------------------------
# Cálculos compartidos por varias pestañas
# ---------------------------------------------------------------------------
cal, ev = ventanas["calibracion"], ventanas["evaluacion"]
sect = res["sectores"]

# Ventana "hoy" para GBM, Opciones y Modelos: los últimos 504 retornos hasta la última fecha (D15).
precios_hoy = limpio.ajustados.iloc[-(config.N_CALIBRACION + 1):]
par_hoy = calibracion.calibrar(precios_hoy, res["q"], config.D)
s0_hoy = limpio.sin_ajustar.iloc[-1]
semilla_base = int(st.session_state.get("semilla", config.SEMILLA))

# Carteras (Paso 11), estimadas con la ventana de calibración. Se usan en Riesgo y en Asignación.
B, alpha_v, h_v = float(cartera["capital"]), float(cartera["alpha"]), int(cartera["h_var"])
cal_v, ev_v = ventanas["calibracion"], ventanas["evaluacion"]
cov_v = calibracion.covarianza_diaria(cal_v, "simple")
opt = asignacion.minimo_var(cov_v, sect, config.W_MIN, config.W_MAX, config.MAX_SECTOR)
w_inv = asignacion.pesos_inverso_var(cov_v, B, alpha_v, h_v)
w_inv_corr, pasos_corr = asignacion.corregir_sectores(w_inv, sect, config.MAX_SECTOR, config.W_MIN, config.W_MAX)
carteras = {"Pesos iguales": asignacion.pesos_iguales(tickers), "Inverso al VaR": w_inv,
            "Inverso al VaR · corregido por sector": w_inv_corr}
if opt.exito:
    carteras["Mínimo VaR"] = opt.pesos

# ---------------------------------------------------------------------------
# Pestañas: cada una vive en su propio archivo de vistas/. Todas reciben el mismo
# contexto `ctx` y devuelven lo que necesitan las siguientes y la exportación.
# ---------------------------------------------------------------------------
ctx = {
    "alpha_v": alpha_v,
    "B": B,
    "cal": cal,
    "cal_v": cal_v,
    "cartera": cartera,
    "carteras": carteras,
    "cov_v": cov_v,
    "estrategias": estrategias,
    "ev": ev,
    "ev_v": ev_v,
    "h_v": h_v,
    "limpio": limpio,
    "meta": meta,
    "opcion": opcion,
    "opt": opt,
    "par_hoy": par_hoy,
    "pasos_corr": pasos_corr,
    "precios_hoy": precios_hoy,
    "res": res,
    "resumen": resumen,
    "s0_hoy": s0_hoy,
    "sect": sect,
    "semilla_base": semilla_base,
    "tickers": tickers,
    "ventanas": ventanas,
    "w_inv_corr": w_inv_corr,
}

tabs = st.tabs(["Datos", "GBM", "Opciones", "Riesgo", "Asignación", "Evaluación", "Modelos"])
pestanas = [vistas.datos, vistas.gbm, vistas.opciones, vistas.riesgo, vistas.asignacion, vistas.evaluacion,
            vistas.modelos]
for tab, vista in zip(tabs, pestanas):
    with tab:
        ctx.update(vista.mostrar(ctx))

vistas.exportacion.mostrar(ctx, zona_exportar)
