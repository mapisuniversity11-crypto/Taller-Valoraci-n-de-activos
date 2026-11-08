"""Botón de exportación (entregable): ZIP con un CSV por resultado y el resumen de parámetros y semilla.

Se dibuja arriba de las pestañas, pero se llena al final, porque usa lo calculado en todas.
"""
import pandas as pd
import streamlit as st

from src import config, exportar, gbm


def mostrar(ctx: dict, zona_exportar) -> None:
    """Arma el ZIP con los resultados de todas las pestañas (en `ctx`) y dibuja el botón en `zona_exportar`."""
    carteras = ctx["carteras"]
    par_hoy = ctx["par_hoy"]
    limpio = ctx["limpio"]
    res = ctx["res"]
    semilla_base = ctx["semilla_base"]
    B = ctx["B"]
    alpha_v = ctx["alpha_v"]
    h_v = ctx["h_v"]
    sect = ctx["sect"]
    resumen = ctx["resumen"]
    tickers = ctx["tickers"]
    opcion = ctx["opcion"]
    meta = ctx["meta"]
    cartera = ctx["cartera"]
    estrategias = ctx["estrategias"]
    cal_v = ctx["cal_v"]
    ev_v = ctx["ev_v"]
    precios_hoy = ctx["precios_hoy"]
    s0_hoy = ctx["s0_hoy"]

    def _g(nombre: str):
        """Resultado de una pestaña, o None si esa parte no se calculó (p. ej., sin cadena de opciones)."""
        return ctx.get(nombre)


    def _var_tabla():
        filas = []
        for r_ in (_g("r_hist"), _g("r_norm"), _g("r_mc")):
            if r_:
                filas.append({"Método": r_["metodo"], "VaR (USD)": r_["var"], "ES (USD)": r_["es"]})
        return pd.DataFrame(filas).set_index("Método") if filas else None


    proy = pd.DataFrame([
        {"Ticker": t, "Horizonte": h_n, "Sesiones": h_s, "S0": float(s0_hoy[t]),
         "P10": gbm.cuantil_gbm(float(s0_hoy[t]), float(par_hoy.loc[t, "a = μ − q"]), float(par_hoy.loc[t, "σ anual"]),
                                h_s / config.D, 0.10),
         "P50": gbm.cuantil_gbm(float(s0_hoy[t]), float(par_hoy.loc[t, "a = μ − q"]), float(par_hoy.loc[t, "σ anual"]),
                                h_s / config.D, 0.50),
         "P90": gbm.cuantil_gbm(float(s0_hoy[t]), float(par_hoy.loc[t, "a = μ − q"]), float(par_hoy.loc[t, "σ anual"]),
                                h_s / config.D, 0.90)}
        for t in tickers for h_n, h_s in gbm.HORIZONTES.items()]).set_index("Ticker")

    c_el_x, probs_x, cmp_x = _g("c_el"), _g("probs"), (_g("contrato_mod") or {}).get("cmp")
    contrato_x = None
    if c_el_x is not None:
        contrato_x = pd.DataFrame([{
            "Contrato": c_el_x["Contrato"], "Subyacente": res.get("ticker_opcion"), "Tipo": opcion["tipo"],
            "Strike": c_el_x["Strike"], "Vencimiento": c_el_x["Vencimiento"], "Días": c_el_x["Días"],
            "Bid": c_el_x["Bid"], "Ask": c_el_x["Ask"], "Prima pagada": _g("prima"),
            "CRR americana": _g("v_am_m"), "CRR europea": _g("v_eu_m"),
            "Merton europea": _g("v_mer"), "Precio medio mercado": (cmp_x or {}).get("medio"),
            "Diferencia δ": (cmp_x or {}).get("delta"),
            "P(ejercicio)": (probs_x or {}).get("p_ejercicio"), "P(ganancia)": (probs_x or {}).get("p_beneficio"),
            "Punto de equilibrio": (probs_x or {}).get("punto_equilibrio"),
        }]).set_index("Contrato")

    crr_x = None
    if _g("v_eu") is not None:
        crr_x = pd.DataFrame([{"Tipo": opcion["tipo"], "S0": opcion["s0"], "K": opcion["k"], "T": opcion["t"],
                               "r": opcion["r"], "q": opcion["q"], "σ": opcion["sigma"], "N": int(opcion["n"]),
                               "Europea": _g("v_eu"), "Americana": _g("v_am")}]).set_index("Tipo")

    pesos_x = pd.DataFrame(carteras)
    pesos_x.insert(0, "Sector", [sect[t] for t in pesos_x.index])
    info_bt_x = _g("info_bt")
    pres_x = _g("pres")
    mer_x = _g("mer")
    det_r_x = _g("det_r")

    tablas_x = {
        "01_precios_ajustados": limpio.ajustados,
        "02_precios_sin_ajustar": limpio.sin_ajustar,
        "03_calidad_por_ticker": limpio.calidad_por_ticker,
        "04_parametros_calibracion": res["parametros"],
        "05_correlaciones": res["corr"],
        "06_covarianza_diaria": res["cov_diaria"],
        "07_parametros_gbm_hoy": par_hoy,
        "08_gbm_proyecciones_P10_P50_P90": proy,
        "09_crr_calculadora": crr_x,
        "10_opciones_candidatos": _g("cands"),
        "11_contrato_elegido": contrato_x,
        "12_var_es_por_metodo": _var_tabla(),
        "13_excepciones_ultimo_anio": _g("exc"),
        "14_estres": _g("tabla_e"),
        "15_pesos_carteras": pesos_x,
        "16_orden_de_compra": _g("compra"),
        "17_estrategias_resultados": pd.DataFrame(info_bt_x).T if info_bt_x else None,
        "18_estrategias_valor_diario": _g("bt"),
        "19_presupuesto_riesgo": pd.DataFrame({"Invertido (USD)": pres_x.invertido}) if pres_x else None,
        "20_rolling_metricas": _g("res_r").set_index("Ticker") if _g("res_r") is not None else None,
        "21_rolling_detalle": (pd.concat({t: d for t, d in det_r_x.items()}, names=["Ticker", "Fecha origen"])
                               if det_r_x else None),
        "22_modelos_hurst": _g("tab_h").set_index("Ticker") if _g("tab_h") is not None else None,
        "23_modelos_merton_saltos": (pd.DataFrame({"Movimiento del día": mer_x.tamanos}) if mer_x else None),
    }

    conf_x = _g("cartera_v")
    resumen_x = {
        "Ejecución": {
            "Exportado": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
            "Fuente de datos": meta.get("fuente", "—"),
            "Fecha de consulta": str(meta.get("consulta", meta.get("cargado_en", "—"))).replace("T", " "),
            "Período limpio": f"{resumen['desde']} a {resumen['hasta']}",
            "Retornos conjuntos": resumen["retornos_conjuntos"],
            "Sesiones descartadas": resumen["sesiones_descartadas"],
        },
        "Semillas": {
            "Semilla de la app": semilla_base,
            "GBM (por acción)": f"semilla + posición del ticker ({semilla_base} a {semilla_base + len(tickers) - 1})",
            "VaR Monte Carlo": semilla_base,
            "Probabilidades de la opción": semilla_base,
            "Hurst (permutaciones)": "semilla + posición del ticker",
        },
        "Universo": {t: sect[t] for t in tickers},
        "Ventanas": {
            "Calibración": f"{len(cal_v) - 1} retornos · {cal_v.index[1]:%Y-%m-%d} a {cal_v.index[-1]:%Y-%m-%d}",
            "Evaluación": f"{len(ev_v) - 1} retornos · {ev_v.index[1]:%Y-%m-%d} a {ev_v.index[-1]:%Y-%m-%d}",
            "GBM y Opciones (hoy)": f"{len(precios_hoy) - 1} retornos hasta {precios_hoy.index[-1]:%Y-%m-%d}",
            "Rolling": f"252 retornos · H = {int(estrategias['h_rolling'])} sesión(es)",
        },
        "Cartera y riesgo": {
            "Capital B (USD)": B, "Confianza α": alpha_v, "Horizonte VaR (días)": h_v,
            "Cartera evaluada en Riesgo": conf_x, "Costo de transacción": float(cartera["costo"]),
            "Peso mínimo": config.W_MIN, "Peso máximo": config.W_MAX, "Tope por sector": config.MAX_SECTOR,
            "Presupuesto de riesgo V (USD)": float(estrategias["presupuesto_v"]),
            "Escenarios VaR Monte Carlo": config.N_ESCENARIOS_VAR,
            "Excepciones último año": _g("n_exc"), "Zona Basilea": _g("zona"),
        },
        "Tasas": {"r continua": res["r"], "Fuente de r": res["fuente_r"],
                  **{f"q {t}": res["q"][t] for t in tickers}},
        "Opción": {"Subyacente": res.get("ticker_opcion"), "Tipo": opcion["tipo"],
                   "Trayectorias GBM": config.N_TRAYECTORIAS_GBM,
                   "Contrato elegido": c_el_x["Contrato"] if c_el_x is not None else "sin cadena de opciones"},
        "Versiones": meta.get("versiones", {}),
    }

    with zona_exportar.container():
        x1, x2 = st.columns([3, 1])
        with x2:
            st.download_button("Exportar resultados (CSV)", exportar.armar_zip(tablas_x, resumen_x),
                               f"quantia_resultados_{pd.Timestamp.now():%Y%m%d_%H%M}.zip", mime="application/zip",
                               width="stretch", key="exportar_todo",
                               help="ZIP con un CSV por resultado y el resumen de parámetros y semilla.")
        with x1:
            st.markdown('<div class="q-nota" style="padding-top:10px">Un archivo ZIP con todos los resultados en CSV '
                        '(datos, parámetros, proyecciones, opción, VaR, carteras, estrategias, evaluación y modelos) y '
                        'el resumen de parámetros y semilla para reproducir la ejecución.</div>', unsafe_allow_html=True)
