"""Pestaña Datos (Pasos 2 y 3): calidad, precios, correlaciones, parámetros y descargas.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src import calibracion, config, datos, lecturas, ui


def mostrar(ctx: dict) -> dict:
    """Dibuja la pestaña. `ctx` trae lo que ya está calculado (datos, entradas, carteras y resultados de las
    pestañas anteriores); devuelve lo que usan las pestañas siguientes y la exportación."""
    cal = ctx["cal"]
    sect = ctx["sect"]
    tickers = ctx["tickers"]
    resumen = ctx["resumen"]
    meta = ctx["meta"]
    limpio = ctx["limpio"]
    res = ctx["res"]
    ventanas = ctx["ventanas"]
    ev = ctx["ev"]

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
        st.markdown('<div class="q-nota">La correlación va de −1 a 1: cerca de 1, las dos acciones suben y bajan '
                    'al mismo tiempo; cerca de 0, se mueven de forma independiente; negativa, tienden a moverse '
                    'en direcciones opuestas.</div>', unsafe_allow_html=True)
        with st.expander("Ver los pares más y menos correlacionados"):
            st.dataframe(lecturas.pares_correlacion(res["corr"], sect).style.format({"Correlación": "{:.2f}"}),
                         hide_index=True, width="stretch")

    with st.container(key="tarjeta_param"):
        st.markdown(ui.titulo_tarjeta("Riesgo frente a retorno · ventana de calibración"), unsafe_allow_html=True)
        par = res["parametros"].reset_index()
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
        with st.expander("Ver la tabla de parámetros calibrados"):
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
        with st.expander("Ver la covarianza diaria de retornos simples Σ_R (n − 1)"):
            st.dataframe(res["cov_diaria"].style.format("{:.6f}"), width="stretch")

    x_cal = calibracion.log_retornos(cal)
    with st.container(key="tarjeta_hist"):
        st.markdown(ui.titulo_tarjeta("¿Cuánto se mueve cada acción en un día?"), unsafe_allow_html=True)
        largo = x_cal.melt(var_name="Ticker", value_name="Log-retorno")
        largo["Sector"] = largo["Ticker"].map(sect)
        colores = ui.colores_sector([sect[t] for t in tickers])
        fig_b = px.box(largo, x="Ticker", y="Log-retorno", color="Sector", color_discrete_map=colores,
                       category_orders={"Ticker": lecturas.orden_por_sector(tickers, sect)})
        fig_b.update_traces(marker=dict(size=3, opacity=.6), line=dict(width=1.2))
        fig_b.update_yaxes(tickformat=".0%", title_text="Log-retorno diario")
        fig_b.update_xaxes(title_text="")
        fig_b.update_layout(legend=dict(orientation="h", y=1.08, x=0, title=""), boxmode="overlay")
        st.plotly_chart(ui.estilo_fig(fig_b, 380), width="stretch", config=ui.PLOTLY_CONFIG)
        st.markdown(ui.html_lectura(lecturas.lectura_dispersion(x_cal)), unsafe_allow_html=True)

    with st.container(key="tarjeta_hist_detalle"):
        st.markdown(ui.titulo_tarjeta("¿Se cumple la curva normal que supone el GBM?"), unsafe_allow_html=True)
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
            st.plotly_chart(ui.estilo_fig(fig_d, 340), width="stretch", config=ui.PLOTLY_CONFIG)
        with h2:
            st.markdown(ui.html_lectura(lecturas.lectura_histograma(serie, elegido)), unsafe_allow_html=True)
            st.markdown('<div class="q-nota">La línea punteada es la curva normal que supone el GBM. Si las barras se '
                        'salen de la curva en los extremos, hay más días de movimientos fuertes de los que el modelo '
                        'espera.</div>', unsafe_allow_html=True)
        with st.expander("Ver mejor y peor día de cada acción"):
            st.dataframe(
                lecturas.resumen_distribucion(x_cal).style.format({
                    "Mejor día": "{:.2%}", "Peor día": "{:.2%}", "Esperados si fuera normal": "{:.1f}",
                }),
                width="stretch",
            )

    with st.container(key="tarjeta_fuentes"):
        st.markdown(ui.titulo_tarjeta("Datos, fuentes y descargas"), unsafe_allow_html=True)
        consulta_txt = str(meta.get("consulta", meta.get("cargado_en", "—"))).replace("T", " ")
        st.markdown(f'<div class="q-nota">Fuente: {meta.get("fuente", "—")} · consulta: {consulta_txt} · período '
                    f'limpio: {resumen["desde"]} a {resumen["hasta"]} · {resumen["fechas_originales"]} fechas originales, '
                    f'{resumen["sesiones_descartadas"]} descartadas. r continua usada: {res["r"]:.2%} '
                    f'({res["fuente_r"]}).</div>', unsafe_allow_html=True)
        d1, d2, d3 = st.columns(3)
        d1.download_button("Precios ajustados (CSV)", datos.a_csv_bytes(limpio.ajustados),
                           "precios_ajustados_limpios.csv", width="stretch")
        d2.download_button("Precios sin ajustar (CSV)", datos.a_csv_bytes(limpio.sin_ajustar),
                           "precios_sin_ajustar_limpios.csv", width="stretch")
        d3.download_button("Parámetros (CSV)", datos.a_csv_bytes(res["parametros"]),
                           "parametros_calibracion.csv", width="stretch")

        with st.expander("Ver calidad de los datos"):
            st.write(
                f"Calibración inicial: {len(cal) - 1} retornos ({cal.index[1].date()} a {cal.index[-1].date()}). "
                f"Evaluación: {len(ev) - 1} retornos ({ev.index[1].date()} a {ev.index[-1].date()}). "
                f"Retornos más antiguos no usados en estas ventanas: {ventanas['retornos_no_usados']}."
            )
            st.dataframe(limpio.calidad_por_ticker.style.format(
                {"Primera fecha": "{:%Y-%m-%d}", "Última fecha": "{:%Y-%m-%d}"}), width="stretch")
            if resumen["fechas_descartadas"]:
                st.write("Fechas descartadas: " + ", ".join(resumen["fechas_descartadas"]))
            st.markdown('<div class="q-nota">Se usan solo sesiones en las que las 10 acciones tienen precio válido '
                        '(ajustado y sin ajustar). Los faltantes no se rellenan: no se inventan retornos cero.</div>',
                        unsafe_allow_html=True)

        with st.expander("Ver universo y procedencia"):
            sect_yahoo = meta.get("sectores_yahoo", {})
            st.dataframe(
                pd.DataFrame({
                    "Ticker": tickers,
                    "Sector usado": [res["sectores"][t] for t in tickers],
                    "Sector Yahoo": [sect_yahoo.get(t, "—") or "no disponible" for t in tickers],
                    "Fuente": meta.get("fuente", "—"),
                    "Fecha de consulta": consulta_txt,
                }),
                hide_index=True, width="stretch",
            )
            if "procedencia" in meta:
                st.caption(f"Procedencia declarada: {meta['procedencia']}")

        with st.expander("Ver dividendos q y tasa libre de riesgo r"):
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
            st.markdown('<div class="q-nota">q = ln(1 + D₁₂/S), con D₁₂ los dividendos pagados en los 365 días previos '
                        'a la última fecha y S el último cierre sin ajustar. r = ln(1 + IRX/100), con IRX el rendimiento '
                        'anual de la letra del Tesoro a 13 semanas. Suponen que el dividendo de los próximos 12 meses será '
                        'similar al de los últimos 12 y que se reparte de forma continua.</div>', unsafe_allow_html=True)

        with st.expander("Ver metadatos de la descarga y versiones de bibliotecas"):
            st.json(meta)

    return {}
