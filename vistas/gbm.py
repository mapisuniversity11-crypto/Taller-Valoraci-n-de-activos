"""Pestaña GBM (Paso 4): proyección del precio bajo la medida física P.

Parámetros calibrados con los últimos 504 retornos hasta hoy (D15); S0 = último cierre sin ajustar;
drift a = μ total − q.
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import config, gbm, lecturas, ui


def mostrar(ctx: dict) -> dict:
    """Dibuja la pestaña. `ctx` trae lo que ya está calculado (datos, entradas, carteras y resultados de las
    pestañas anteriores); devuelve lo que usan las pestañas siguientes y la exportación."""
    tickers = ctx["tickers"]
    semilla_base = ctx["semilla_base"]
    s0_hoy = ctx["s0_hoy"]
    par_hoy = ctx["par_hoy"]
    limpio = ctx["limpio"]
    sect = ctx["sect"]
    precios_hoy = ctx["precios_hoy"]

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

    return {}
