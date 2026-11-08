"""Pestaña Evaluación (Paso 13): evaluación rolling del GBM.
"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src import lecturas, ui

from . import memoria


def mostrar(ctx: dict) -> dict:
    """Dibuja la pestaña. `ctx` trae lo que ya está calculado (datos, entradas, carteras y resultados de las
    pestañas anteriores); devuelve lo que usan las pestañas siguientes y la exportación."""
    estrategias = ctx["estrategias"]
    limpio = ctx["limpio"]
    ev_v = ctx["ev_v"]
    tickers = ctx["tickers"]
    # Resultados que se comparten; quedan en None si esa parte no se calcula
    det_r = res_r = None

    h_r = int(estrategias["h_rolling"])
    det_r, res_r = memoria.rolling_gbm(limpio.ajustados, ev_v.index[0], h_r, 252)
    gana = int((res_r["¿Le gana al ingenuo?"] == "Sí").sum())
    st.markdown(ui.html_kpis([
        ("MAPE promedio · GBM", f"{res_r['MAPE GBM (%)'].mean():.3f} %".replace(".", ","), f"pronóstico a {h_r} sesión(es)"),
        ("MAPE promedio · ingenuo", f"{res_r['MAPE ingenuo (%)'].mean():.3f} %".replace(".", ","), "mañana = precio de hoy"),
        ("El GBM le gana al ingenuo", f"{gana} de {len(res_r)}", "acciones con menor MAPE"),
        ("Precio dentro de la banda P10–P90", f"{res_r['Cobertura P10–P90'].mean():.0%}", "se espera alrededor de 80 %"),
    ]), unsafe_allow_html=True)
    st.markdown(ui.html_lectura(lecturas.lectura_evaluacion(res_r, h_r)), unsafe_allow_html=True)

    with st.container(key="tarjeta_rolling"):
        st.markdown(ui.titulo_tarjeta("Pronóstico frente a precio real"), unsafe_allow_html=True)
        r1, _ = st.columns([1, 2])
        with r1:
            t_r = st.selectbox("Acción", tickers, key="rolling_ticker")
        d_r = det_r[t_r]
        fila_r = res_r.set_index("Ticker").loc[t_r]
        fig_r = go.Figure()
        fig_r.add_trace(go.Scatter(x=d_r["Fecha objetivo"], y=d_r["P90"], mode="lines", line=dict(width=0),
                                   showlegend=False, hoverinfo="skip"))
        fig_r.add_trace(go.Scatter(x=d_r["Fecha objetivo"], y=d_r["P10"], mode="lines", line=dict(width=0), fill="tonexty",
                                   fillcolor="rgba(217,181,106,.15)", name="Banda P10–P90 del GBM", hoverinfo="skip"))
        fig_r.add_trace(go.Scatter(x=d_r["Fecha objetivo"], y=d_r["Observado"], mode="lines", name="Precio real",
                                   line=dict(color=ui.TEXTO, width=1.6)))
        fig_r.add_trace(go.Scatter(x=d_r["Fecha objetivo"], y=d_r["Pronóstico GBM"], mode="lines", name="Pronóstico GBM",
                                   line=dict(color=ui.ORO_CLARO, width=1.6, dash="dot")))
        fig_r.update_yaxes(title_text="Precio ajustado (USD)")
        fig_r.update_layout(legend=dict(orientation="h", y=1.12, x=0), hovermode="x unified")
        st.plotly_chart(ui.estilo_fig(fig_r, 360), width="stretch", config=ui.PLOTLY_CONFIG)
        fig_err = go.Figure(go.Bar(x=d_r["Fecha objetivo"], y=d_r["Error GBM"], name="Error del GBM",
                                   marker_color=np.where(d_r["Error GBM"] >= 0, ui.PALETA[5], ui.PALETA[8])))
        fig_err.update_yaxes(title_text="Error (USD)")
        fig_err.update_layout(bargap=0, showlegend=False)
        st.plotly_chart(ui.estilo_fig(fig_err, 200), width="stretch", config=ui.PLOTLY_CONFIG)
        st.markdown(ui.html_lectura(lecturas.lectura_evaluacion_accion(fila_r, t_r)), unsafe_allow_html=True)

    with st.container(key="tarjeta_rolling_tabla"):
        st.markdown(ui.titulo_tarjeta("RMSE y MAPE de las 10 acciones"), unsafe_allow_html=True)
        st.dataframe(res_r.drop(columns=["Desde", "Hasta"]).style.format({
            "RMSE GBM": "{:.3f}", "RMSE ingenuo": "{:.3f}", "MAPE GBM (%)": "{:.3f}", "MAPE ingenuo (%)": "{:.3f}",
            "Cobertura P10–P90": "{:.0%}"}), hide_index=True, width="stretch")
        with st.expander("Ver detalle del método"):
            st.markdown(f"""
- **Ventana móvil:** para cada fecha de origen t se usan los 252 log-retornos disponibles hasta t (253 precios
  ajustados) y se reestiman μ̂ y σ̂. La ventana avanza una sesión por vez; no se usa información posterior a t.
- **Pronóstico (Ec. 12):** P̂ = P_t · exp(μ̂ · H / 252), la media condicional del GBM. Ingenuo: P̂ = P_t.
- **Métricas (Ec. 13):** RMSE en unidades del precio ajustado; MAPE en %. Solo objetivos observados y positivos.
- **Prueba:** H = {h_r} sesión(es), m = {int(res_r['m'].min())} pares por acción, objetivos del
  {res_r['Desde'].min():%Y-%m-%d} al {res_r['Hasta'].max():%Y-%m-%d}. Los 252 días son la ventana de calibración,
  no el horizonte pronosticado.
- **Banda P10–P90** (complemento): cuantiles del GBM a H sesiones con los mismos μ̂ y σ̂.
""")

    return {"det_r": det_r, "res_r": res_r}
