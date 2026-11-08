"""Pestaña Riesgo (Paso 10): VaR y ES por tres métodos, excepciones, fuera de muestra y estrés.

Estimación con la ventana de calibración; el último año queda para evaluar (D19).
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import config, lecturas, riesgo, ui

from . import memoria


def mostrar(ctx: dict) -> dict:
    """Dibuja la pestaña. `ctx` trae lo que ya está calculado (datos, entradas, carteras y resultados de las
    pestañas anteriores); devuelve lo que usan las pestañas siguientes y la exportación."""
    carteras = ctx["carteras"]
    cal_v = ctx["cal_v"]
    B = ctx["B"]
    alpha_v = ctx["alpha_v"]
    h_v = ctx["h_v"]
    cov_v = ctx["cov_v"]
    semilla_base = ctx["semilla_base"]
    limpio = ctx["limpio"]
    sect = ctx["sect"]
    ev_v = ctx["ev_v"]
    cartera = ctx["cartera"]
    # Resultados que se comparten; quedan en None si esa parte no se calcula
    cartera_v = exc = n_exc = r_hist = r_mc = r_norm = tabla_e = w_v = zona = None

    nombres_c = list(carteras)
    c_sel, _ = st.columns([1, 2])
    with c_sel:
        cartera_v = st.selectbox("Cartera evaluada", nombres_c, key="riesgo_cartera",
                                 index=nombres_c.index("Mínimo VaR") if "Mínimo VaR" in nombres_c else 0)
    w_v = carteras[cartera_v]
    r_hist = riesgo.var_historico(cal_v, w_v, B, alpha_v, h_v)
    r_norm = riesgo.var_normal(cov_v, w_v, B, alpha_v, h_v)
    r_mc = memoria.var_montecarlo(cal_v, w_v, B, alpha_v, h_v, config.N_ESCENARIOS_VAR, semilla_base)
    exc = memoria.excepciones(limpio.ajustados, w_v, B, alpha_v, ev_v.index[1])
    n_exc = int(exc["Excepción"].sum())
    zona = riesgo.zona_basilea(n_exc, len(exc), alpha_v)
    conf = f"{alpha_v:.0%} · {h_v} días"

    st.markdown(f'<div class="q-nota">Cartera evaluada: <b>{cartera_v}</b>, {lecturas._usd(B)}. Los pesos de cada '
                'cartera se ven en la pestaña Asignación.</div>', unsafe_allow_html=True)
    st.markdown(ui.html_kpis([
        (f"VaR Monte Carlo · {conf}", lecturas._usd(r_mc["var"]), f"{r_mc['var'] / B:.1%} del capital".replace(".", ",")),
        ("Pérdida promedio en la cola (ES)", lecturas._usd(r_mc["es"]), "promedio del peor 1 % de escenarios"),
        (f"Peor pérdida real de {h_v} días", lecturas._usd(r_hist["peor"]), f"{r_hist['fecha_peor']:%Y-%m-%d}"),
        ("Excepciones en el último año", f"{n_exc} de {len(exc)}", f"esperadas ≈ {len(exc) * (1 - alpha_v):.1f} · zona {zona}"
         .replace(".", ",")),
    ]), unsafe_allow_html=True)

    with st.container(key="tarjeta_var"):
        st.markdown(ui.titulo_tarjeta(f"¿Cuánto podría perder la cartera en {h_v} días?"), unsafe_allow_html=True)
        v1, v2 = st.columns([1.5, 1], gap="medium")
        with v1:
            fig_v = px.histogram(x=r_mc["perdidas"], nbins=90, color_discrete_sequence=[ui.ORO])
            fig_v.update_traces(marker_line_width=0, opacity=.8,
                                hovertemplate="Pérdida %{x:,.0f} USD: %{y} escenarios<extra></extra>")
            fig_v.add_vline(x=0, line_color="#3a3224")
            fig_v.add_vline(x=r_mc["var"], line_color=ui.ORO_CLARO, line_dash="dot", annotation_text="VaR",
                            annotation_font_color=ui.ORO_CLARO)
            fig_v.add_vline(x=r_mc["es"], line_color=ui.PALETA[8], line_dash="dot", annotation_text="ES",
                            annotation_font_color=ui.PALETA[8], annotation_position="top right")
            fig_v.update_xaxes(title_text=f"Pérdida en {h_v} días (USD) · negativa = ganancia")
            fig_v.update_yaxes(title_text="Escenarios simulados")
            fig_v.update_layout(showlegend=False, bargap=0.02)
            st.plotly_chart(ui.estilo_fig(fig_v, 380), width="stretch", config=ui.PLOTLY_CONFIG)
        with v2:
            st.markdown(ui.html_lectura(lecturas.lectura_var(B, alpha_v, h_v, r_mc["var"], r_mc["es"], r_hist["peor"],
                                                             "Monte Carlo")), unsafe_allow_html=True)
            metodos = pd.DataFrame([
                {"Método": "Histórico", "VaR": r_hist["var"], "ES": r_hist["es"]},
                {"Método": "Normal", "VaR": r_norm["var"], "ES": r_norm["es"]},
                {"Método": "Monte Carlo", "VaR": r_mc["var"], "ES": r_mc["es"]},
            ])
            metodos["% del capital"] = metodos["VaR"] / B
            st.dataframe(metodos.style.format({"VaR": "{:,.0f}", "ES": "{:,.0f}", "% del capital": "{:.1%}"}),
                         hide_index=True, width="stretch")
        st.markdown(ui.html_lectura(lecturas.lectura_metodos(r_hist["var"], r_norm["var"], r_mc["var"])),
                    unsafe_allow_html=True)
        with st.expander("Ver detalle de los métodos"):
            st.markdown(f"""
- **Histórico:** {r_hist['n']} bloques de {h_v} días **solapados** de la ventana de calibración (un bloque por cada día
  de inicio). Hay más bloques, pero los vecinos comparten días y no son independientes.
- **Normal (Ec. 9):** VaR = B · Φ⁻¹(α) · √(h · wᵀΣ_R w), con Σ_R la covarianza diaria de retornos simples y media cero.
- **Monte Carlo:** {f"{config.N_ESCENARIOS_VAR:,}".replace(",", ".")} escenarios GBM conjuntos bajo P (correlaciones de la ventana de
  calibración, factorización de Cholesky), cartera mantenida sin rebalanceo durante los {h_v} días.
- **Cuantil:** método "higher" (la menor pérdida observada que deja al menos el {alpha_v:.0%} de los casos por debajo).
  **ES:** promedio de las pérdidas mayores o iguales al VaR (los empates entran en la cola).
- **Ventana de estimación:** {cal_v.index[1]:%Y-%m-%d} a {cal_v.index[-1]:%Y-%m-%d}. El último año se usa solo para evaluar.
""")

    x1, x2 = st.columns(2, gap="medium")
    with x1, st.container(key="tarjeta_exc"):
        st.markdown(ui.titulo_tarjeta("¿El VaR funcionó? · último año"), unsafe_allow_html=True)
        fig_e = go.Figure()
        fig_e.add_trace(go.Bar(x=exc.index, y=exc["Pérdida real"], name="Pérdida real diaria",
                               marker_color=np.where(exc["Excepción"], ui.PALETA[8], "rgba(217,181,106,.45)"),
                               hovertemplate="%{x|%Y-%m-%d}: %{y:,.0f} USD<extra></extra>"))
        fig_e.add_trace(go.Scatter(x=exc.index, y=exc["VaR diario"], name="VaR diario", mode="lines",
                                   line=dict(color=ui.ORO_CLARO, width=2)))
        fig_e.update_yaxes(title_text="USD (positivo = pérdida)")
        fig_e.update_layout(legend=dict(orientation="h", y=1.12, x=0), bargap=0)
        st.plotly_chart(ui.estilo_fig(fig_e, 300), width="stretch", config=ui.PLOTLY_CONFIG)
        st.markdown(ui.html_lectura(lecturas.lectura_excepciones(n_exc, len(exc), alpha_v, zona)), unsafe_allow_html=True)
    with x2, st.container(key="tarjeta_estres"):
        st.markdown(ui.titulo_tarjeta("Escenarios de estrés"), unsafe_allow_html=True)
        tabla_e, info_e = riesgo.estres(w_v, sect, cov_v, B, alpha_v, h_v)
        tabla_e["Veces el VaR"] = tabla_e["Pérdida"] / r_norm["var"]
        st.dataframe(tabla_e.style.format({"Pérdida": "{:,.0f}", "Veces el VaR": "{:.1f}"}),
                     hide_index=True, width="stretch")
        st.markdown(ui.html_lectura(lecturas.lectura_estres(tabla_e, r_norm["var"])), unsafe_allow_html=True)
        st.markdown(f'<div class="q-nota">Sector de mayor peso: {info_e["sector_mayor"]}'
                    f'{" (empate entre sectores: se toma el primero de la cartera)" if info_e["empate"] else ""}. '
                    '"Veces el VaR" se compara con el VaR normal.</div>', unsafe_allow_html=True)

    with st.container(key="tarjeta_oos"):
        st.markdown(ui.titulo_tarjeta("¿Cómo le fue a la cartera en el último año?"), unsafe_allow_html=True)
        fm = riesgo.fuera_de_muestra(ev_v, w_v, B, float(cartera["costo"]))
        st.markdown(ui.html_lectura(lecturas.lectura_fuera_muestra(fm, B, ev_v.index[0])), unsafe_allow_html=True)
        with st.expander("Ver la evolución del valor y las caídas"):
            fig_o = go.Figure()
            fig_o.add_trace(go.Scatter(x=fm.valor.index, y=fm.valor, mode="lines", name="Valor de la cartera",
                                       line=dict(color=ui.ORO_CLARO, width=2), fill="tozeroy",
                                       fillcolor="rgba(217,181,106,.08)"))
            fig_o.add_hline(y=B, line_color=ui.TENUE, line_dash="dot", annotation_text="Capital inicial",
                            annotation_font_color=ui.TENUE)
            fig_o.update_yaxes(title_text="USD", range=[fm.valor.min() * 0.97, fm.valor.max() * 1.02])
            st.plotly_chart(ui.estilo_fig(fig_o, 300), width="stretch", config=ui.PLOTLY_CONFIG)
            fig_dd = go.Figure(go.Scatter(x=fm.caida.index, y=fm.caida, mode="lines", fill="tozeroy",
                                          line=dict(color=ui.PALETA[8], width=1.5), fillcolor="rgba(208,122,122,.18)",
                                          name="Caída desde el máximo"))
            fig_dd.update_yaxes(tickformat=".0%", title_text="Caída desde el máximo")
            st.plotly_chart(ui.estilo_fig(fig_dd, 220), width="stretch", config=ui.PLOTLY_CONFIG)
            st.markdown(f'<div class="q-nota">Compra el {ev_v.index[0]:%Y-%m-%d} con costo de '
                        f'{float(cartera["costo"]):.2%} del monto y mantenimiento sin rebalanceo. Precios ajustados '
                        '(incluyen dividendos).</div>', unsafe_allow_html=True)

    return {
        "cartera_v": cartera_v,
        "exc": exc,
        "n_exc": n_exc,
        "r_hist": r_hist,
        "r_mc": r_mc,
        "r_norm": r_norm,
        "tabla_e": tabla_e,
        "w_v": w_v,
        "zona": zona,
    }
