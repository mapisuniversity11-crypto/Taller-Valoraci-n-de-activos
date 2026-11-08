"""Pestaña Asignación (Pasos 11 y 12): carteras, orden de compra, estrategias y presupuesto.
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import asignacion, config, lecturas, ui

from . import memoria


def mostrar(ctx: dict) -> dict:
    """Dibuja la pestaña. `ctx` trae lo que ya está calculado (datos, entradas, carteras y resultados de las
    pestañas anteriores); devuelve lo que usan las pestañas siguientes y la exportación."""
    opt = ctx["opt"]
    B = ctx["B"]
    s0_hoy = ctx["s0_hoy"]
    cov_v = ctx["cov_v"]
    alpha_v = ctx["alpha_v"]
    h_v = ctx["h_v"]
    sect = ctx["sect"]
    tickers = ctx["tickers"]
    limpio = ctx["limpio"]
    w_inv_corr = ctx["w_inv_corr"]
    estrategias = ctx["estrategias"]
    carteras = ctx["carteras"]
    ev_v = ctx["ev_v"]
    cartera = ctx["cartera"]
    pasos_corr = ctx["pasos_corr"]
    # Resultados que se comparten; quedan en None si esa parte no se calcula
    bt = compra = info_bt = pres = None

    if not opt.exito:
        st.error("No se pudo calcular la cartera de mínimo VaR: " + (" ".join(opt.problemas) or opt.mensaje))
    else:
        w_mv = opt.pesos
        vars_c = {n: asignacion.var_cartera(w, cov_v, B, alpha_v, h_v) for n, w in carteras.items()}
        compra = asignacion.acciones_enteras(w_mv, B, s0_hoy)
        w_real = compra["Peso real"]
        var_real = asignacion.var_cartera(w_real, cov_v, B, alpha_v, h_v)
        checks = asignacion.verificar(w_mv, sect, config.W_MIN, config.W_MAX, config.MAX_SECTOR)
        checks.append((opt.exito, f"El optimizador encontró solución ({opt.mensaje})."))

        st.markdown(ui.html_kpis([
            ("VaR · mínimo VaR", lecturas._usd(vars_c["Mínimo VaR"]), f"{alpha_v:.0%} · {h_v} días · VaR normal"),
            ("Menos riesgo que pesos iguales", f"{1 - vars_c['Mínimo VaR'] / vars_c['Pesos iguales']:.0%}",
             f"frente a {lecturas._usd(vars_c['Pesos iguales'])}"),
            ("Efectivo sobrante", lecturas._usd(B - compra["Monto invertido"].sum()), "al comprar acciones enteras"),
            ("Restricciones", f"{sum(ok for ok, _ in checks)} de {len(checks)} ✓", "suma, límites por acción y por sector"),
        ]), unsafe_allow_html=True)

        with st.container(key="tarjeta_pesos"):
            st.markdown(ui.titulo_tarjeta("¿Cuánto invertir en cada acción? · tres carteras"), unsafe_allow_html=True)
            orden_a = lecturas.orden_por_sector(tickers, sect)
            largo_w = pd.concat([
                pd.DataFrame({"Ticker": orden_a, "Peso": carteras[n][orden_a].to_numpy(), "Cartera": n})
                for n in ("Pesos iguales", "Inverso al VaR", "Mínimo VaR")
            ])
            fig_w = px.bar(largo_w, x="Ticker", y="Peso", color="Cartera", barmode="group",
                           color_discrete_sequence=[ui.TENUE, ui.PALETA[4], ui.ORO_CLARO])
            fig_w.add_hline(y=config.W_MAX, line_dash="dot", line_color=ui.ORO, annotation_text="Máximo 20 %",
                            annotation_font_color=ui.ORO)
            fig_w.add_hline(y=config.W_MIN, line_dash="dot", line_color="#3a3224", annotation_text="Mínimo 2 %",
                            annotation_font_color=ui.TENUE, annotation_position="bottom right")
            fig_w.update_yaxes(tickformat=".0%", title_text="Peso en la cartera")
            fig_w.update_xaxes(title_text="")
            fig_w.update_layout(legend=dict(orientation="h", y=1.1, x=0, title=""))
            st.plotly_chart(ui.estilo_fig(fig_w, 380), width="stretch", config=ui.PLOTLY_CONFIG)
            st.markdown(ui.html_lectura(lecturas.lectura_asignacion(w_mv, vars_c["Mínimo VaR"], vars_c["Pesos iguales"],
                                                                    vars_c["Inverso al VaR"], config.W_MAX)),
                        unsafe_allow_html=True)
            with st.expander("Ver la comparación detallada de las carteras"):
                comp = pd.DataFrame([{
                    "Cartera": n, "VaR": vars_c[n], "VaR (% del capital)": vars_c[n] / B,
                    "Peso máximo": w.max(), "Sector de mayor peso": asignacion.peso_por_sector(w, sect).idxmax(),
                    "Peso de ese sector": asignacion.peso_por_sector(w, sect).max(),
                } for n, w in carteras.items()])
                st.dataframe(comp.style.format({"VaR": "{:,.0f}", "VaR (% del capital)": "{:.1%}", "Peso máximo": "{:.1%}",
                                                "Peso de ese sector": "{:.1%}"}), hide_index=True, width="stretch")
                st.markdown(ui.html_lectura(lecturas.lectura_correccion(pasos_corr, config.MAX_SECTOR)),
                            unsafe_allow_html=True)
                st.markdown('<div class="q-nota">Solo se usa el riesgo: covarianza de la ventana de calibración con '
                            'media cero, sin rentabilidad esperada, Sharpe ni momentum. La cartera inversa al VaR '
                            'ignora las correlaciones, por eso no es una solución óptima.</div>', unsafe_allow_html=True)

        b1, b2 = st.columns([1.4, 1], gap="medium")
        with b1, st.container(key="tarjeta_compra"):
            st.markdown(ui.titulo_tarjeta("Orden de compra · cartera de mínimo VaR"), unsafe_allow_html=True)
            tabla_c = compra.assign(Sector=[sect[t] for t in compra.index]).reset_index(names="Ticker")
            st.dataframe(tabla_c[["Ticker", "Sector", "Peso objetivo", "Monto objetivo", "Precio", "Acciones",
                                  "Monto invertido", "Peso real"]].style.format({
                "Peso objetivo": "{:.1%}", "Monto objetivo": "{:,.0f}", "Precio": "{:.2f}", "Monto invertido": "{:,.0f}",
                "Peso real": "{:.1%}"}), hide_index=True, width="stretch")
            st.markdown(ui.html_lectura(lecturas.lectura_compra(compra, B, var_real)), unsafe_allow_html=True)
        with b2, st.container(key="tarjeta_checks_asig"):
            st.markdown(ui.titulo_tarjeta("Verificación de restricciones"), unsafe_allow_html=True)
            st.markdown(ui.html_checklist(checks), unsafe_allow_html=True)
            ps_mv = asignacion.peso_por_sector(w_mv, sect)
            fig_s = px.bar(x=ps_mv.values, y=ps_mv.index, orientation="h", color_discrete_sequence=[ui.ORO])
            fig_s.add_vline(x=config.MAX_SECTOR, line_dash="dot", line_color=ui.PALETA[8], annotation_text="Tope 35 %",
                            annotation_font_color=ui.PALETA[8])
            fig_s.update_xaxes(tickformat=".0%", title_text="Peso por sector", range=[0, 0.4])
            fig_s.update_yaxes(title_text="")
            st.plotly_chart(ui.estilo_fig(fig_s, 240), width="stretch", config=ui.PLOTLY_CONFIG)

        with st.container(key="tarjeta_estrategias_bt"):
            st.markdown(ui.titulo_tarjeta("Estrategias en el último año"), unsafe_allow_html=True)
            bt, info_bt = memoria.backtest(limpio.ajustados, ev_v.index[0],
                                           {"Pesos iguales": carteras["Pesos iguales"],
                                            "Inverso al VaR · corregido por sector": w_inv_corr,
                                            "Mínimo VaR · sin rebalanceo": w_mv},
                                           sect, B, float(cartera["costo"]), config.N_CALIBRACION,
                                           config.W_MIN, config.W_MAX, config.MAX_SECTOR)
            colores_bt = [ui.TENUE, ui.PALETA[4], ui.ORO, ui.ORO_CLARO]
            fig_bt = go.Figure()
            for color, nombre in zip(colores_bt, bt.columns):
                fig_bt.add_trace(go.Scatter(x=bt.index, y=bt[nombre], mode="lines", name=nombre,
                                            line=dict(color=color, width=2.5 if "mensual" in nombre else 1.6)))
            fig_bt.add_hline(y=B, line_dash="dot", line_color="#3a3224")
            fig_bt.update_yaxes(title_text="Valor de la cartera (USD)")
            fig_bt.update_layout(legend=dict(orientation="h", y=1.15, x=0))
            st.plotly_chart(ui.estilo_fig(fig_bt, 360), width="stretch", config=ui.PLOTLY_CONFIG)
            st.markdown(ui.html_lectura(lecturas.lectura_estrategias(info_bt)), unsafe_allow_html=True)
            with st.expander("Ver resultados de cada estrategia"):
                st.dataframe(pd.DataFrame([{"Estrategia": n, **{k: v for k, v in i.items()}} for n, i in info_bt.items()])
                             .rename(columns={"costos": "Costos", "rebalanceos": "Rebalanceos", "rentabilidad": "Rentabilidad",
                                              "caida_maxima": "Caída máxima", "valor_final": "Valor final"})
                             .style.format({"Costos": "{:,.0f}", "Rentabilidad": "{:.1%}", "Caída máxima": "{:.1%}",
                                            "Valor final": "{:,.0f}"}), hide_index=True, width="stretch")
                st.markdown(f'<div class="q-nota">Todas las estrategias compran el {ev_v.index[0]:%Y-%m-%d} y pagan '
                            f'{float(cartera["costo"]):.2%} sobre lo negociado. El rebalanceo mensual recalcula el mínimo '
                            f'VaR el primer día hábil de cada mes con los {config.N_CALIBRACION} retornos anteriores '
                            '(sin datos futuros).</div>', unsafe_allow_html=True)

        with st.container(key="tarjeta_presupuesto"):
            v_max = float(estrategias["presupuesto_v"])
            st.markdown(ui.titulo_tarjeta(f"Presupuesto de riesgo · VaR máximo {lecturas._usd(v_max)}"),
                        unsafe_allow_html=True)
            pres = asignacion.presupuesto_riesgo(w_mv, cov_v, B, alpha_v, h_v, v_max)
            st.markdown(ui.html_kpis([
                ("Fracción invertida f", f"{pres.f:.0%}", "f = mín{1, V / VaR con todo invertido}"),
                ("Invertido en acciones", lecturas._usd(pres.invertido.sum()), "cartera de mínimo VaR"),
                ("En efectivo", lecturas._usd(pres.efectivo), "sin riesgo"),
                ("VaR resultante", lecturas._usd(pres.var_ajustado), f"antes: {lecturas._usd(pres.var_total)}"),
            ]), unsafe_allow_html=True)
            st.markdown(ui.html_lectura(lecturas.lectura_presupuesto(pres, v_max)), unsafe_allow_html=True)
            st.markdown('<div class="q-nota">El presupuesto V se cambia en el panel de entradas (sección Estrategias). '
                        'La elección de la opción no cambia esta asignación, que se basa solo en el VaR.</div>',
                        unsafe_allow_html=True)

    return {"bt": bt, "compra": compra, "info_bt": info_bt, "pres": pres}
