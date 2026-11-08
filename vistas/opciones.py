"""Pestaña Opciones (Pasos 5 a 9): calculadora CRR, contratos reales, probabilidades y mercado.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config, crr, lecturas, opciones, ui


def mostrar(ctx: dict) -> dict:
    """Dibuja la pestaña. `ctx` trae lo que ya está calculado (datos, entradas, carteras y resultados de las
    pestañas anteriores); devuelve lo que usan las pestañas siguientes y la exportación."""
    opcion = ctx["opcion"]
    res = ctx["res"]
    par_hoy = ctx["par_hoy"]
    semilla_base = ctx["semilla_base"]
    limpio = ctx["limpio"]
    # Resultados que se comparten; quedan en None si esa parte no se calcula
    c_el = cands = contrato_mod = prima = probs = v_am = v_am_m = v_eu = v_eu_m = None

    # ---------- Parte 1: calculadora CRR (Paso 8) ----------
    o = opcion
    with st.container(key="tarjeta_crr"):
        st.markdown(ui.titulo_tarjeta("Calculadora · árbol binomial CRR"), unsafe_allow_html=True)
        st.markdown(f'<div class="q-nota">S0 = {o["s0"]:.2f} · K = {o["k"]:.2f} · T = {o["t"]:.2f} años · '
                    f'r = {o["r"]:.2%} · q = {o["q"]:.2%} · σ = {o["sigma"]:.2%} · N = {int(o["n"])} · {o["tipo"]} '
                    '(se cambian en el panel de entradas, sección Opción).</div>', unsafe_allow_html=True)
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
    contrato_mod = None      # contrato elegido, para valorarlo también con Merton en la pestaña Modelos
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
            if cands.empty:
                st.warning("No se encontraron candidatos en la cadena descargada.")
            else:
                n_venc = cands["Vencimiento"].nunique()
                n_ok = int(cands["Cotización válida"].sum())
                st.markdown(ui.html_lectura(
                    f"Se revisaron <b>{len(cands)} contratos reales</b> en {n_venc} vencimientos: para cada uno, los strikes "
                    f"más cercanos a los precios que el GBM proyecta como escenario bajo (P25), central (P50) y alto (P75). "
                    f"{n_ok} tienen cotización utilizable. {t_op} cotiza hoy a {lecturas._usd(s0_op)}."),
                    unsafe_allow_html=True)
            if not cands.empty:
              with st.expander("Ver la tabla de contratos candidatos"):
                st.markdown(f'<div class="q-nota">Consulta: {str(meta_cad.get("consulta", "—")).replace("T", " ")} · '
                            f'{meta_cad.get("fuente", "")} (las cotizaciones pueden ser diferidas). Precio de {t_op}: '
                            f'cierre sin ajustar. Strikes elegidos según los cuantiles P25, P50 y P75 del GBM (Ec. 4).</div>',
                            unsafe_allow_html=True)
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
                    st.markdown(ui.html_lectura(lecturas.lectura_sensibilidad(sens)), unsafe_allow_html=True)
                    with st.expander("Ver la tabla de sensibilidad"):
                        st.dataframe(sens.style.format({"Drift a": "{:.1%}", "Volatilidad σ": "{:.1%}",
                                                        "P(ejercicio)": "{:.0%}", "P(beneficio)": "{:.0%}"}),
                                     hide_index=True, width="stretch")

            # ---------- Parte 3: teórico vs. mercado (Pasos 7 y 9) ----------
            with st.container(key="tarjeta_mercado"):
                st.markdown(ui.titulo_tarjeta("Precio teórico frente al mercado"), unsafe_allow_html=True)
                n_val = max(int(o["n"]), 200)
                v_am_m = crr.precio_crr(s0_op, c_el["Strike"], c_el["T valoración"], r_op, q_op, sig_op, n_val,
                                        tipo_op, "americana")
                v_eu_m = crr.precio_crr(s0_op, c_el["Strike"], c_el["T valoración"], r_op, q_op, sig_op, n_val,
                                        tipo_op, "europea")
                cmp = opciones.comparar(v_am_m, c_el["Bid"], c_el["Ask"])
                contrato_mod = dict(k=float(c_el["Strike"]), t=float(c_el["T valoración"]), dias=int(c_el["Días"]),
                                    venc=str(c_el["Vencimiento"]), cmp=cmp, v_am=v_am_m, v_eu=v_eu_m,
                                    bid=float(c_el["Bid"]), ask=float(c_el["Ask"]))
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
                with st.expander("Ver la ficha del contrato y el método"):
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

    return {
        "c_el": c_el,
        "cands": cands,
        "contrato_mod": contrato_mod,
        "prima": prima,
        "probs": probs,
        "v_am": v_am,
        "v_am_m": v_am_m,
        "v_eu": v_eu,
        "v_eu_m": v_eu_m,
    }
