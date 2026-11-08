"""Pestaña Modelos: cada modelo relaja un supuesto del GBM y se aplica al caso.

Browniano estándar, Merton y fraccional están completos; Ornstein-Uhlenbeck / Vasicek, herding y
opciones reales quedan para Danna Caballero (ver docs/guia_modelos.md).
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import config, crr, gbm, lecturas, modelos, ui

from . import memoria


def mostrar(ctx: dict) -> dict:
    """Dibuja la pestaña. `ctx` trae lo que ya está calculado (datos, entradas, carteras y resultados de las
    pestañas anteriores); devuelve lo que usan las pestañas siguientes y la exportación."""
    semilla_base = ctx["semilla_base"]
    opcion = ctx["opcion"]
    contrato_mod = ctx["contrato_mod"]
    h_v = ctx["h_v"]
    tickers = ctx["tickers"]
    s0_hoy = ctx["s0_hoy"]
    precios_hoy = ctx["precios_hoy"]
    res = ctx["res"]
    limpio = ctx["limpio"]
    par_hoy = ctx["par_hoy"]
    # Resultados que se comparten; quedan en None si esa parte no se calcula
    mer = tab_h = v_mer = None

    st.markdown('<div class="q-nota" style="font-size:13px;margin-bottom:10px">El GBM descansa en supuestos: movimientos '
                'proporcionales al precio, sin saltos y sin memoria. Cada modelo de esta pestaña cambia uno de esos '
                'supuestos y lo prueba con los datos de nuestra cartera.</div>', unsafe_allow_html=True)
    sub = st.tabs(["Browniano estándar", "Merton · saltos", "Browniano fraccional",
                   "Ornstein-Uhlenbeck · Vasicek", "Herding · manada", "Opciones reales"])

    # ---------- Browniano estándar ----------
    with sub[0]:
        b1, b2, _ = st.columns([1, 1, 2])
        with b1:
            t_b = st.selectbox("Acción", tickers, key="bro_ticker",
                               index=tickers.index(config.TICKER_OPCION) if config.TICKER_OPCION in tickers else 0)
        with b2:
            hor_b = st.selectbox("Horizonte", list(gbm.HORIZONTES), index=3, key="bro_horizonte")
        ses_b = gbm.HORIZONTES[hor_b]
        p0_b = float(s0_hoy[t_b])
        cal_b = modelos.calibrar_browniano(precios_hoy[t_b], config.D)
        st.markdown(ui.html_intro(lecturas.INTRO_BROWNIANO), unsafe_allow_html=True)
        st.markdown(ui.html_variables(lecturas.variables_browniano(cal_b, p0_b)), unsafe_allow_html=True)

        sem_b = semilla_base + tickers.index(t_b)          # mismos choques que la pestaña GBM
        tr_b = modelos.simular_browniano(p0_b, cal_b.m, cal_b.s, ses_b, config.N_TRAYECTORIAS_GBM, sem_b)
        a_b, sig_b = float(par_hoy.loc[t_b, "a = μ − q"]), float(par_hoy.loc[t_b, "σ anual"])
        tr_g = gbm.simular_gbm(p0_b, a_b, sig_b, ses_b, config.N_TRAYECTORIAS_GBM, sem_b)
        pb, pg = np.percentile(tr_b, [10, 50, 90], axis=0), np.percentile(tr_g, [10, 50, 90], axis=0)
        p_neg = {1: modelos.prob_negativo(p0_b, cal_b.m, cal_b.s, 1), 10: modelos.prob_negativo(p0_b, cal_b.m, cal_b.s, 10)}

        with st.container(key="tarjeta_bro_abanico"):
            st.markdown(ui.titulo_tarjeta(f"Browniano estándar frente a GBM · {t_b} · {hor_b}"), unsafe_allow_html=True)
            st.markdown(ui.html_kpis([
                ("Rango · browniano estándar", f"{pb[0, -1]:,.0f} – {pb[2, -1]:,.0f}".replace(",", "."), "P10–P90 en USD"),
                ("Rango · GBM", f"{pg[0, -1]:,.0f} – {pg[2, -1]:,.0f}".replace(",", "."), "P10–P90 en USD"),
                ("Precio negativo a 10 años", lecturas._pct(p_neg[10], 3), "browniano estándar · en el GBM es 0 %"),
            ]), unsafe_allow_html=True)
            fechas_b = pd.bdate_range(limpio.sin_ajustar.index[-1], periods=ses_b + 1)
            fig_bb = go.Figure()
            for p_, nom, color, relleno in ((pg, "GBM", ui.ORO_CLARO, "rgba(217,181,106,.16)"),
                                            (pb, "Browniano estándar", ui.PALETA[3], "rgba(138,160,184,.16)")):
                fig_bb.add_trace(go.Scatter(x=fechas_b, y=p_[2], mode="lines", line=dict(width=0), showlegend=False,
                                            hoverinfo="skip"))
                fig_bb.add_trace(go.Scatter(x=fechas_b, y=p_[0], mode="lines", line=dict(width=0), fill="tonexty",
                                            fillcolor=relleno, name=f"Rango P10–P90 · {nom}", hoverinfo="skip"))
                fig_bb.add_trace(go.Scatter(x=fechas_b, y=p_[1], mode="lines", name=f"Mediana · {nom}",
                                            line=dict(color=color, width=2)))
            fig_bb.update_yaxes(title_text="Precio (USD)")
            fig_bb.update_layout(legend=dict(orientation="h", y=1.15, x=0), hovermode="x unified")
            st.plotly_chart(ui.estilo_fig(fig_bb, 380), width="stretch", config=ui.PLOTLY_CONFIG)

        niv = modelos.volatilidad_por_nivel(limpio.ajustados[t_b])
        c_usd = float(niv["Precio medio"].corr(niv["Movimiento diario USD"]))
        c_pct = float(niv["Precio medio"].corr(niv["Movimiento diario %"]))
        with st.container(key="tarjeta_bro_nivel"):
            st.markdown(ui.titulo_tarjeta("¿Los movimientos son en dólares fijos o en porcentaje?"), unsafe_allow_html=True)
            fig_nv = px.scatter(niv, x="Precio medio", y="Movimiento diario USD", trendline=None,
                                color_discrete_sequence=[ui.ORO_CLARO],
                                hover_data={"Desde": "|%Y-%m-%d", "Movimiento diario %": ":.2%"})
            fig_nv.update_traces(marker=dict(size=12, line=dict(width=1, color="#070706")))
            if len(niv) > 2:
                coef = np.polyfit(niv["Precio medio"], niv["Movimiento diario USD"], 1)
                xs_nv = np.linspace(niv["Precio medio"].min(), niv["Precio medio"].max(), 50)
                fig_nv.add_scatter(x=xs_nv, y=np.polyval(coef, xs_nv), mode="lines", name="Tendencia",
                                   line=dict(color=ui.TENUE, dash="dot"))
            fig_nv.update_xaxes(title_text=f"Precio medio del trimestre (USD)")
            fig_nv.update_yaxes(title_text="Movimiento diario típico (USD)")
            fig_nv.update_layout(showlegend=False)
            st.plotly_chart(ui.estilo_fig(fig_nv, 320), width="stretch", config=ui.PLOTLY_CONFIG)
            st.markdown('<div class="q-nota">Cada punto es un trimestre de la historia descargada: su precio medio y cuánto '
                        'se movió la acción en un día típico (desviación del cambio diario en dólares).</div>',
                        unsafe_allow_html=True)
            st.markdown(ui.html_lectura(lecturas.lectura_browniano(t_b, p_neg, c_usd, c_pct, hor_b)),
                        unsafe_allow_html=True)
        with st.expander("Ver ecuaciones y calibración"):
            st.markdown(f"""
- **Browniano estándar:** P_t+Δt = P_t + m·Δt + s·√Δt·Z, con Z ~ N(0, 1). El precio puede volverse negativo.
- **Calibración:** m = 252 · media(ΔP), s = √252 · desv(ΔP) con divisor n − 1, donde ΔP es el cambio diario del precio
  ajustado en la misma ventana de 504 retornos que usa la pestaña GBM.
- **Probabilidad de precio negativo:** P(P_T < 0) = Φ((−P₀ − m·T) / (s·√T)).
- **Comparación justa:** las {config.N_TRAYECTORIAS_GBM:,} trayectorias de ambos modelos usan los mismos choques Z
  (misma semilla que la pestaña GBM).
""".replace(",", "."))

    # ---------- Merton: saltos ----------
    with sub[1]:
        t_m = res.get("ticker_opcion", config.TICKER_OPCION)
        m1_, _ = st.columns([1, 3])
        with m1_:
            k_salto = st.select_slider("Qué se considera un salto", options=[3.0, 4.0, 5.0], value=3.0, key="mer_umbral",
                                       format_func=lambda v: f"más de {v:.0f} veces lo normal")
        x_m = np.log(precios_hoy[t_m] / precios_hoy[t_m].shift(1)).dropna()
        mer = modelos.calibrar_merton(x_m, k_salto, config.D)
        st.markdown(ui.html_intro(lecturas.INTRO_MERTON), unsafe_allow_html=True)
        st.markdown(ui.html_variables(lecturas.variables_merton(mer, k_salto)), unsafe_allow_html=True)

        with st.container(key="tarjeta_mer_saltos"):
            st.markdown(ui.titulo_tarjeta(f"Días de salto de {t_m} · últimos {len(x_m)} días"), unsafe_allow_html=True)
            fig_ms = go.Figure()
            serie_m = precios_hoy[t_m]
            fig_ms.add_trace(go.Scatter(x=serie_m.index, y=serie_m, mode="lines", name="Precio ajustado",
                                        line=dict(color=ui.TEXTO, width=1.4)))
            for signo, nom, color in ((1, "Salto hacia arriba", ui.PALETA[2]), (-1, "Salto hacia abajo", ui.PALETA[1])):
                f_s = mer.tamanos[np.sign(mer.tamanos) == signo].index
                fig_ms.add_trace(go.Scatter(x=f_s, y=serie_m.loc[f_s], mode="markers", name=nom,
                                            marker=dict(size=11, color=color, line=dict(color="#070706", width=1)),
                                            customdata=mer.tamanos.loc[f_s].values,
                                            hovertemplate="%{x|%d/%m/%Y}: %{customdata:.1%}<extra></extra>"))
            fig_ms.update_yaxes(title_text="Precio ajustado (USD)")
            fig_ms.update_layout(legend=dict(orientation="h", y=1.12, x=0))
            st.plotly_chart(ui.estilo_fig(fig_ms, 340), width="stretch", config=ui.PLOTLY_CONFIG)
            st.markdown(ui.html_lectura(lecturas.lectura_saltos(t_m, mer, len(x_m))), unsafe_allow_html=True)

        s0_m = float(s0_hoy[t_m])
        q_m, r_m = float(res["q"][t_m]), float(res["r"])
        sig_m = float(par_hoy.loc[t_m, "σ anual"])
        tipo_m = opcion["tipo"]
        if contrato_mod is not None:
            k_m, t_mm, dias_m, cmp_m = contrato_mod["k"], contrato_mod["t"], contrato_mod["dias"], contrato_mod["cmp"]
            nombre_c = f"el contrato elegido en Opciones ({tipo_m}, strike {k_m:.2f}, vence {contrato_mod['venc']})"
        else:
            k_m, dias_m, cmp_m = round(s0_m), 91, None
            t_mm = dias_m / 365
            nombre_c = f"una {tipo_m} at-the-money a 3 meses (no hay cadena de opciones descargada)"
        v_bs_m = crr.black_scholes(s0_m, k_m, t_mm, r_m, q_m, sig_m, tipo_m)
        v_mer = modelos.precio_merton(s0_m, k_m, t_mm, r_m, q_m, mer.sigma, mer.lam, mer.mu_j, mer.delta_j, tipo_m)
        p_salto = modelos.prob_algun_salto(mer.lam, t_mm)

        with st.container(key="tarjeta_mer_opcion"):
            st.markdown(ui.titulo_tarjeta("¿Cuánto vale la opción si hay saltos?"), unsafe_allow_html=True)
            st.markdown(f'<div class="q-nota">Se valora {nombre_c}. S0 = {s0_m:.2f}, r = {r_m:.2%}, q = {q_m:.2%}.</div>',
                        unsafe_allow_html=True)
            valida_m = bool(cmp_m and cmp_m.get("valida"))
            st.markdown(ui.html_kpis([
                ("Sin saltos (GBM)", lecturas._usd(v_bs_m), f"europea · σ = {sig_m:.1%}".replace(".", ",")),
                ("Con saltos (Merton)", lecturas._usd(v_mer), f"europea · {mer.lam:.1f} saltos/año".replace(".", ",")),
                ("Precio medio del mercado", lecturas._usd(cmp_m["medio"]) if valida_m else "—",
                 "entre bid y ask" if valida_m else "sin cotización utilizable"),
                ("Probabilidad de un salto", f"{p_salto:.0%}", f"antes del vencimiento · {dias_m} días"),
            ]), unsafe_allow_html=True)
            g1_, g2_ = st.columns([1.4, 1], gap="medium")
            with g1_:
                fig_mo = go.Figure()
                if valida_m:
                    bid_m, ask_m = contrato_mod["bid"], contrato_mod["ask"]
                    fig_mo.add_trace(go.Scatter(x=[bid_m, ask_m], y=["Mercado"] * 2, mode="lines",
                                                line=dict(color=ui.ORO, width=18), name="Rango bid – ask",
                                                hovertemplate="%{x:.2f}<extra></extra>"))
                fig_mo.add_trace(go.Scatter(x=[v_bs_m], y=["Modelos"], mode="markers", name="Sin saltos (GBM)",
                                            marker=dict(symbol="circle-open", size=14, color=ui.TEXTO, line=dict(width=2))))
                fig_mo.add_trace(go.Scatter(x=[v_mer], y=["Modelos"], mode="markers", name="Con saltos (Merton)",
                                            marker=dict(symbol="diamond", size=16, color=ui.ORO_CLARO)))
                fig_mo.update_xaxes(title_text="Prima por acción (USD)")
                fig_mo.update_layout(legend=dict(orientation="h", y=1.3, x=0))
                st.plotly_chart(ui.estilo_fig(fig_mo, 240), width="stretch", config=ui.PLOTLY_CONFIG)
            with g2_:
                st.markdown(ui.html_lectura(lecturas.lectura_merton(v_bs_m, v_mer, cmp_m, p_salto, tipo_m, dias_m)),
                            unsafe_allow_html=True)
        with st.expander("Ver ecuaciones, saltos detectados y supuestos"):
            st.markdown(f"""
- **Modelo:** el log-precio sigue un GBM con volatilidad σ más saltos que llegan como un proceso de Poisson con
  intensidad λ por año; cada salto es normal en logaritmo, con media μ_J y desviación δ_J.
- **Detección de saltos:** días en que |x − mediana| supera {k_salto:.0f} veces la desviación robusta
  (1,4826 · desviación absoluta mediana, que no se infla con los propios saltos). Umbral: {mer.umbral:.2%}.
- **Calibración:** σ con los días normales; λ = saltos / años de la muestra; μ_J y δ_J con los días de salto, descontando
  el retorno medio de un día normal. Ventana: los mismos {len(x_m)} retornos que usa la pestaña GBM.
- **Precio (Merton, 1976):** promedio de precios Black-Scholes ponderado por la probabilidad de n saltos antes del
  vencimiento: σ_n² = σ² + n·δ_J²/T y r_n = r − λκ + n·ln(1 + κ)/T, con κ = e^(μ_J + δ_J²/2) − 1.
- **Supuesto:** se valora la versión europea. El contrato real es americano; en la pestaña Opciones, el ejercicio
  anticipado vale {lecturas._usd(contrato_mod["v_am"] - contrato_mod["v_eu"]) if contrato_mod else "—"} según el árbol CRR.
""")
            if len(mer.tamanos):
                st.dataframe(pd.DataFrame({"Fecha": mer.tamanos.index.strftime("%Y-%m-%d"),
                                           "Movimiento del día": mer.tamanos.values})
                             .style.format({"Movimiento del día": "{:.2%}"}), hide_index=True, width="stretch")

    # ---------- Browniano fraccional ----------
    with sub[2]:
        x_all = np.log(limpio.ajustados / limpio.ajustados.shift(1)).dropna()
        tab_h = memoria.tabla_hurst(x_all, semilla_base)
        f1_, _ = st.columns([1, 3])
        with f1_:
            t_f = st.selectbox("Acción", tickers, key="frac_ticker")
        h_f = float(tab_h.set_index("Ticker").loc[t_f, "H"])
        st.markdown(ui.html_intro(lecturas.INTRO_FRACCIONAL), unsafe_allow_html=True)
        st.markdown(ui.html_variables(lecturas.variables_fraccional(h_f, t_f)), unsafe_allow_html=True)

        with st.container(key="tarjeta_frac_hurst"):
            st.markdown(ui.titulo_tarjeta(f"Memoria de las 10 acciones · {len(x_all)} retornos"), unsafe_allow_html=True)
            colores_h = {"Sin memoria": ui.ORO_CLARO, "Tendencia": ui.PALETA[2], "Reversión": ui.PALETA[1]}
            fig_h = go.Figure()
            fig_h.add_trace(go.Bar(x=tab_h["Ticker"], y=tab_h["Sin memoria hasta"] - tab_h["Sin memoria desde"],
                                   base=tab_h["Sin memoria desde"], marker_color="rgba(232,226,212,.10)",
                                   name="Zona sin memoria", hoverinfo="skip", width=0.7))
            for lect, color in colores_h.items():
                sel = tab_h[tab_h["Lectura"] == lect]
                if len(sel):
                    fig_h.add_trace(go.Scatter(x=sel["Ticker"], y=sel["H"], mode="markers", name=lect,
                                               marker=dict(size=14, color=color, line=dict(color="#070706", width=1)),
                                               hovertemplate="%{x}: H = %{y:.3f}<extra></extra>"))
            fig_h.add_hline(y=0.5, line_color=ui.TENUE, line_dash="dot", annotation_text="H = 0,5 (supuesto del GBM)",
                            annotation_font_color=ui.TENUE)
            fig_h.update_yaxes(title_text="Exponente de Hurst H")
            fig_h.update_layout(legend=dict(orientation="h", y=1.12, x=0), barmode="overlay")
            st.plotly_chart(ui.estilo_fig(fig_h, 340), width="stretch", config=ui.PLOTLY_CONFIG)
            st.markdown('<div class="q-nota">La franja gris es el rango de H que se obtiene al desordenar los días al azar, '
                        'es decir, lo que daría una acción sin memoria con la misma cantidad de datos. Si el punto cae '
                        'dentro, no hay evidencia de memoria.</div>', unsafe_allow_html=True)
            st.markdown(ui.html_lectura(lecturas.lectura_hurst(tab_h, h_v)), unsafe_allow_html=True)

        v_f = modelos.volatilidad_por_horizonte(x_all[t_f])
        fac_f = modelos.factor_var_hurst(h_v, h_f)
        with st.container(key="tarjeta_frac_escala"):
            st.markdown(ui.titulo_tarjeta(f"¿Crece el riesgo de {t_f} con la raíz del tiempo?"), unsafe_allow_html=True)
            e1_, e2_ = st.columns([1.3, 1], gap="medium")
            with e1_:
                fig_e = go.Figure()
                fig_e.add_trace(go.Scatter(x=v_f["Sesiones"], y=v_f["Real"], mode="lines+markers", name="Real",
                                           line=dict(color=ui.ORO_CLARO, width=2.5)))
                fig_e.add_trace(go.Scatter(x=v_f["Sesiones"], y=v_f["Regla √m (GBM)"], mode="lines+markers",
                                           name="Regla de la raíz del tiempo (GBM)",
                                           line=dict(color=ui.TENUE, width=2, dash="dot")))
                fig_e.update_xaxes(title_text="Horizonte (sesiones)")
                fig_e.update_yaxes(title_text="Volatilidad del retorno acumulado", tickformat=".0%")
                fig_e.update_layout(legend=dict(orientation="h", y=1.12, x=0), hovermode="x unified")
                st.plotly_chart(ui.estilo_fig(fig_e, 320), width="stretch", config=ui.PLOTLY_CONFIG)
            with e2_:
                st.markdown(ui.html_kpis([
                    (f"H de {t_f}", f"{h_f:.2f}".replace(".", ","), str(tab_h.set_index("Ticker").loc[t_f, "Lectura"])),
                    (f"VaR a {h_v} días con H", f"{fac_f - 1:+.1%}".replace(".", ","), "frente a la regla √h"),
                ]), unsafe_allow_html=True)
                st.markdown(ui.html_lectura(lecturas.lectura_escalamiento(t_f, h_f, fac_f, h_v, v_f,
                                                                           str(tab_h.set_index("Ticker").loc[t_f, "Lectura"]))),
                            unsafe_allow_html=True)

        with st.container(key="tarjeta_frac_caminos"):
            st.markdown(ui.titulo_tarjeta(f"Cómo se vería {t_f} con distinta memoria · 1 año"), unsafe_allow_html=True)
            sig_f = float(par_hoy.loc[t_f, "σ anual"])
            fechas_f = pd.bdate_range(limpio.sin_ajustar.index[-1], periods=253)
            fig_c = go.Figure()
            for h_ej, nom, color in ((0.3, "H = 0,3 · reversión", ui.PALETA[1]), (0.5, "H = 0,5 · sin memoria (GBM)", ui.TEXTO),
                                     (0.7, "H = 0,7 · tendencia", ui.PALETA[2])):
                b_h = modelos.simular_fbm(h_ej, 252, 1, semilla_base)[0]
                fig_c.add_trace(go.Scatter(x=fechas_f, y=float(s0_hoy[t_f]) * np.exp(sig_f * b_h), mode="lines", name=nom,
                                           line=dict(color=color, width=1.6)))
            fig_c.update_yaxes(title_text="Precio (USD)")
            fig_c.update_layout(legend=dict(orientation="h", y=1.12, x=0), hovermode="x unified")
            st.plotly_chart(ui.estilo_fig(fig_c, 320), width="stretch", config=ui.PLOTLY_CONFIG)
            st.markdown(ui.html_lectura(
                "Los tres caminos usan los mismos choques aleatorios y la misma volatilidad anual; solo cambia la memoria. "
                "Con <b>H = 0,3</b> el precio zigzaguea y se devuelve; con <b>H = 0,7</b> forma rachas largas de subida o de "
                f"bajada. El H estimado de {t_f} ({lecturas._num(h_f)}) indica a cuál de los tres se parece más."),
                unsafe_allow_html=True)

        with st.expander("Ver la tabla de H y el método"):
            st.dataframe(tab_h.style.format({"H": "{:.3f}", "Sin memoria desde": "{:.3f}", "Sin memoria hasta": "{:.3f}"}),
                         hide_index=True, width="stretch")
            st.markdown(f"""
- **Estimación de H (varianza agregada):** se suman los log-retornos en ventanas móviles de m = 1, 2, 4, 8, 16 y 32
  sesiones y se mide la desviación de esas sumas. Si crece como m^H, H es la pendiente de log(desviación) contra log(m).
- **Zona sin memoria:** percentiles 5 y 95 de H en 200 versiones de la misma serie con los días desordenados al azar
  (semilla {semilla_base}). Incluye el sesgo que tiene el estimador con esta cantidad de datos.
- **Datos:** toda la historia limpia ({len(x_all)} retornos), porque estimar memoria requiere muestras largas.
- **Caminos:** browniano fraccional por Cholesky, con incrementos de covarianza ½(|k+1|^2H − 2|k|^2H + |k−1|^2H) y
  varianza 1 a un año, multiplicados por la σ anual de la acción.
- **VaR:** el VaR a h días con memoria se escala con h^H en lugar de √h; el cambio relativo es h^H / √h − 1.
""")

    # ---------- Modelos a cargo de la segunda integrante ----------
    pendientes = [
        ("Ornstein-Uhlenbeck · Vasicek", "La tasa libre de riesgo (IRX) y la diferencia de precio entre XOM y CVX",
         "¿Qué tanto puede moverse la tasa durante la vida de la opción y cuánto cambia la prima? Si dos acciones del "
         "mismo sector se separan, ¿vuelven a juntarse?"),
        ("Herding · comportamiento en manada", "Las 10 acciones y la cartera",
         "¿Qué le pasa a nuestro riesgo cuando todos los inversionistas actúan igual? Correlación promedio en el tiempo "
         "y simulación de inversionistas que se copian entre sí, con su efecto en el VaR."),
        ("Opciones reales", "Un proyecto de ejemplo con la volatilidad de XOM y CVX, r del Tesoro y el árbol CRR",
         "¿Cuánto vale la flexibilidad de esperar, expandir o abandonar un proyecto?"),
    ]
    for i_p, (nom_p, datos_p, preg_p) in enumerate(pendientes, start=3):
        with sub[i_p]:
            with st.container(key=f"tarjeta_pendiente_{i_p}"):
                st.markdown(ui.titulo_tarjeta(f"{nom_p} · en construcción"), unsafe_allow_html=True)
                st.markdown(ui.html_intro(preg_p), unsafe_allow_html=True)
                st.markdown(f'<div class="q-nota">Datos: {datos_p}. Sigue la misma estructura de las otras subpestañas: qué '
                            'busca el modelo, sus variables y qué pasa si suben o bajan, la aplicación a nuestro caso con '
                            'un gráfico y su lectura, y las ecuaciones en un desplegable. Guía: docs/guia_modelos.md.</div>',
                            unsafe_allow_html=True)

    return {"mer": mer, "tab_h": tab_h, "v_mer": v_mer}
