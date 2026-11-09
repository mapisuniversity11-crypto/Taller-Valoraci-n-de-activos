"""Pestaña Modelos: cada modelo relaja un supuesto del GBM y se aplica al caso.

Seis subpestañas: browniano estándar, Merton y browniano fraccional (María Paula Muñoz Malaver);
Ornstein-Uhlenbeck · Vasicek, herding y opciones reales (Danna Caballero). Guía: docs/guia_modelos.md.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm

from src import config, crr, datos, gbm, lecturas, modelos, riesgo, ui

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
    sect = ctx["sect"]
    cov_v = ctx["cov_v"]
    w_v = ctx["w_v"]
    cartera_v = ctx["cartera_v"]
    B = ctx["B"]
    alpha_v = ctx["alpha_v"]
    # Resultados que se comparten; quedan en None si esa parte no se calcula
    mer = tab_h = v_mer = ou_tabla = var_manada = rho_movil = or_tabla = None
    x_all = np.log(limpio.ajustados / limpio.ajustados.shift(1)).dropna()   # toda la historia limpia

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

    # Opción que se usa como ejemplo en Vasicek (la misma de Merton: el contrato elegido o una ATM a 3 meses)
    t_op = res.get("ticker_opcion", config.TICKER_OPCION)
    s0_op, q_op, r_op = float(s0_hoy[t_op]), float(res["q"][t_op]), float(res["r"])
    sig_op, tipo_op = float(par_hoy.loc[t_op, "σ anual"]), opcion["tipo"]
    if contrato_mod is not None:
        k_op, t_vida, dias_op = contrato_mod["k"], contrato_mod["t"], contrato_mod["dias"]
        nombre_op = f"el contrato elegido en Opciones ({t_op} {tipo_op}, strike {k_op:.2f}, vence {contrato_mod['venc']})"
    else:
        k_op, dias_op = float(round(s0_op)), 91
        t_vida = dias_op / 365
        nombre_op = f"una {tipo_op} at-the-money de {t_op} a 3 meses (no hay cadena de opciones descargada)"

    # ---------- Ornstein-Uhlenbeck · Vasicek ----------
    with sub[3]:
        st.markdown(ui.html_intro(lecturas.INTRO_OU), unsafe_allow_html=True)
        filas_ou = []
        irx = res.get("irx")
        if irx is None or len(irx) < 60:
            with st.container(key="tarjeta_vas_sin_datos"):
                st.markdown(ui.titulo_tarjeta("Tasa del Tesoro · sin histórico"), unsafe_allow_html=True)
                st.markdown(ui.html_lectura(
                    "No hay histórico de la tasa del Tesoro (^IRX) en esta ejecución. Ejecute el análisis con la fuente "
                    "Yahoo Finance para descargarlo; con «Guardar descarga como respaldo» marcado queda guardado en "
                    "data/raw/tasa_irx.csv para las siguientes ejecuciones. La parte de acciones del mismo sector, abajo, "
                    "funciona con cualquier fuente."), unsafe_allow_html=True)
        else:
            r_ser = datos.r_continua_desde_irx(irx)
            ou_r = modelos.calibrar_ou(r_ser, config.D)
            st.markdown(ui.html_variables(lecturas.variables_vasicek(ou_r, "la tasa del Tesoro",
                                                                     lambda v: lecturas._pct(v, 2))),
                        unsafe_allow_html=True)
            ses_r = max(1, int(round(t_vida * config.D)))
            meses_r = max(1, int(round(t_vida * 12)))
            tr_r = modelos.simular_ou(r_op, ou_r.kappa, ou_r.theta, ou_r.sigma, ses_r, config.N_TRAYECTORIAS_GBM,
                                      semilla_base)
            pr = np.percentile(tr_r, [10, 50, 90], axis=0)
            with st.container(key="tarjeta_vas_tasa"):
                st.markdown(ui.titulo_tarjeta(f"¿Cuánto puede moverse la tasa en los próximos {dias_op} días?"),
                            unsafe_allow_html=True)
                st.markdown(ui.html_kpis([
                    ("Tasa hoy", lecturas._pct(r_op, 2),
                     "continua · ^IRX de Yahoo" if "IRX" in str(res["fuente_r"]) else "continua · valor manual"),
                    ("Nivel de largo plazo θ", lecturas._pct(ou_r.theta, 2), "al que tiende a volver"),
                    ("Rango al vencimiento", f"{lecturas._pct(pr[0, -1], 2)} – {lecturas._pct(pr[2, -1], 2)}",
                     "80 % de los escenarios (P10–P90)"),
                    ("Vida media", lecturas._anios_o_dias(ou_r.vida_media),
                     {"fuerte": "reversión significativa al 5 %", "débil": "reversión solo al 10 %",
                      "ninguna": "sin reversión significativa"}[ou_r.evidencia]),
                ]), unsafe_allow_html=True)
                hist_r = r_ser.iloc[-2 * config.D:]
                fechas_r = pd.bdate_range(hist_r.index[-1], periods=ses_r + 1)
                fig_r = go.Figure()
                fig_r.add_trace(go.Scatter(x=hist_r.index, y=hist_r, mode="lines", name="Tasa observada (^IRX)",
                                           line=dict(color=ui.TEXTO, width=1.6),
                                           hovertemplate="%{x|%d/%m/%Y}: %{y:.2%}<extra></extra>"))
                fig_r.add_trace(go.Scatter(x=fechas_r, y=pr[2], mode="lines", line=dict(width=0), showlegend=False,
                                           hoverinfo="skip"))
                fig_r.add_trace(go.Scatter(x=fechas_r, y=pr[0], mode="lines", line=dict(width=0), fill="tonexty",
                                           fillcolor="rgba(217,181,106,.22)", name="Rango probable (P10–P90)",
                                           hovertemplate="P10: %{y:.2%}<extra></extra>"))
                fig_r.add_trace(go.Scatter(x=fechas_r, y=pr[1], mode="lines", name="Mediana · Vasicek",
                                           line=dict(color=ui.ORO_CLARO, width=2.5),
                                           hovertemplate="Mediana: %{y:.2%}<extra></extra>"))
                fig_r.add_hline(y=ou_r.theta, line_color=ui.TENUE, line_dash="dot", line_width=1,
                                annotation_text="Nivel de largo plazo θ", annotation_font_color=ui.TENUE)
                fig_r.update_yaxes(title_text="Tasa continua anual", tickformat=".1%")
                fig_r.update_layout(legend=dict(orientation="h", y=1.12, x=0), hovermode="x unified")
                st.plotly_chart(ui.estilo_fig(fig_r, 360), width="stretch", config=ui.PLOTLY_CONFIG)
                st.markdown(ui.html_lectura(lecturas.lectura_tasa(ou_r, r_op, (pr[0, -1], pr[2, -1]), meses_r)),
                            unsafe_allow_html=True)

            # Lo que descuenta la opción es la tasa PROMEDIO durante su vida, no la de un día
            m_rb, s_rb = modelos.tasa_promedio_vasicek(r_op, ou_r.kappa, ou_r.theta, ou_r.sigma, t_vida)
            r_bajo, r_alto = m_rb + norm.ppf(0.10) * s_rb, m_rb + norm.ppf(0.90) * s_rb
            n_vas = min(int(opcion["n"]), 300)

            def _prima(r_):
                return crr.precio_crr(s0_op, k_op, t_vida, r_, q_op, sig_op, n_vas, tipo_op, "americana")

            primas_r = {"baja": _prima(r_bajo), "hoy": _prima(r_op), "alta": _prima(r_alto)}
            with st.container(key="tarjeta_vas_prima"):
                st.markdown(ui.titulo_tarjeta("¿Cuánto cambia la prima si cambia la tasa?"), unsafe_allow_html=True)
                st.markdown(f'<div class="q-nota">Se valora {nombre_op} con el árbol CRR americano (N = {n_vas}). S0 = '
                            f'{s0_op:.2f}, q = {q_op:.2%}, σ = {sig_op:.2%}. Solo cambia la tasa: la tasa promedio entre '
                            'hoy y el vencimiento según Vasicek.</div>', unsafe_allow_html=True)
                v1_, v2_ = st.columns([1.4, 1], gap="medium")
                with v1_:
                    grid_r = np.linspace(min(r_bajo, r_op) - s_rb, max(r_alto, r_op) + s_rb, 21)
                    grid_r = grid_r[grid_r > -0.05]
                    fig_pr = go.Figure()
                    fig_pr.add_vrect(x0=r_bajo, x1=r_alto, fillcolor="rgba(217,181,106,.12)", line_width=0,
                                     annotation_text="80 % de los escenarios", annotation_font_color=ui.TENUE,
                                     annotation_position="top left")
                    fig_pr.add_trace(go.Scatter(x=grid_r, y=[_prima(x) for x in grid_r], mode="lines",
                                                name="Prima CRR", line=dict(color=ui.ORO_CLARO, width=2.5),
                                                hovertemplate="r = %{x:.2%}: USD %{y:.3f}<extra></extra>"))
                    fig_pr.add_trace(go.Scatter(x=[r_bajo, r_op, r_alto], y=[primas_r["baja"], primas_r["hoy"],
                                                                             primas_r["alta"]],
                                                mode="markers+text", text=["P10", "Hoy", "P90"], textposition="top center",
                                                textfont=dict(color=ui.TENUE), name="Escenarios",
                                                marker=dict(size=11, color=ui.ORO, line=dict(color="#070706", width=1)),
                                                hovertemplate="r = %{x:.2%}: USD %{y:.3f}<extra></extra>"))
                    fig_pr.update_xaxes(title_text="Tasa promedio hasta el vencimiento", tickformat=".1%")
                    fig_pr.update_yaxes(title_text="Prima por acción (USD)")
                    fig_pr.update_layout(showlegend=False)
                    st.plotly_chart(ui.estilo_fig(fig_pr, 300), width="stretch", config=ui.PLOTLY_CONFIG)
                with v2_:
                    st.markdown(ui.html_kpis([
                        ("Prima · tasa baja (P10)", lecturas._usd(primas_r["baja"]), f"r̄ = {lecturas._pct(r_bajo, 2)}"),
                        ("Prima · tasa alta (P90)", lecturas._usd(primas_r["alta"]), f"r̄ = {lecturas._pct(r_alto, 2)}"),
                    ]), unsafe_allow_html=True)
                st.markdown(ui.html_lectura(lecturas.lectura_prima_tasa(primas_r, primas_r["hoy"], tipo_op, dias_op,
                                                                        r_bajo, r_alto)), unsafe_allow_html=True)
            filas_ou.append({"Serie": "Tasa del Tesoro (^IRX, continua)", "Desde": r_ser.index[0], "Hasta": r_ser.index[-1],
                             "Observaciones": ou_r.n + 1, "κ": ou_r.kappa, "θ": ou_r.theta, "σ": ou_r.sigma, "b AR(1)": ou_r.b,
                             "t Dickey-Fuller": ou_r.t_df, "Evidencia de reversión": ou_r.evidencia,
                             "Vida media (sesiones)": ou_r.vida_media * config.D, "Valor hoy": r_op,
                             "Tasa promedio P10": r_bajo, "Tasa promedio P90": r_alto,
                             "Prima tasa P10": primas_r["baja"], "Prima hoy": primas_r["hoy"],
                             "Prima tasa P90": primas_r["alta"]})

        # --- Dos acciones del mismo sector ---
        pares = [(a, b) for i, a in enumerate(tickers) for b in tickers[i + 1:] if sect[a] == sect[b]]
        pares = pares or [(tickers[0], tickers[1])]
        nombres_p = [f"{a} – {b}" for a, b in pares]
        with st.container(key="tarjeta_vas_par"):
            st.markdown(ui.titulo_tarjeta("Si dos acciones del mismo sector se separan, ¿vuelven a juntarse?"),
                        unsafe_allow_html=True)
            p1_, _ = st.columns([1, 3])
            with p1_:
                par_sel = st.selectbox("Par de acciones", nombres_p, key="ou_par",
                                       index=nombres_p.index("XOM – CVX") if "XOM – CVX" in nombres_p else 0)
            a_p, b_p = pares[nombres_p.index(par_sel)]
            pa, pb_ = limpio.ajustados[a_p], limpio.ajustados[b_p]
            spread = (np.log(pa / pa.iloc[0]) - np.log(pb_ / pb_.iloc[0])).rename("Diferencia")
            ou_s = modelos.calibrar_ou(spread, config.D)
            st.markdown(ui.html_variables(lecturas.variables_vasicek(ou_s, f"la diferencia {a_p} − {b_p}",
                                                                     lambda v: lecturas._pct(v, 1))),
                        unsafe_allow_html=True)
            ses_s = 126
            fechas_s = pd.bdate_range(spread.index[-1], periods=ses_s + 1)
            t_s = np.arange(ses_s + 1) / config.D
            med_s = np.array([modelos.media_ou(ou_s.x0, ou_s.kappa, ou_s.theta, t) for t in t_s])
            de_s = np.array([modelos.desv_ou(ou_s.kappa, ou_s.sigma, t) for t in t_s])
            z90 = norm.ppf(0.90)
            fig_s = go.Figure()
            fig_s.add_trace(go.Scatter(x=spread.index, y=spread, mode="lines", name=f"ln({a_p}) − ln({b_p})",
                                       line=dict(color=ui.TEXTO, width=1.4),
                                       hovertemplate="%{x|%d/%m/%Y}: %{y:.1%}<extra></extra>"))
            if ou_s.kappa > 0:
                for signo in (1, -1):
                    fig_s.add_hline(y=ou_s.theta + signo * 2 * ou_s.desv_largo_plazo, line_color=ui.TENUE,
                                    line_dash="dash", line_width=1)
            fig_s.add_hline(y=ou_s.theta, line_color=ui.ORO, line_dash="dot", line_width=1.5,
                            annotation_text="Nivel de largo plazo θ (líneas punteadas: ± 2 desviaciones)",
                            annotation_font_color=ui.TENUE, annotation_position="top left")
            fig_s.add_trace(go.Scatter(x=fechas_s, y=med_s + z90 * de_s, mode="lines", line=dict(width=0),
                                       showlegend=False, hoverinfo="skip"))
            fig_s.add_trace(go.Scatter(x=fechas_s, y=med_s - z90 * de_s, mode="lines", line=dict(width=0),
                                       fill="tonexty", fillcolor="rgba(217,181,106,.22)",
                                       name="Rango probable a 6 meses (P10–P90)", hoverinfo="skip"))
            fig_s.add_trace(go.Scatter(x=fechas_s, y=med_s, mode="lines", name="Camino esperado",
                                       line=dict(color=ui.ORO_CLARO, width=2.5),
                                       hovertemplate="%{x|%d/%m/%Y}: %{y:.1%}<extra></extra>"))
            fig_s.update_yaxes(title_text=f"Rendimiento de {a_p} menos {b_p} (log)", tickformat=".0%")
            fig_s.update_layout(legend=dict(orientation="h", y=1.12, x=0), hovermode="x unified")
            st.markdown(ui.html_kpis([
                ("Diferencia hoy", lecturas._pct(float(np.expm1(ou_s.x0))), f"rendimiento de {a_p} sobre {b_p}"),
                ("Distancia al nivel normal", f"{lecturas._num(ou_s.z_hoy, 1)} desv.", "0 = en su nivel de largo plazo"),
                ("Vida media", lecturas._anios_o_dias(ou_s.vida_media), "tiempo en cerrar media separación"),
                ("Prueba de reversión", f"t = {lecturas._num(ou_s.t_df, 2)}",
                 {"fuerte": "significativa al 5 %", "débil": "significativa solo al 10 %",
                  "ninguna": "no significativa"}[ou_s.evidencia]),
            ]), unsafe_allow_html=True)
            st.plotly_chart(ui.estilo_fig(fig_s, 380), width="stretch", config=ui.PLOTLY_CONFIG)
            st.markdown(ui.html_lectura(lecturas.lectura_spread(a_p, b_p, ou_s, ou_s.x0, spread.index[0])),
                        unsafe_allow_html=True)
        filas_ou.append({"Serie": f"ln({a_p}) − ln({b_p})", "Desde": spread.index[0], "Hasta": spread.index[-1],
                         "Observaciones": ou_s.n + 1, "κ": ou_s.kappa, "θ": ou_s.theta, "σ": ou_s.sigma, "b AR(1)": ou_s.b,
                         "t Dickey-Fuller": ou_s.t_df, "Evidencia de reversión": ou_s.evidencia,
                         "Vida media (sesiones)": ou_s.vida_media * config.D, "Valor hoy": ou_s.x0})
        ou_tabla = pd.DataFrame(filas_ou).set_index("Serie")

        with st.expander("Ver ecuaciones, calibración y curva de Vasicek"):
            st.markdown(f"""
- **Modelo:** dx = κ(θ − x)dt + σ dW. Si x está por encima de θ, el término κ(θ − x) es negativo y lo empuja hacia
  abajo (y al revés). Vida media de una desviación: ln 2 / κ. Desviación de largo plazo: σ / √(2κ).
- **Calibración exacta:** regresión AR(1) diaria x_t+1 = a + b·x_t + ε, con b = e^(−κΔt), θ = a / (1 − b) y
  σ = desv(ε)·√(2κ / (1 − b²)), Δt = 1/252.
- **¿Revierte de verdad?** Prueba de Dickey-Fuller: t = (b − 1) / ee(b). Hay reversión significativa si t < −2,86 (5 %)
  o, de forma débil, si t < −2,57 (10 %). Si b ≥ 1, se toma κ = 0 (camino al azar).
- **Tasa (Vasicek):** x = ln(1 + IRX/100), la misma conversión que usa la app para r. Se calibra con todo el histórico
  descargado. Las trayectorias usan la transición exacta y la semilla {semilla_base}.
- **Prima:** un pago en T se descuenta con la tasa promedio r̄ = (1/T)∫r dt, que en Vasicek es normal con
  media θ + (r0 − θ)(1 − e^(−κT))/(κT). La opción se revalora con r̄ en sus percentiles 10 y 90.
- **Supuesto:** las trayectorias de la tasa están bajo la medida física (sin prima por riesgo de tasa). Es una prueba
  de sensibilidad, no un modelo de opciones con tasa estocástica.
- **Acciones del mismo sector:** x = ln(P_A / P_A,0) − ln(P_B / P_B,0) con precios ajustados de toda la historia.
""")
            st.dataframe(ou_tabla.style.format({"κ": "{:.3f}", "θ": "{:.4f}", "σ": "{:.4f}", "b AR(1)": "{:.5f}",
                                                "t Dickey-Fuller": "{:.2f}", "Vida media (sesiones)": "{:.0f}",
                                                "Valor hoy": "{:.4f}", "Tasa promedio P10": "{:.4f}",
                                                "Tasa promedio P90": "{:.4f}", "Prima tasa P10": "{:.3f}",
                                                "Prima hoy": "{:.3f}", "Prima tasa P90": "{:.3f}",
                                                "Desde": "{:%Y-%m-%d}", "Hasta": "{:%Y-%m-%d}"}, na_rep="—"),
                         width="stretch")
            if irx is not None and len(irx) >= 60:
                curva = modelos.curva_vasicek(r_op, ou_r.kappa, ou_r.theta, ou_r.sigma)
                st.markdown("**Curva de tasas implícita (Vasicek, bono cero cupón):** "
                            + " · ".join(f"{p:g} años: {lecturas._pct(y, 2)}" for p, y in
                                         zip(curva["Plazo (años)"], curva["Tasa"])))

    # ---------- Herding: comportamiento en manada ----------
    with sub[4]:
        h1_, h2_, _ = st.columns([1, 1.2, 1.8])
        with h1_:
            ventana_h = st.select_slider("Ventana de la correlación", options=[21, 63, 126], value=63, key="her_ventana",
                                         format_func=lambda v: f"{v} sesiones")
        with h2_:
            h_sel = st.slider("Intensidad de manada h", 0.0, 1.0, 0.25, 0.05, key="her_h",
                              help="Parte del movimiento de cada acción que viene de copiar a la multitud.")
        rho_movil = memoria.correlacion_movil(x_all, ventana_h)
        cs = modelos.csad(x_all)
        corr_v = cov_v / np.outer(np.sqrt(np.diag(cov_v)), np.sqrt(np.diag(cov_v)))
        rho_cal = modelos.correlacion_promedio(corr_v)
        rho_pico = float(rho_movil.max())
        h_pico = modelos.h_para_correlacion(rho_cal, rho_pico)
        st.markdown(ui.html_intro(lecturas.INTRO_HERDING), unsafe_allow_html=True)
        st.markdown(ui.html_variables(lecturas.variables_herding(float(rho_movil.iloc[-1]), rho_pico, h_sel, cs.g2)),
                    unsafe_allow_html=True)

        with st.container(key="tarjeta_her_correlacion"):
            st.markdown(ui.titulo_tarjeta(f"¿Se mueven más juntas nuestras acciones en algunos momentos? · ventanas de "
                                          f"{ventana_h} sesiones"), unsafe_allow_html=True)
            fig_rc = go.Figure()
            fig_rc.add_trace(go.Scatter(x=rho_movil.index, y=rho_movil, mode="lines", name="Correlación promedio",
                                        line=dict(color=ui.ORO_CLARO, width=1.8), fill="tozeroy",
                                        fillcolor="rgba(217,181,106,.08)",
                                        hovertemplate="%{x|%d/%m/%Y}: %{y:.2f}<extra></extra>"))
            fig_rc.add_hline(y=rho_cal, line_color=ui.TENUE, line_dash="dot", line_width=1,
                             annotation_text="Usada en el VaR (ventana de calibración)", annotation_font_color=ui.TENUE)
            pico_f = rho_movil.idxmax()
            fig_rc.add_trace(go.Scatter(x=[pico_f], y=[rho_pico], mode="markers", name="Pico",
                                        marker=dict(size=12, color=ui.PALETA[8], line=dict(color="#070706", width=1)),
                                        hovertemplate="Pico %{x|%d/%m/%Y}: %{y:.2f}<extra></extra>"))
            fig_rc.update_yaxes(title_text="Correlación promedio entre pares", range=[0, 1])
            fig_rc.update_layout(showlegend=False, hovermode="x unified")
            st.plotly_chart(ui.estilo_fig(fig_rc, 320), width="stretch", config=ui.PLOTLY_CONFIG)
            st.markdown(ui.html_lectura(lecturas.lectura_correlacion_movil(rho_movil, rho_cal, ventana_h)),
                        unsafe_allow_html=True)

        with st.container(key="tarjeta_her_csad"):
            st.markdown(ui.titulo_tarjeta("En los días extremos, ¿todas las acciones hacen lo mismo?"),
                        unsafe_allow_html=True)
            c1_, c2_ = st.columns([1.4, 1], gap="medium")
            with c1_:
                dcs = cs.datos
                fig_cs = go.Figure()
                fig_cs.add_trace(go.Scatter(x=dcs["R_m"], y=dcs["CSAD"], mode="markers", name="Días",
                                            marker=dict(size=5, color="rgba(232,226,212,.35)"),
                                            customdata=dcs.index.strftime("%d/%m/%Y"),
                                            hovertemplate="%{customdata}: mercado %{x:.2%}, CSAD %{y:.2%}<extra></extra>"))
                xs_cs = np.linspace(dcs["R_m"].min(), dcs["R_m"].max(), 100)
                fig_cs.add_trace(go.Scatter(x=xs_cs, y=cs.a + cs.g1 * np.abs(xs_cs) + cs.g2 * xs_cs ** 2, mode="lines",
                                            name="Curva ajustada", line=dict(color=ui.ORO_CLARO, width=2.5),
                                            hoverinfo="skip"))
                fig_cs.update_xaxes(title_text="Retorno promedio del mercado (10 acciones)", tickformat=".0%")
                fig_cs.update_yaxes(title_text="Dispersión entre acciones (CSAD)", tickformat=".1%")
                fig_cs.update_layout(showlegend=False)
                st.plotly_chart(ui.estilo_fig(fig_cs, 320), width="stretch", config=ui.PLOTLY_CONFIG)
            with c2_:
                st.markdown(ui.html_lectura(lecturas.lectura_csad(cs)), unsafe_allow_html=True)

        var_manada = modelos.var_por_manada(cov_v, w_v, B, alpha_v, h_v)
        v_h = riesgo.var_normal(modelos.cov_manada(cov_v, h_sel), w_v, B, alpha_v, h_v)["var"]
        v_pico = riesgo.var_normal(modelos.cov_manada(cov_v, h_pico), w_v, B, alpha_v, h_v)["var"]
        v0_h, v1_h = float(var_manada["VaR"].iloc[0]), float(var_manada["VaR"].iloc[-1])
        perd_0 = memoria.simular_manada(cov_v, w_v, B, 0.0, h_v, config.N_ESCENARIOS_VAR, semilla_base)
        perd_h = memoria.simular_manada(cov_v, w_v, B, h_sel, h_v, config.N_ESCENARIOS_VAR, semilla_base)
        var_mc0, _ = riesgo.var_es(perd_0, alpha_v)
        var_mch, es_mch = riesgo.var_es(perd_h, alpha_v)
        with st.container(key="tarjeta_her_var"):
            st.markdown(ui.titulo_tarjeta(f"¿Qué le pasa al VaR de la cartera si todos se copian? · {cartera_v}"),
                        unsafe_allow_html=True)
            st.markdown(ui.html_kpis([
                ("VaR sin manada adicional", lecturas._usd(v0_h), f"normal · {alpha_v:.0%} · {h_v} días".replace(".", ",")),
                (f"VaR con h = {h_sel:.2f}".replace(".", ","), lecturas._usd(v_h),
                 f"{v_h / v0_h - 1:+.1%} frente a hoy".replace(".", ",")),
                ("VaR en el pico histórico", lecturas._usd(v_pico), f"correlación {rho_pico:.2f}".replace(".", ",")),
                ("VaR si todos se copian (h = 1)", lecturas._usd(v1_h), f"{v1_h / B:.1%} del capital".replace(".", ",")),
            ]), unsafe_allow_html=True)
            g1h, g2h = st.columns([1.2, 1], gap="medium")
            with g1h:
                fig_vh = go.Figure()
                fig_vh.add_trace(go.Scatter(x=var_manada["h"], y=var_manada["ES"], mode="lines", name="ES",
                                            line=dict(color=ui.TENUE, width=2, dash="dot"),
                                            hovertemplate="h = %{x:.2f}: USD %{y:,.0f}<extra></extra>"))
                fig_vh.add_trace(go.Scatter(x=var_manada["h"], y=var_manada["VaR"], mode="lines", name="VaR",
                                            line=dict(color=ui.ORO_CLARO, width=2.5),
                                            hovertemplate="h = %{x:.2f}: USD %{y:,.0f}<extra></extra>"))
                fig_vh.add_trace(go.Scatter(x=[h_sel, h_pico], y=[v_h, v_pico], mode="markers+text",
                                            text=["Elegido", "Pico histórico"], textposition="top left",
                                            textfont=dict(color=ui.TENUE), name="Escenarios",
                                            marker=dict(size=11, color=ui.ORO, line=dict(color="#070706", width=1)),
                                            hovertemplate="h = %{x:.2f}: USD %{y:,.0f}<extra></extra>"))
                fig_vh.update_xaxes(title_text="Intensidad de manada h (0 = correlaciones de hoy, 1 = todas iguales)")
                fig_vh.update_yaxes(title_text=f"Pérdida a {h_v} días (USD)")
                fig_vh.update_layout(legend=dict(orientation="h", y=1.12, x=0))
                st.plotly_chart(ui.estilo_fig(fig_vh, 320), width="stretch", config=ui.PLOTLY_CONFIG)
            with g2h:
                fig_hh = go.Figure()
                for perd, nom, color in ((perd_0, "Sin manada", "rgba(232,226,212,.45)"),
                                         (perd_h, f"Con h = {h_sel:.2f}".replace(".", ","), "rgba(217,181,106,.6)")):
                    fig_hh.add_trace(go.Histogram(x=perd, nbinsx=70, name=nom, marker_color=color,
                                                  hovertemplate="USD %{x:,.0f}: %{y}<extra></extra>"))
                for valor, texto, color in ((var_mc0, "VaR sin manada", ui.TEXTO), (var_mch, "VaR con manada", ui.ORO_CLARO)):
                    fig_hh.add_vline(x=valor, line_color=color, line_dash="dot", line_width=1.5)
                fig_hh.update_layout(barmode="overlay", legend=dict(orientation="h", y=1.15, x=0))
                fig_hh.update_xaxes(title_text="Pérdida simulada (USD)")
                fig_hh.update_yaxes(title_text="Escenarios")
                st.plotly_chart(ui.estilo_fig(fig_hh, 320), width="stretch", config=ui.PLOTLY_CONFIG)
            n_esc = f"{config.N_ESCENARIOS_VAR:,}".replace(",", ".")
            st.markdown(f'<div class="q-nota">Simulación de {n_esc} escenarios (semilla {semilla_base}) '
                        f'con los mismos choques en los dos casos: el VaR simulado pasa de {lecturas._usd(var_mc0)} a '
                        f'{lecturas._usd(var_mch)} y el ES con manada es {lecturas._usd(es_mch)}. Pesos de la cartera '
                        'elegida en la pestaña Riesgo; covarianza de la ventana de calibración.</div>',
                        unsafe_allow_html=True)
            st.markdown(ui.html_lectura(lecturas.lectura_var_manada(v0_h, v_h, v_pico, h_sel, h_pico, rho_pico, v1_h, B,
                                                                    alpha_v, h_v)), unsafe_allow_html=True)

        with st.expander("Ver ecuaciones y el método"):
            st.markdown(f"""
- **Correlación promedio:** promedio de las 45 correlaciones entre pares de acciones, en ventanas móviles de
  {ventana_h} sesiones de log-retornos ({len(x_all)} retornos de toda la historia).
- **Prueba CSAD (Chang, Cheng y Khorana, 2000):** CSAD_t = (1/N)·Σ|R_i,t − R_m,t|, con R_m,t el promedio simple del día.
  Se estima CSAD_t = a + γ1·|R_m,t| + γ2·R_m,t². Sin manada, γ2 ≥ 0; con manada, γ2 < 0 y significativo.
  Errores estándar de Newey-West con 5 rezagos. Resultado: a = {cs.a:.5f}, γ1 = {cs.g1:.4f}, γ2 = {cs.g2:.4f}
  (t = {cs.t_g2:.2f}), R² = {cs.r2:.3f}.
- **Inversionistas que se copian:** el movimiento de cada acción es r_i = σ_i·(√h·M + √(1 − h)·ε_i). M es lo que hace
  la multitud (igual para todas) y ε_i la información propia de cada acción, con las correlaciones reales. Así, la
  volatilidad de cada acción no cambia, pero la correlación pasa a ser ρ_ij(h) = h + (1 − h)·ρ_ij.
- **VaR:** VaR normal de la pestaña Riesgo (media cero) con esa covarianza: B·Φ⁻¹(α)·√(h_VaR·wᵀΣ(h)w). Con h = 0
  coincide con el de Riesgo. El pico histórico equivale a h = (ρ_pico − ρ̄) / (1 − ρ̄) = {h_pico:.2f}.
""")
            st.dataframe(var_manada.style.format({"h": "{:.2f}", "Correlación promedio": "{:.2f}", "VaR": "{:,.0f}",
                                                  "ES": "{:,.0f}"}), hide_index=True, width="stretch")

    # ---------- Opciones reales ----------
    with sub[5]:
        energia = [t for t in tickers if sect[t] == "Energía"][:2] or tickers[:2]
        sig_ref = {t: float(par_hoy.loc[t, "σ anual"]) for t in energia}
        q_ref = float(np.mean([res["q"][t] for t in energia]))
        nombre_prom = "Promedio " + " y ".join(energia)
        fuentes_s = [nombre_prom, *energia, "Manual"]
        o1, o2, o3, o4, o5 = st.columns(5)
        with o1:
            v0_or = st.number_input("Valor del proyecto V₀ (USD M)", value=100.0, min_value=1.0, step=5.0, key="or_v0")
            expansion_or = st.number_input("Expansión (% del valor)", value=50.0, min_value=0.0, step=5.0,
                                           key="or_expansion") / 100
        with o2:
            inv_or = st.number_input("Inversión I (USD M)", value=110.0, min_value=0.0, step=5.0, key="or_inversion")
            costo_exp_or = st.number_input("Costo de expandir (USD M)", value=40.0, min_value=0.0, step=5.0,
                                           key="or_costo_exp")
        with o3:
            t_or = st.number_input("Plazo para decidir T (años)", value=3.0, min_value=0.25, step=0.25, key="or_t")
            rescate_or = st.number_input("Valor de rescate A (USD M)", value=60.0, min_value=0.0, step=5.0,
                                         key="or_rescate")
        with o4:
            n_or = int(st.number_input("Pasos del árbol N", value=36, min_value=2, max_value=200, step=6, key="or_n"))
            delta_or = st.number_input("Flujos perdidos δ (decimal)", value=round(q_ref, 4), min_value=0.0,
                                       max_value=0.5, format="%.4f", key="or_delta",
                                       help=f"Por defecto, el dividendo promedio de {' y '.join(energia)}.")
        with o5:
            fuente_s = st.selectbox("Volatilidad σ", fuentes_s, key="or_fuente_sigma")
            if fuente_s == "Manual":
                sig_or = st.number_input("σ manual (decimal)", value=0.25, min_value=0.01, format="%.4f",
                                         key="or_sigma_manual")
            else:
                sig_or = float(np.mean(list(sig_ref.values()))) if fuente_s == nombre_prom else sig_ref[fuente_s]
                st.markdown(f'<div class="q-nota" style="padding-top:8px">σ = {lecturas._pct(sig_or)} · r = '
                            f'{lecturas._pct(r_op, 2)}</div>', unsafe_allow_html=True)
        nombre_sig = {nombre_prom: f"el promedio de {' y '.join(energia)}", "Manual": "un valor manual"}.get(
            fuente_s, fuente_s)
        st.markdown(ui.html_intro(lecturas.INTRO_OPCIONES_REALES), unsafe_allow_html=True)
        st.markdown(ui.html_variables(lecturas.variables_opciones_reales(v0_or, inv_or, sig_or, r_op, delta_or, t_or,
                                                                         nombre_sig)), unsafe_allow_html=True)
        try:
            res_or = {tipo: modelos.opcion_real(v0_or, sig_or, r_op, delta_or, t_or, n_or, tipo, inv_or, expansion_or,
                                                costo_exp_or, rescate_or) for tipo in modelos.TIPOS_OPCION_REAL}
        except ValueError as e_or:
            res_or = None
            st.warning(f"No se puede construir el árbol con estos valores: {e_or}")
        if res_or is not None:
            vpn_or = v0_or - inv_or
            or_tabla = pd.DataFrame([{"Flexibilidad": tipo, "VPN estático": o_.vpn, "Valor de la opción": o_.opcion,
                                      "VPN estratégico": o_.vpn_estrategico, "Valor de la flexibilidad":
                                      modelos.flexibilidad(o_), "Decisión hoy": o_.decision[0][0]}
                                     for tipo, o_ in res_or.items()]).set_index("Flexibilidad")
            st.markdown(ui.html_kpis([
                ("VPN estático", lecturas._usd(vpn_or) + " M", "invertir hoy, sin flexibilidad"),
                ("Con opción de esperar", lecturas._usd(res_or["esperar"].vpn_estrategico) + " M",
                 f"decisión hoy: {res_or['esperar'].decision[0][0].lower()}"),
                ("Con expandir y abandonar", lecturas._usd(res_or["expandir o abandonar"].vpn_estrategico) + " M",
                 "proyecto hecho hoy, con las dos opciones"),
                ("Valor de expandir o abandonar", lecturas._usd(res_or["expandir o abandonar"].opcion) + " M",
                 "lo que suman las dos opciones al VPN"),
            ]), unsafe_allow_html=True)

            with st.container(key="tarjeta_or_valor"):
                st.markdown(ui.titulo_tarjeta("¿Cuánto vale la flexibilidad del proyecto?"), unsafe_allow_html=True)
                nombres_or = ["VPN estático", "Esperar", "Expandir", "Abandonar", "Expandir o abandonar"]
                valores_or = [vpn_or] + [res_or[t].vpn_estrategico for t in modelos.TIPOS_OPCION_REAL]
                colores_or = [ui.TENUE] + [ui.ORO_CLARO, ui.PALETA[5], ui.PALETA[8], ui.ORO]
                fig_or = go.Figure(go.Bar(x=nombres_or, y=valores_or, marker_color=colores_or,
                                          text=[f"{v:,.1f}".replace(",", "X").replace(".", ",").replace("X", ".")
                                                for v in valores_or], textposition="outside",
                                          textfont=dict(color=ui.TEXTO),
                                          hovertemplate="%{x}: USD %{y:.2f} M<extra></extra>"))
                fig_or.add_hline(y=0, line_color=ui.TENUE, line_width=1)
                fig_or.update_yaxes(title_text="VPN con la flexibilidad (USD M)")
                fig_or.update_layout(showlegend=False)
                st.plotly_chart(ui.estilo_fig(fig_or, 340), width="stretch", config=ui.PLOTLY_CONFIG)
                st.markdown(ui.html_lectura(lecturas.lectura_opciones_reales(res_or, vpn_or)), unsafe_allow_html=True)

            with st.container(key="tarjeta_or_mapa"):
                st.markdown(ui.titulo_tarjeta("¿Cuándo conviene ejercer la flexibilidad?"), unsafe_allow_html=True)
                m1o, _ = st.columns([3, 1])
                with m1o:
                    tipo_mapa = st.radio("Flexibilidad", list(modelos.TIPOS_OPCION_REAL), horizontal=True,
                                         key="or_tipo_mapa")
                o_m = res_or[tipo_mapa]
                colores_d = {"Esperar": ui.TENUE, "Seguir": ui.TENUE, "No invertir": "#4a4438", "Invertir": ui.PALETA[5],
                             "Expandir": ui.PALETA[5], "Abandonar": ui.PALETA[8]}
                nodos = pd.DataFrame([{"t": t_, "V": v_, "Decisión": d_}
                                      for t_, vs, ds in zip(o_m.tiempos, o_m.valores_v, o_m.decision)
                                      for v_, d_ in zip(vs, ds)])
                fig_dm = go.Figure()
                for dec, sel in nodos.groupby("Decisión", sort=False):
                    fig_dm.add_trace(go.Scatter(x=sel["t"], y=sel["V"], mode="markers", name=dec,
                                                marker=dict(size=7 if n_or <= 60 else 5, color=colores_d.get(dec, ui.TEXTO),
                                                            line=dict(color="#070706", width=0.5)),
                                                hovertemplate=f"{dec}<br>año %{{x:.2f}} · V = USD %{{y:.1f}} M<extra></extra>"))
                ref = {"esperar": (inv_or, "Inversión I"), "abandonar": (rescate_or, "Rescate A")}.get(tipo_mapa)
                if ref:
                    fig_dm.add_hline(y=ref[0], line_color=ui.TENUE, line_dash="dot", annotation_text=ref[1],
                                     annotation_font_color=ui.TENUE)
                fig_dm.update_yaxes(title_text="Valor del proyecto V (USD M)",
                                    range=[0, min(float(nodos["V"].max()), 3 * max(v0_or, inv_or, rescate_or)) * 1.05])
                fig_dm.update_xaxes(title_text="Años")
                fig_dm.update_layout(legend=dict(orientation="h", y=1.12, x=0))
                st.plotly_chart(ui.estilo_fig(fig_dm, 380), width="stretch", config=ui.PLOTLY_CONFIG)
                st.markdown(ui.html_lectura(lecturas.lectura_mapa_decision(o_m, tipo_mapa)), unsafe_allow_html=True)

            sens_or = modelos.sensibilidad_volatilidad(v0_or, r_op, delta_or, t_or, n_or, inv_or, expansion_or,
                                                       costo_exp_or, rescate_or)
            with st.container(key="tarjeta_or_sigma"):
                st.markdown(ui.titulo_tarjeta("¿Vale más la flexibilidad cuando hay más incertidumbre?"),
                            unsafe_allow_html=True)
                fig_so = go.Figure()
                for tipo, color in zip(modelos.TIPOS_OPCION_REAL, [ui.ORO_CLARO, ui.PALETA[5], ui.PALETA[8], ui.ORO]):
                    fig_so.add_trace(go.Scatter(x=sens_or["σ"], y=sens_or[tipo], mode="lines", name=tipo.capitalize(),
                                                line=dict(color=color, width=2.2),
                                                hovertemplate="σ = %{x:.0%}: USD %{y:.2f} M<extra></extra>"))
                for i_r, (t_r, s_r) in enumerate(sorted(sig_ref.items(), key=lambda kv: kv[1])):
                    fig_so.add_vline(x=s_r, line_color=ui.TENUE, line_dash="dot", annotation_text=t_r,
                                     annotation_font_color=ui.TENUE,
                                     annotation_position="top left" if i_r == 0 else "top right")
                fig_so.update_xaxes(title_text="Volatilidad del proyecto σ", tickformat=".0%")
                fig_so.update_yaxes(title_text="Valor de la flexibilidad (USD M)")
                fig_so.update_layout(legend=dict(orientation="h", y=1.12, x=0), hovermode="x unified")
                st.plotly_chart(ui.estilo_fig(fig_so, 320), width="stretch", config=ui.PLOTLY_CONFIG)
                st.markdown(ui.html_lectura(lecturas.lectura_sensibilidad_or(sens_or, sig_ref)), unsafe_allow_html=True)

            with st.expander("Ver ecuaciones, tabla y supuestos"):
                par_or = res_or["esperar"].parametros
                st.markdown(f"""
- **Árbol:** el mismo CRR de la pestaña Opciones, pero sobre el valor del proyecto V en lugar del precio de una acción:
  Δt = T/N = {par_or.dt:.4f}, u = e^(σ√Δt) = {par_or.u:.4f}, d = 1/u, p = (e^((r − δ)Δt) − d)/(u − d) = {par_or.p:.4f}.
  δ hace el papel del dividendo q: son los flujos que no se reciben mientras se espera.
- **Valor de ejercer en cada nodo (opción americana):** esperar → V − I; expandir → e·V − I_e (crecer un
  {expansion_or:.0%} pagando {costo_exp_or:.1f}); abandonar → A − V; las dos → el mayor, y se ejerce una sola.
  Retroceso: valor = máx(ejercer, e^(−rΔt)[p·V_arriba + (1 − p)·V_abajo]).
- **VPN estratégico:** VPN estático + valor de la flexibilidad. En «esperar», el proyecto aún no se ha hecho y el
  VPN estratégico es el valor de la opción; el valor de esperar es lo que supera a máx(VPN, 0).
- **Control:** la opción de esperar coincide con una call americana CRR (S0 = V₀, K = I, q = δ) y la de abandonar con
  una put americana (K = A); lo verifican las pruebas automáticas.
- **Supuestos:** el valor del proyecto se mueve como un GBM con la volatilidad de las acciones de energía (es una
  aproximación: la acción también refleja deuda y otros proyectos) y la tasa r del Tesoro. Cifras en millones de USD,
  editables arriba.
""")
                st.dataframe(or_tabla.style.format({"VPN estático": "{:.2f}", "Valor de la opción": "{:.2f}",
                                                    "VPN estratégico": "{:.2f}", "Valor de la flexibilidad": "{:.2f}"}),
                             width="stretch")

    return {"mer": mer, "tab_h": tab_h, "v_mer": v_mer, "ou_tabla": ou_tabla, "var_manada": var_manada,
            "rho_movil": rho_movil, "or_tabla": or_tabla}
