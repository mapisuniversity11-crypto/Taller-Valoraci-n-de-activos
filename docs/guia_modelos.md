# Guía para agregar un modelo a la pestaña Modelos

La pestaña **Modelos** tiene seis subpestañas. Tres ya están completas (browniano estándar, Merton y browniano
fraccional) y sirven de ejemplo. Las otras tres muestran por ahora una tarjeta "en construcción":

| Subpestaña | Datos sugeridos | Pregunta que responde |
|---|---|---|
| Ornstein-Uhlenbeck · Vasicek | Tasa del Tesoro a 13 semanas (`^IRX`, histórico con yfinance) y la diferencia de log-precios XOM − CVX | ¿Cuánto puede moverse la tasa durante la vida de la opción y cuánto cambia la prima? Si dos acciones del mismo sector se separan, ¿vuelven a juntarse? |
| Herding · manada | Las 10 acciones y la cartera | ¿Qué pasa con el riesgo cuando todos actúan igual? Correlación promedio en el tiempo y simulación de inversionistas que se copian, con su efecto en el VaR |
| Opciones reales | Proyecto de ejemplo (cifras editables), σ de XOM y CVX, r del Tesoro, árbol CRR de `src/crr.py` | ¿Cuánto vale poder esperar, expandir o abandonar? |

## Estructura que debe tener cada subpestaña

Mismo orden y mismo estilo que las tres que ya están:

1. **Selectores** arriba, si hacen falta (`st.selectbox`, en columnas angostas).
2. **Qué busca el modelo:** dos o tres frases en lenguaje de inversionista, sin teoría.
   `st.markdown(ui.html_intro(texto), unsafe_allow_html=True)`
3. **Variables:** una tarjeta por variable con su valor calibrado, qué significa y qué pasa si sube.
   `st.markdown(ui.html_variables([(símbolo, nombre, valor, qué significa, qué pasa si sube), ...]), unsafe_allow_html=True)`
4. **Aplicación a nuestro caso:** una o dos tarjetas, cada una con un gráfico simple y su lectura.
   ```python
   with st.container(key="tarjeta_<nombre>"):       # la key debe empezar por "tarjeta"
       st.markdown(ui.titulo_tarjeta("Pregunta que responde el gráfico"), unsafe_allow_html=True)
       st.plotly_chart(ui.estilo_fig(fig, 340), width="stretch", config=ui.PLOTLY_CONFIG)
       st.markdown(ui.html_lectura(lecturas.lectura_<modelo>(...)), unsafe_allow_html=True)
   ```
5. **Ecuaciones, tablas y método** solo dentro de `st.expander("Ver ...")`.

## Dónde va cada cosa

- **Cálculos:** funciones en `src/modelos.py`, sin Streamlit, para poder probarlas.
- **Frases:** en `src/lecturas.py`. Se calculan con los datos de cada ejecución, así que se actualizan solas.
  Usa `_usd`, `_pct` y `_num` para el formato en español.
- **Pantalla:** en `vistas/modelos.py`, dentro de `with sub[3]:`, `with sub[4]:` o `with sub[5]:`. Hay que borrar la
  tarjeta "en construcción" de la lista `pendientes`. No hace falta tocar `app.py` ni las otras vistas.
- **Pruebas:** en `tests/test_modelos.py`, por ejemplo un caso con respuesta conocida.
- **Decisiones:** una entrada nueva en `docs/decisiones.md`.

## Variables ya disponibles en `vistas/modelos.py`

Al inicio de `mostrar(ctx)` se sacan del contexto `ctx` las variables que usa la pestaña. Si necesitas otra de esta
tabla, agrégala igual: `w_v = ctx["w_v"]`. Las que vienen de otras pestañas (como `w_v`, de Riesgo) ya están en `ctx`
porque Riesgo se dibuja antes que Modelos.

| Variable | Contenido |
|---|---|
| `tickers`, `sect` | Lista de tickers y su sector |
| `limpio.ajustados` | Precios ajustados limpios, toda la historia |
| `precios_hoy` | Últimos 504 retornos (505 precios) ajustados |
| `s0_hoy` | Último cierre sin ajustar por ticker |
| `par_hoy` | Parámetros calibrados en esa ventana (`"σ anual"`, `"a = μ − q"`, ...) |
| `res["r"]`, `res["q"]` | Tasa libre de riesgo continua y dividendos continuos |
| `opcion` | Entradas del panel de la opción (S0, K, T, r, q, σ, N, tipo) |
| `carteras`, `cov_v`, `B`, `alpha_v`, `h_v` | Pesos de las carteras, covarianza diaria, capital, confianza y horizonte del VaR |
| `cartera_v`, `w_v`, `r_hist`, `r_norm`, `r_mc` | Cartera elegida en Riesgo, sus pesos y su VaR/ES por los tres métodos |
| `semilla_base` | Semilla de la app; úsala para que los resultados sean reproducibles |

## Reglas de estilo

- Gráficos simples: líneas, barras o puntos, con colores de `ui` (`ui.ORO_CLARO`, `ui.TEXTO`, `ui.TENUE`, `ui.PALETA`).
- Cada gráfico tiene título en forma de pregunta y una lectura que la responde con los números del caso.
- Unidades: tasas y volatilidades anuales en decimal, tiempo en años, 252 sesiones por año.
- Si un cálculo tarda (simulaciones grandes), guárdalo en memoria con una función en `vistas/memoria.py`.
- Si algo de tu modelo debe salir en la exportación CSV, devuélvelo en el `return` de `mostrar` y agrégalo a
  `tablas_x` en `vistas/exportacion.py`.
- Después de cambiar algo en `src/` o `vistas/`, reinicia la app (Ctrl+C y `streamlit run app.py`).
- Antes de empezar a trabajar: `git pull`. Al terminar: `pytest`, luego `git add`, `git commit` y `git push`.
