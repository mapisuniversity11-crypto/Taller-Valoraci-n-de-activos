# Bitácora de decisiones

Cada decisión con su fecha, qué se decidió y por qué. Sirve para la sustentación
("cada integrante explicará una función y una decisión").

---

## D1 — Herramienta: Python + Streamlit
**Fecha:** 2026-10-21
**Decisión:** construir la app en Streamlit.
**Por qué:** es la opción más rápida para una app de análisis con entradas y pestañas;
permite tablas editables (corrección manual de sectores) y gráficos interactivos sin
programar frontend. Dash daría más control visual, pero no lo necesitamos.

## D2 — Separar cálculo e interfaz
**Fecha:** 2026-10-21
**Decisión:** toda la lógica vive en `src/`; `app.py` solo lee entradas y muestra.
**Por qué:** cada función se puede probar sin abrir la app (por ejemplo, el control
del CRR), y cada integrante puede explicar su módulo de forma independiente.

## D3 — Universo de 10 acciones en 5 sectores
**Fecha:** 2026-10-21

| Sector | Tickers |
|---|---|
| Tecnología | AAPL, MSFT |
| Financiero | JPM, BAC |
| Energía | XOM, CVX |
| Salud | JNJ, PFE |
| Consumo básico | KO, PG |

**Por qué:**
- Todas cotizan en USD en NYSE/NASDAQ → mismo calendario, sesiones comunes casi completas.
- Historia larga y alta liquidez → descarga confiable de más de 3 años.
- Cinco sectores (el taller pide al menos cuatro) con dos acciones cada uno: con límite
  de 20 % por acción, un sector podría llegar a 40 %, así que el tope de 35 % por sector
  sí es relevante, pero el problema de mínimo VaR sigue siendo factible
  (5 sectores × 35 % = 175 % ≥ 100 %).
- Sectores con comportamientos distintos (tecnología vs. defensivos vs. energía) →
  correlaciones variadas, más interesante para la diversificación.

## D4 — Acción para la opción: AAPL
**Fecha:** 2026-10-21
**Decisión:** usar AAPL como subyacente del contrato.
**Por qué:** es de las cadenas de opciones más líquidas del mercado (muchos strikes y
vencimientos, spreads estrechos), lo que facilita encontrar los 9 candidatos y tener
bid/ask utilizable. Las opciones sobre acciones de EE. UU. son de estilo **americano**.

## D5 — Período de descarga: ~4 años
**Fecha:** 2026-10-21
**Decisión:** descargar aproximadamente 4 años de datos diarios.
**Por qué:** se necesitan al menos 756 retornos conjuntos (504 de calibración + 252 de
evaluación); 3 años calendario dan aproximadamente 750, que se quedaría corto tras
festivos y descartes. 4 años da margen.

## D6 — Dividendos qᵢ por acción
**Fecha:** 2026-10-21
**Decisión:** la tasa de dividendos se ingresa por acción (columna `q` de la tabla), no como un único valor.
**Por qué:** el Paso 3 define aᵢ = μᵢ − qᵢ con qᵢ propio de cada acción. Ver D12 para cómo se calcula.

## D7 — Entradas del CRR en la entrega 1
**Fecha:** 2026-10-21
**Decisión:** S0, K, T, r, σ y q del árbol se ingresan manualmente, con los valores del caso de control por defecto.
**Por qué:** en la semana 1 aún no se ha elegido el contrato (Pasos 5–6). Así el árbol se puede validar
contra 10,4506 sin tocar nada. En la semana 2, K y T vendrán del contrato elegido, S0 del cierre
sin ajustar, σ de la calibración y r de la entrada general.

## D8 — Limpieza: sesiones comunes completas
**Fecha:** 2026-10-21
**Decisión:** una fecha se conserva solo si las 10 acciones tienen precio ajustado **y** sin ajustar
válido (> 0). Los precios ≤ 0 se tratan como faltantes. Los duplicados de fecha se eliminan
conservando la última observación. No se rellenan faltantes (ni con ceros ni con forward-fill).
**Por qué:** el Paso 2.3 pide sesiones comunes completas y prohíbe inventar retornos cero. Un
forward-fill crearía precisamente un retorno cero falso. Se exige también el precio sin ajustar
porque se necesita como S0 en la misma fecha. Todas las fechas descartadas se reportan en la app.

## D9 — Ventanas de calibración y evaluación
**Fecha:** 2026-10-21
**Decisión:** evaluación = últimos 252 retornos; calibración inicial = los 504 retornos
inmediatamente anteriores. Si hay más de 756 retornos, los más antiguos no entran en estas
ventanas (la app reporta cuántos), pero quedan disponibles para el rolling.
Los parámetros, la covarianza y la correlación de la pestaña Datos se calculan con la
ventana de calibración.
**Por qué:** así lo define el Paso 2.4. Usar la ventana de calibración evita mirar datos
del período de evaluación (no usar datos futuros), y el Paso 11 pide estimar Σ_R con entrenamiento.

## D10 — Fuentes de datos y respaldo
**Fecha:** 2026-10-21
**Decisión:** la app ofrece tres orígenes: Yahoo Finance, respaldo del repositorio (`data/raw`)
y CSV subido por el usuario (con procedencia obligatoria). Cada descarga de Yahoo se guarda
cruda en `data/raw` junto con `metadatos.json` (fuente, hora de consulta, fechas, `end`
exclusivo usado, sectores de Yahoo y versiones de bibliotecas).
**Por qué:** el Paso 2.2 pide guardar datos y versiones; el Paso 2.3 pide poder cargar un CSV
con procedencia si la descarga falla. El respaldo del repo hace la app reproducible el día de la
sustentación aunque Yahoo no responda.

## D11 — Sector: tabla editable vs. metadato de Yahoo
**Fecha:** 2026-10-21
**Decisión:** el sector que usa la app es el de la tabla editable de la barra lateral. El sector
que reporta Yahoo se muestra al lado para comparar.
**Por qué:** el metadato de Yahoo a veces falla o usa otra clasificación; el Paso 2.1 pide
permitir la corrección manual.

## D12 — Cálculo de qᵢ con dividendos reales
**Fecha:** 2026-10-21
**Decisión:** por defecto, qᵢ = ln(1 + D₁₂ / S), donde D₁₂ son los dividendos pagados por acción
en los 365 días previos a la última fecha (Yahoo, `Ticker.dividends`) y S es el último cierre
sin ajustar. Si falla, se usa el valor manual de la tabla.
**Por qué:** con q = 0 se ignoraban dividendos reales del 2–6 % (KO, PG, XOM, CVX, PFE, JNJ).
El Paso 3 pide documentar la aproximación del spot exdividendo.
**Supuestos de la aproximación:** el dividendo de los próximos 12 meses será similar al de los
últimos 12, y se trata como un pago continuo. Se usa el precio sin ajustar porque D₁₂ está
expresado en dólares por acción reales (no ajustados por dividendos).
**Alternativa descartada:** escribir rendimientos fijos tomados de páginas web; las fuentes no
coincidían entre sí y quedaban desactualizados.

## D13 — Tasa libre de riesgo r
**Fecha:** 2026-10-21
**Decisión:** por defecto, r = ln(1 + IRX/100), con IRX el rendimiento anual (en %) de la letra
del Tesoro de EE. UU. a 13 semanas (`^IRX` en Yahoo), en la última fecha disponible. Si falla o el
valor está fuera de 0–20 %, se usa el valor manual (4 %).
**Por qué:** es la referencia estándar de tasa libre de riesgo en USD a corto plazo; se toma de la
misma fuente y fecha que los precios; se convierte a tasa continua porque el CRR y el GBM la usan así.
**Limitación:** ^IRX es una tasa a 3 meses; para opciones a 6 meses una tasa a ese plazo sería más
precisa. La diferencia esperada es pequeña y se documentará en la semana 2.
**Nota:** al buscar la tasa en la web, dos fuentes daban valores distintos para la misma semana
(3,86 % y 4,27 %), otra razón para tomarla directamente del mercado al ejecutar.

## D14 — Diseño de la interfaz en dos fases
**Fecha:** 2026-10-23
**Decisión:** al abrir la app (tras la portada) solo se muestran las entradas, a pantalla completa y
agrupadas en tarjetas. Al ejecutar, aparecen los resultados y las entradas pasan a un panel lateral
que se abre con un botón dorado flotante. Tema negro y dorado (Cinzel para la marca, Inter para el texto).
**Por qué:** separa la configuración del análisis, deja todo el espacio a los resultados y mantiene
las entradas a un clic para recalcular. El panel usa la barra lateral propia de Streamlit (se abre
desde la izquierda), que es más estable que un panel construido a mano.

## D15 — Parámetros del GBM para proyectar desde hoy
**Fecha:** 2026-10-24
**Decisión:** la pestaña GBM calibra a y σ con los **últimos 504 retornos hasta hoy** (no con la
ventana de calibración inicial, que termina hace un año). S0 es el último cierre sin ajustar y el
drift es a = μ total − q (Paso 7). Se simulan 5.000 trayectorias por acción con Δt = 1/252; la
semilla de cada acción es la semilla de la app más la posición de la acción en la lista.
**Por qué:** se proyecta hacia adelante desde hoy, así que conviene usar la información más reciente
con la misma longitud de ventana. La ventana de calibración inicial se reserva para el backtest
(Paso 13), donde sí importa no usar datos posteriores a cada fecha. Sumar la posición a la semilla
evita que todas las acciones reciban los mismos números aleatorios, sin perder reproducibilidad.

---

_Plantilla para nuevas decisiones:_

```
## Dn — Título
**Fecha:**
**Responsable:**
**Decisión:**
**Por qué:**
**Alternativas consideradas:**
```
