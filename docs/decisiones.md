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
