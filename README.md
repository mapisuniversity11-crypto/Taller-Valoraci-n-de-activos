# Quantia

**Valoración de opciones y riesgo de portafolio.** Taller de Valoración de Activos:
modelo binomial, GBM, opciones y asignación sectorial por VaR.
Universidad Externado de Colombia.

**Caso:** USD 100.000, diez acciones en USD, cinco sectores.

## Integrantes

| Nombre | Responsabilidad principal |
|---|---|
| María Paula Muñoz Malaver | Datos, calibración, integración de la app |
| _(compañera)_ | Árbol binomial CRR, _(por definir)_ |

## Cómo ejecutar

Requiere Python 3.10 o superior.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Pruebas:

```bash
pytest
```

## Estructura

```
app.py              Interfaz Streamlit (solo muestra; no calcula)
.streamlit/         Tema de la app (negro y dorado)
src/
  config.py         Valores por defecto (tickers, fechas, α, h, semilla...)
  ui.py             Elementos visuales: portada, tema negro y dorado, tarjetas
  assets/           Imagen de la franja superior
  datos.py          Paso 2  — descarga, limpieza, calidad, respaldo CSV
  calibracion.py    Paso 3  — retornos, σ, μ, covarianzas
  crr.py            Paso 8  — árbol binomial CRR
  gbm.py            Paso 4  — simulación GBM
  opciones.py       Pasos 5–7, 9 — contratos y comparación con mercado
  riesgo.py         Paso 10 — VaR / ES
  rolling.py        Paso 13 — evaluación rolling del GBM
  asignacion.py     Pasos 11–12 — asignación y estrategias
tests/              Pruebas automáticas (p. ej. control CRR = 10,4506)
data/raw/           Datos de respaldo descargados + metadatos
docs/
  decisiones.md     Bitácora de decisiones y su justificación
  plan.md           Plan de trabajo por entrega
```

## Entregas

| Semana | Contenido | Estado |
|---|---|---|
| 1 | Datos y CRR | Lista |
| 2 | GBM, contrato, rolling, VaR | Lista |
| 3 | Asignación y demostración | Asignación lista; falta exportación CSV y demostración |

## Reproducibilidad

- Semilla fija (configurable en la app).
- Datos de respaldo en `data/raw/` con fecha y hora de descarga.
- Versiones de bibliotecas registradas junto a cada descarga.

## Créditos

- La portada de cintas de metal líquido adapta el shader del componente **ShaderMetal**
  publicado en [21st.dev](https://21st.dev). _(Pendiente: confirmar autor y licencia en la
  página del componente.)_
- Fuentes: Cinzel e Inter (Google Fonts, licencia SIL Open Font License).
