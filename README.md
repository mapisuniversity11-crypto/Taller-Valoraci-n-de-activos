# Quantia

**Valoración de opciones y riesgo de portafolio.** Taller de Valoración de Activos:
modelo binomial, GBM, opciones y asignación sectorial por VaR.
Universidad Externado de Colombia.

**Caso:** USD 100.000, diez acciones en USD, cinco sectores.

## Integrantes

| Nombre | Responsabilidad principal |
|---|---|
| María Paula Muñoz Malaver | Datos, calibración, integración de la app; modelos browniano estándar, Merton y browniano fraccional; exportación CSV |
| Danna Caballero | Árbol binomial CRR; modelos Ornstein-Uhlenbeck / Vasicek, herding y opciones reales |

## Cómo ejecutar

Requiere Python 3.10 o superior.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements-dev.txt   # app + pytest (para desplegar basta requirements.txt)
streamlit run app.py
```

Pruebas:

```bash
pytest
```

## Despliegue (Streamlit Community Cloud)

1. En share.streamlit.io, **Create app** → este repositorio, rama `main`, archivo `app.py`.
2. **Advanced settings → Python version → 3.13** (las versiones de `requirements.txt` están fijadas para 3.13).
3. **Deploy.** Cada `git push` actualiza la app publicada.

Si Yahoo Finance no responde desde el servidor, la app usa sola el respaldo de `data/raw` y lo avisa (D25).
Conviene actualizar ese respaldo el día anterior a la sustentación: correr la app localmente con
"Guardar descarga como respaldo" marcado y subir `data/raw`.

## Estructura

```
app.py              Portada, panel de entradas, ejecución y orden de las pestañas
vistas/             Un archivo por pestaña: datos, gbm, opciones, riesgo, asignacion,
                    evaluacion, modelos; exportacion (botón CSV); memoria (cálculos en caché)
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
  exportar.py       Exportación de resultados en CSV (ZIP) con resumen de parámetros y semilla
  modelos.py        Pestaña Modelos — browniano estándar, Merton, browniano fraccional
tests/              Pruebas automáticas (p. ej. control CRR = 10,4506)
data/raw/           Datos de respaldo descargados + metadatos
docs/
  decisiones.md     Bitácora de decisiones y su justificación
  plan.md           Plan de trabajo por entrega
  guia_modelos.md   Cómo agregar un modelo a la pestaña Modelos
requirements.txt    Versiones exactas para ejecutar y desplegar
requirements-dev.txt  Lo anterior + pytest
```

## Entregas

| Semana | Contenido | Estado |
|---|---|---|
| 1 | Datos y CRR | Lista |
| 2 | GBM, contrato, rolling, VaR | Lista |
| 3 | Asignación y demostración | Asignación y exportación CSV listas; falta demostración |

## Reproducibilidad

- Semilla fija (configurable en la app).
- Datos de respaldo en `data/raw/` con fecha y hora de descarga.
- Versiones de bibliotecas registradas junto a cada descarga.
- Botón **Exportar resultados (CSV)**: un ZIP con un CSV por resultado y `00_resumen_parametros_y_semilla.csv`.

## Créditos

- La portada de cintas de metal líquido adapta el shader del componente **Shader Metal** de jay sharma
  ([@radiumcoders en 21st.dev](https://21st.dev/@radiumcoders/components/shader-metal.md)), publicado con licencia
  MIT. El diseño de la portada (título, animaciones, colores y velo) es propio.
- Fuentes: Cinzel e Inter (Google Fonts, licencia SIL Open Font License).
