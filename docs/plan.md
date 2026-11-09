# Plan de trabajo y estado

## Entrega 1 — Datos y CRR · lista
- [x] Estructura del repositorio, README y bitácora de decisiones
- [x] Universo: 10 acciones en 5 sectores (D3)
- [x] Descarga de Yahoo (ajustado y sin ajustar), limpieza, calidad y respaldo en `data/raw`
- [x] Calibración: retornos, σ, μ total, a = μ − q, covarianza y correlación (n − 1)
- [x] Árbol binomial CRR europeo y americano; prueba de control = 10,4506
- [x] Pestaña Datos y calculadora CRR

## Entrega 2 — GBM, contrato, rolling y VaR · lista
- [x] GBM: trayectorias, cuantiles P10–P50–P90 y probabilidades (Paso 4)
- [x] Cadena de opciones real, contratos candidatos y elección (Pasos 5 a 7)
- [x] Valoración CRR del contrato frente al mercado (Paso 9)
- [x] VaR y ES histórico, normal y Monte Carlo; excepciones, fuera de muestra y estrés (Paso 10)
- [x] Evaluación rolling del GBM: RMSE y MAPE frente al pronóstico ingenuo (Paso 13)

## Entrega 3 — Asignación y demostración
- [x] Carteras de mínimo VaR, inverso al VaR y corrección por sector; orden de compra (Paso 11)
- [x] Estrategias en el último año y presupuesto de riesgo (Paso 12)
- [x] Exportación de resultados en CSV con resumen de parámetros y semilla (D24)
- [x] Pestaña Modelos: browniano estándar, Merton y browniano fraccional (D23)
- [x] Respaldo automático si Yahoo falla, versiones fijas y cálculos en memoria (D25, D26)
- [x] Pestaña Modelos: Ornstein-Uhlenbeck / Vasicek, herding y opciones reales (Danna Caballero, D27–D29)
- [ ] Desplegar en Streamlit Community Cloud y probar el enlace
- [ ] Actualizar el respaldo `data/raw` el día anterior a la sustentación (incluye `tasa_irx.csv` para Vasicek)
- [ ] Preparar la sustentación: cada integrante explica una función y una decisión
- [ ] Etiqueta final en git

## Puntos de evaluación (/100)
| Componente | Puntos | Dónde está en la app |
|---|---|---|
| Datos | 15 | Datos |
| Opciones / mercado | 15 | Opciones |
| GBM, rolling y RMSE/MAPE | 20 | GBM y Evaluación |
| VaR / ES | 15 | Riesgo |
| Asignación / estrategias | 20 | Asignación |
| Funcionamiento y sustentación | 15 | Toda la app |
