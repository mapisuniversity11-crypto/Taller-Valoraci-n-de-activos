# Plan de trabajo

## Entrega 1 — Datos y CRR (viernes 23 de octubre de 2026, 12:00 m.)

### Miércoles 21 (noche)
- [x] Estructura del repositorio, README, bitácora de decisiones
- [x] Elegir tickers y sectores (D3)
- [x] Esqueleto de la app con entradas, pestañas y botón (Paso 1)
- [ ] Subir a GitHub

### Jueves 22
- [ ] `datos.py`: descarga ajustada + sin ajustar, `end` exclusiva, metadatos y versiones
- [ ] `datos.py`: limpieza, sesiones comunes, reporte de calidad, respaldo/carga CSV
- [ ] `calibracion.py`: retornos, σ, μ total, a = μ − q, covarianza y correlación (n − 1)
- [ ] `crr.py`: europea y americana, call y put, validación de pQ, T = 0
- [ ] Pruebas: control CRR = 10,4506; divisor n − 1
- [ ] Pestaña Datos: calidad, precios normalizados, histogramas, correlaciones, parámetros
- [ ] Pestaña Opciones: árbol de 3 pasos y gráfico de convergencia

### Viernes 23 (mañana)
- [ ] Prueba completa de punta a punta (incluye carga desde CSV si falla Yahoo)
- [ ] Actualizar README y bitácora
- [ ] Etiqueta `entrega-1` en git y entrega

## Entrega 2 — GBM, contrato, rolling, VaR
Pasos 4, 5, 6, 7, 9, 10 y 13.

## Entrega 3 — Asignación y demostración
Pasos 11 y 12, exportación CSV y resumen de parámetros y semilla, preparación de la sustentación.

## Puntos de evaluación (/100)
| Componente | Puntos |
|---|---|
| Datos | 15 |
| Opciones / mercado | 15 |
| GBM, rolling y RMSE/MAPE | 20 |
| VaR / ES | 15 |
| Asignación / estrategias | 20 |
| Funcionamiento y sustentación | 15 |
