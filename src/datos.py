"""Descarga, limpieza y validación de precios (Paso 2). Entrega 1.

Funciones previstas:
- descargar_precios(tickers, inicio, fin): ajustados (auto_adjust=True) y cierre sin ajustar. Recordar: `end` es exclusiva.
- limpiar(df): ordenar fechas, quitar duplicados, revisar precios <= 0 y faltantes; sesiones comunes completas, sin rellenar con ceros.
- reporte_calidad(df_crudo, df_limpio): filas descartadas por motivo, rango de fechas, n retornos.
- guardar_respaldo / cargar_csv: CSV en data/raw/ + metadatos (fuente, fecha y hora de descarga, versiones de bibliotecas).
"""
