"""Exportación de resultados (entregable del taller): un ZIP con un CSV por resultado y un
resumen de parámetros y semillas para reproducir la ejecución.

Los CSV usan coma como separador y punto decimal (formato estándar, se abre en Excel con
"Datos > Desde texto/CSV" y en Python con pandas.read_csv).
"""
from __future__ import annotations

import io
import zipfile
from datetime import datetime

import pandas as pd


def resumen_a_tabla(resumen: dict) -> pd.DataFrame:
    """dict {sección: {parámetro: valor}} -> tabla Sección · Parámetro · Valor."""
    filas = [{"Sección": sec, "Parámetro": k, "Valor": v}
             for sec, items in resumen.items() for k, v in items.items()]
    return pd.DataFrame(filas, columns=["Sección", "Parámetro", "Valor"])


def armar_zip(tablas: dict[str, pd.DataFrame | None], resumen: dict) -> bytes:
    """Empaqueta las tablas no vacías como CSV y agrega 00_resumen_parametros_y_semilla.csv
    y un LEEME.txt con la lista de archivos."""
    buf = io.BytesIO()
    incluidos = []
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("00_resumen_parametros_y_semilla.csv",
                   resumen_a_tabla(resumen).to_csv(index=False).encode("utf-8-sig"))
        incluidos.append("00_resumen_parametros_y_semilla.csv")
        for nombre, tabla in tablas.items():
            if tabla is None or len(tabla) == 0:
                continue
            archivo = f"{nombre}.csv"
            z.writestr(archivo, tabla.to_csv().encode("utf-8-sig"))
            incluidos.append(archivo)
        leeme = (f"Quantia · resultados exportados el {datetime.now():%Y-%m-%d %H:%M}\n\n"
                 "Archivos:\n" + "\n".join(f"  - {a}" for a in incluidos) +
                 "\n\nPara reproducir: use los mismos datos (data/raw), las mismas entradas y la semilla "
                 "indicadas en 00_resumen_parametros_y_semilla.csv.\n")
        z.writestr("LEEME.txt", leeme.encode("utf-8"))
    return buf.getvalue()
