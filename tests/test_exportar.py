"""Pruebas de la exportación de resultados (CSV en ZIP + resumen de parámetros y semilla)."""
import io
import zipfile

import pandas as pd

from src import exportar


def test_zip_con_resumen_y_tablas():
    tablas = {"01_a": pd.DataFrame({"x": [1.5, 2.0]}, index=["A", "B"]), "02_vacia": pd.DataFrame(), "03_nada": None}
    resumen = {"Semillas": {"Semilla de la app": 42}, "Cartera": {"Capital": 100000}}
    z = zipfile.ZipFile(io.BytesIO(exportar.armar_zip(tablas, resumen)))
    assert sorted(z.namelist()) == ["00_resumen_parametros_y_semilla.csv", "01_a.csv", "LEEME.txt"]
    r = pd.read_csv(z.open("00_resumen_parametros_y_semilla.csv"), encoding="utf-8-sig")
    assert list(r.columns) == ["Sección", "Parámetro", "Valor"]
    assert str(r.set_index("Parámetro").loc["Semilla de la app", "Valor"]) == "42"
    a = pd.read_csv(z.open("01_a.csv"), index_col=0, encoding="utf-8-sig")
    assert a.loc["B", "x"] == 2.0
