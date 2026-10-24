"""Pruebas de los resúmenes de la pestaña Datos (src/lecturas.py)."""
import numpy as np
import pandas as pd

from src import lecturas as L

SECT = {"A": "S1", "B": "S1", "C": "S2", "D": "S2"}


def _corr():
    m = np.array([[1, .9, .1, .2], [.9, 1, .0, .1], [.1, .0, 1, .8], [.2, .1, .8, 1]])
    return pd.DataFrame(m, index=list("ABCD"), columns=list("ABCD"))


def test_orden_por_sector_agrupa():
    assert L.orden_por_sector(["A", "C", "B", "D"], SECT) == ["A", "B", "C", "D"]


def test_pares_mas_y_menos_correlacionados():
    p = L.pares_correlacion(_corr(), SECT, n=1)
    assert p.iloc[0]["Par"] == "A – B" and p.iloc[1]["Par"] == "B – C"


def test_correlacion_media_por_grupo():
    mismo, distinto = L.correlacion_media_por_grupo(_corr(), SECT)
    assert abs(mismo - 0.85) < 1e-12 and abs(distinto - 0.1) < 1e-12


def test_dias_extremos():
    fechas = pd.bdate_range("2024-01-01", periods=200)
    x = pd.DataFrame({"A": np.zeros(200)}, index=fechas)
    x.iloc[::2, 0] = 0.01
    x.iloc[::2, 0] *= 1
    x.iloc[50, 0] = 0.5          # un día extremo evidente
    r = L.resumen_distribucion(x)
    assert r.loc["A", "Días extremos (>3σ)"] == 1
    assert r.loc["A", "Fecha mejor día"] == fechas[50].date()
