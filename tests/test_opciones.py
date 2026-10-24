"""Pruebas de contratos, probabilidades y comparación (Pasos 5, 6 y 9) con una cadena simulada."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src import opciones as op

HOY = date(2026, 10, 23)


def _cadena():
    filas = []
    for venc in ("2026-11-20", "2027-01-15", "2027-04-16"):
        for tipo in ("call", "put"):
            for k in np.arange(250, 460, 5.0):
                filas.append({"contractSymbol": f"AAPL{venc}{tipo[0]}{k}", "strike": k, "lastPrice": 5.0,
                              "bid": 4.9 if k != 350 else 0.0, "ask": 5.1, "volume": 10,
                              "openInterest": int(1000 - abs(k - 350) * 5), "impliedVolatility": 0.3,
                              "lastTradeDate": "2026-10-23", "tipo": tipo, "vencimiento": venc})
    return pd.DataFrame(filas)


def test_elegir_vencimientos_cercanos_a_1_3_6_meses():
    disp = ["2026-10-30", "2026-11-20", "2026-12-18", "2027-01-15", "2027-03-19", "2027-04-16", "2027-06-18"]
    assert op.elegir_vencimientos(disp, HOY) == ["2026-11-20", "2027-01-15", "2027-04-16"]


def test_nueve_candidatos_con_strikes_distintos_por_vencimiento():
    c = op.candidatos(_cadena(), "call", 347.7, 0.20, 0.28, HOY)
    assert len(c) == 9
    assert all(g["Strike"].nunique() == 3 for _, g in c.groupby("Vencimiento"))
    assert (c["T valoración"] == c["Días"] / 365).all()


def test_cotizacion_invalida_se_marca():
    assert not op.cotizacion_valida(0.0, 5.1)
    assert not op.cotizacion_valida(5.2, 5.1)
    assert op.cotizacion_valida(4.9, 5.1)


def test_sugerencia_elige_valido_y_liquido():
    c = op.candidatos(_cadena(), "call", 347.7, 0.20, 0.28, HOY)
    idx, motivo = op.sugerir(c)
    assert c.loc[idx, "Cotización válida"]
    assert "líquido" in motivo


def test_probabilidades_coinciden_con_la_formula_exacta():
    p = op.probabilidades(100, 0.08, 0.2, 105, 0.5, 0.5, 0.04, "call", prima=3.0, n=50_000)
    assert p["p_ejercicio"] == pytest.approx(p["p_ejercicio_exacta"], abs=0.01)
    assert p["p_beneficio"] < p["p_ejercicio"]          # ejercer no es lo mismo que ganar
    assert p["punto_equilibrio"] == pytest.approx(105 + 3 * np.exp(0.04 * 0.5))


def test_comparacion_con_el_mercado():
    r = op.comparar(10.2, 9.8, 10.4)
    assert r["valida"] and r["posicion"] == "dentro"
    assert r["delta"] == pytest.approx(10.2 - 10.1)
    assert not op.comparar(10.2, 0, 10.4)["valida"]
