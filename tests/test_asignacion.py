"""Pruebas de asignación y estrategias (Pasos 11 y 12)."""
import numpy as np
import pandas as pd
import pytest

from src import asignacion as asg

T = ["A1", "A2", "B1", "B2", "C1", "C2", "D1", "D2", "E1", "E2"]
SECT = {t: t[0] for t in T}


def _cov(semilla=0):
    rng = np.random.default_rng(semilla)
    vol = np.linspace(0.009, 0.02, len(T))
    corr = np.full((len(T), len(T)), 0.25)
    for i in range(0, len(T), 2):
        corr[i, i + 1] = corr[i + 1, i] = 0.75
    np.fill_diagonal(corr, 1)
    return pd.DataFrame(np.outer(vol, vol) * corr, index=T, columns=T)


def test_minimo_var_cumple_restricciones_y_reduce_riesgo():
    cov = _cov()
    res = asg.minimo_var(cov, SECT)
    assert res.exito
    assert all(ok for ok, _ in asg.verificar(res.pesos, SECT, 0.02, 0.20, 0.35))
    v_min = asg.var_cartera(res.pesos, cov, 1e5, 0.99, 10)
    for w in (asg.pesos_iguales(T), asg.pesos_inverso_var(cov, 1e5, 0.99, 10)):
        assert v_min <= asg.var_cartera(w, cov, 1e5, 0.99, 10) + 1e-6


def test_inverso_var_da_mas_peso_a_la_menos_volatil():
    w = asg.pesos_inverso_var(_cov(), 1e5, 0.99, 10)
    assert w.idxmax() == "A1" and w.idxmin() == "E2" and w.sum() == pytest.approx(1)


def test_factibilidad_detecta_restricciones_imposibles():
    assert asg.factibilidad(T, SECT, 0.02, 0.20, 0.35) == []
    assert asg.factibilidad(T, SECT, 0.02, 0.08, 0.35)               # 10 × 8 % < 100 %
    res = asg.minimo_var(_cov(), SECT, 0.02, 0.08, 0.35)
    assert not res.exito and res.problemas


def test_corregir_sectores_respeta_el_tope():
    w = pd.Series([0.2, 0.2, 0.15, 0.15, 0.06, 0.06, 0.06, 0.06, 0.03, 0.03], index=T)   # A pesa 40 %
    corr, pasos = asg.corregir_sectores(w, SECT)
    assert asg.peso_por_sector(corr, SECT).max() <= 0.35 + 1e-9
    assert corr.sum() == pytest.approx(1) and pasos


def test_acciones_enteras_y_efectivo():
    w = pd.Series(0.1, index=T)
    precios = pd.Series(33.0, index=T)
    t = asg.acciones_enteras(w, 100_000, precios)
    assert (t["Acciones"] == 303).all()                               # 10.000 / 33 = 303,03
    assert 100_000 - t["Monto invertido"].sum() == pytest.approx(10 * (10_000 - 303 * 33))


def test_presupuesto_de_riesgo_ec11():
    cov = _cov()
    w = asg.pesos_iguales(T)
    v = asg.var_cartera(w, cov, 1e5, 0.99, 10)
    p = asg.presupuesto_riesgo(w, cov, 1e5, 0.99, 10, v / 2)
    assert p.f == pytest.approx(0.5) and p.efectivo == pytest.approx(5e4)
    assert p.var_ajustado == pytest.approx(v / 2)
    assert asg.presupuesto_riesgo(w, cov, 1e5, 0.99, 10, v * 3).f == 1


def test_backtest_sin_errores():
    rng = np.random.default_rng(1)
    chol = np.linalg.cholesky(_cov().to_numpy())
    x = rng.standard_normal((800, len(T))) @ chol.T
    p = pd.DataFrame(100 * np.exp(np.cumsum(x, axis=0)), index=pd.bdate_range("2023-01-02", periods=800), columns=T)
    df, info = asg.backtest(p, p.index[-252], {"Iguales": asg.pesos_iguales(T)}, SECT, 1e5, 0.001)
    assert len(df) == 252 and info["Mínimo VaR · rebalanceo mensual"]["rebalanceos"] >= 12
    assert df.iloc[0, 0] == pytest.approx(1e5 * 0.999)
