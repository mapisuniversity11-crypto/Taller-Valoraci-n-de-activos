"""Pruebas de la evaluación rolling del GBM (Paso 13)."""
import numpy as np
import pandas as pd
import pytest

from src import rolling


def _precios(n=600, semilla=0):
    rng = np.random.default_rng(semilla)
    x = rng.normal(0.0004, 0.015, size=(n, 2))
    return pd.DataFrame(100 * np.exp(np.cumsum(x, axis=0)), index=pd.bdate_range("2023-01-02", periods=n),
                        columns=["A", "B"])


def test_pronostico_ec12_y_metricas_ec13():
    p = _precios()
    inicio = p.index[-252]
    det, res = rolling.rolling_gbm(p, inicio, h=1)
    d = det["A"]
    # Reestimación manual en la primera fecha de origen con los 252 log-retornos hasta t
    t = d.index[0]
    pos = p.index.get_loc(t)
    x = np.log(p["A"] / p["A"].shift(1)).iloc[pos - 251:pos + 1]
    sigma = np.sqrt(252) * x.std(ddof=1)
    mu = 252 * x.mean() + 0.5 * sigma ** 2
    assert d.loc[t, "Pronóstico GBM"] == pytest.approx(p["A"].iloc[pos] * np.exp(mu / 252))
    assert d.loc[t, "Observado"] == pytest.approx(p["A"].iloc[pos + 1])
    fila = res.set_index("Ticker").loc["A"]
    assert fila["m"] == 251                               # el último origen no tiene objetivo a H = 1
    assert fila["RMSE GBM"] == pytest.approx(np.sqrt((d["Error GBM"] ** 2).mean()))
    assert fila["MAPE ingenuo (%)"] == pytest.approx(100 * (d["Error ingenuo"].abs() / d["Observado"]).mean())


def test_horizonte_mayor_reduce_pares():
    p = _precios()
    _, r1 = rolling.rolling_gbm(p, p.index[-252], h=1)
    _, r5 = rolling.rolling_gbm(p, p.index[-252], h=5)
    assert (r5["m"] == r1["m"] - 4).all()


def test_cobertura_de_la_banda_cerca_del_80():
    p = _precios(n=1500, semilla=3)
    _, r = rolling.rolling_gbm(p, p.index[-1000], h=1)
    assert 0.72 < r["Cobertura P10–P90"].mean() < 0.88


def test_sin_datos_futuros():
    p = _precios()
    det, _ = rolling.rolling_gbm(p, p.index[-252], h=1)
    p2 = p.copy()
    p2.iloc[-50:] *= 3                                    # cambiar el futuro no altera pronósticos pasados
    det2, _ = rolling.rolling_gbm(p2, p.index[-252], h=1)
    t = det["A"].index[100]
    assert det["A"].loc[t, "Pronóstico GBM"] == pytest.approx(det2["A"].loc[t, "Pronóstico GBM"])
