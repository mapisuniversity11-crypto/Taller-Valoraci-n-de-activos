"""Pruebas de VaR, ES, excepciones, fuera de muestra y estrés (Paso 10)."""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from src import riesgo as rg

T = ["A", "B", "C"]


def _precios(n=800, semilla=0):
    rng = np.random.default_rng(semilla)
    cov = np.array([[1, .5, .2], [.5, 1, .3], [.2, .3, 1]]) * 0.012 ** 2
    x = rng.multivariate_normal([0.0003] * 3, cov, size=n)
    return pd.DataFrame(100 * np.exp(np.cumsum(x, axis=0)), index=pd.bdate_range("2022-01-03", periods=n), columns=T)


def test_var_es_convencion():
    L = np.arange(1, 101, dtype=float)          # pérdidas 1..100
    var, es = rg.var_es(L, 0.99)
    assert var == 100 and es == 100
    var, es = rg.var_es(L, 0.95)
    assert var == 96 and es == pytest.approx(np.mean([96, 97, 98, 99, 100]))


def test_var_normal_formula_ec9():
    p = _precios()
    r = (p / p.shift(1) - 1).dropna()
    w = rg.pesos_iguales(T)
    res = rg.var_normal(r.cov(), w, 100_000, 0.99, 10)
    esperado = 100_000 * norm.ppf(0.99) * np.sqrt(10 * w @ r.cov().to_numpy() @ w)
    assert res["var"] == pytest.approx(esperado)
    assert res["es"] > res["var"]


def test_tres_metodos_dan_ordenes_de_magnitud_parecidos():
    p = _precios()
    w = rg.pesos_iguales(T)
    r = (p / p.shift(1) - 1).dropna()
    h = rg.var_historico(p, w, 100_000, 0.99, 10)["var"]
    n = rg.var_normal(r.cov(), w, 100_000, 0.99, 10)["var"]
    m = rg.var_montecarlo(p, w, 100_000, 0.99, 10)["var"]
    assert 0.6 < h / n < 1.6 and 0.7 < m / n < 1.3


def test_bloques_historicos_solapados():
    p = _precios(n=100)
    L = rg.perdidas_historicas(p, rg.pesos_iguales(T), 1.0, 10)
    assert len(L) == 90                          # un bloque por cada día de inicio posible


def test_excepciones_sin_datos_futuros():
    p = _precios(n=600)
    df = rg.excepciones(p, rg.pesos_iguales(T), 100_000, 0.99, p.index[-252])
    assert len(df) == 252
    assert df["VaR diario"].gt(0).all()
    assert rg.zona_basilea(2, 250, 0.99) == "verde" and rg.zona_basilea(12, 250, 0.99) == "roja"


def test_fuera_de_muestra_con_costos():
    p = _precios(n=50)
    f = rg.fuera_de_muestra(p, rg.pesos_iguales(T), 100_000, 0.001)
    assert f.valor.iloc[0] == pytest.approx(100_000 * 0.999)
    assert f.caida.max() <= 1e-12 and f.caida_maxima <= 0


def test_estres():
    p = _precios()
    r = (p / p.shift(1) - 1).dropna()
    w = pd.Series([0.5, 0.3, 0.2], index=T)
    e, info = rg.estres(w, {"A": "S1", "B": "S1", "C": "S2"}, r.cov(), 100_000, 0.99, 10)
    assert info["sector_mayor"] == "S1" and not info["empate"]
    assert e.loc[0, "Pérdida"] == pytest.approx(10_000)
    assert e.loc[1, "Pérdida"] == pytest.approx(100_000 * (0.2 * 0.8 + 0.05 * 0.2))   # S1 pesa 80 %
    normal = rg.var_normal(r.cov(), w, 100_000, 0.99, 10)["var"]
    assert e.loc[2, "Pérdida"] > normal
