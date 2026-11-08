"""Pruebas de los modelos adicionales: browniano estándar, Merton y browniano fraccional."""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from src import crr, gbm, modelos


# ---------- Browniano estándar ----------
def test_browniano_usa_los_mismos_choques_que_el_gbm():
    b = modelos.simular_browniano(100, 5, 20, 10, n=3, semilla=7)
    g = gbm.simular_gbm(100, 0.05, 0.2, 10, n=3, semilla=7)
    z = np.random.default_rng(7).standard_normal((3, 10))
    assert np.allclose(np.diff(b, axis=1), 5 / 252 + 20 * np.sqrt(1 / 252) * z)
    assert np.allclose(np.diff(np.log(g), axis=1), (0.05 - 0.02) / 252 + 0.2 * np.sqrt(1 / 252) * z)


def test_prob_negativo_coincide_con_la_simulacion():
    p = modelos.prob_negativo(100, 0, 60, 4)
    assert p == pytest.approx(norm.cdf(-100 / 120))
    tr = modelos.simular_browniano(100, 0, 60, 4 * 252, n=20_000, semilla=1)
    assert (tr[:, -1] < 0).mean() == pytest.approx(p, abs=0.01)


def test_calibracion_browniano():
    precios = pd.Series(100 + np.arange(253) * 0.1)
    c = modelos.calibrar_browniano(precios)
    assert c.m == pytest.approx(25.2) and c.s == pytest.approx(0, abs=1e-9)


# ---------- Merton ----------
def test_merton_sin_saltos_es_black_scholes():
    assert modelos.precio_merton(100, 95, 0.5, 0.04, 0.01, 0.25, 0, 0, 0, "put") == pytest.approx(
        crr.black_scholes(100, 95, 0.5, 0.04, 0.01, 0.25, "put"))


def test_merton_frente_a_montecarlo_y_paridad():
    s0, k, t, r, q, sig, lam, mj, dj = 100, 100, 0.5, 0.04, 0.01, 0.2, 2.0, -0.05, 0.1
    call = modelos.precio_merton(s0, k, t, r, q, sig, lam, mj, dj, "call")
    put = modelos.precio_merton(s0, k, t, r, q, sig, lam, mj, dj, "put")
    assert call - put == pytest.approx(s0 * np.exp(-q * t) - k * np.exp(-r * t), abs=1e-8)
    rng = np.random.default_rng(1)
    n = 400_000
    kap = np.exp(mj + 0.5 * dj ** 2) - 1
    saltos = rng.poisson(lam * t, n)
    j = mj * saltos + dj * np.sqrt(saltos) * rng.standard_normal(n)
    st_ = s0 * np.exp((r - q - lam * kap - 0.5 * sig ** 2) * t + sig * np.sqrt(t) * rng.standard_normal(n) + j)
    assert call == pytest.approx(np.exp(-r * t) * np.maximum(st_ - k, 0).mean(), rel=0.01)


def test_detecta_saltos_sembrados():
    x = pd.Series(np.random.default_rng(3).normal(0, 0.01, 504), index=pd.bdate_range("2024-01-01", periods=504))
    x.iloc[[50, 200, 400]] = [0.08, -0.09, 0.07]
    m = modelos.calibrar_merton(x, k=5)
    assert set(m.fechas) == set(x.index[[50, 200, 400]])
    assert m.lam == pytest.approx(3 / 2)
    assert m.sigma == pytest.approx(0.01 * np.sqrt(252), rel=0.1)


# ---------- Browniano fraccional ----------
def test_hurst_sin_memoria_cerca_de_medio():
    x = np.random.default_rng(0).standard_normal(1000)
    h = modelos.hurst(x)
    lo, hi = modelos.intervalo_sin_memoria(x, 100)
    assert lo < h < hi and 0.4 < h < 0.6


def test_hurst_detecta_tendencia():
    camino = modelos.simular_fbm(0.75, 1000, 1, semilla=2)[0]
    h = modelos.hurst(np.diff(camino))
    assert h > modelos.intervalo_sin_memoria(np.diff(camino), 100)[1]


def test_fbm_varianza_anual_y_factor_var():
    assert modelos.simular_fbm(0.7, 252, 3000, 1)[:, -1].var() == pytest.approx(1, rel=0.1)
    assert modelos.factor_var_hurst(10, 0.5) == pytest.approx(1)
    assert modelos.factor_var_hurst(10, 0.6) > 1 > modelos.factor_var_hurst(10, 0.4)
