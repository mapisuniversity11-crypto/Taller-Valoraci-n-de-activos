"""Pruebas del GBM (Paso 4)."""
import numpy as np
import pytest

from src import gbm


def test_forma_y_punto_de_partida():
    tr = gbm.simular_gbm(100, 0.08, 0.2, 21, n=300, semilla=1)
    assert tr.shape == (300, 22)
    assert np.all(tr[:, 0] == 100)


def test_precios_siempre_positivos():
    tr = gbm.simular_gbm(50, -0.5, 0.9, 252, n=2000, semilla=3)
    assert (tr > 0).all()


def test_semilla_fija_reproduce():
    a = gbm.simular_gbm(100, 0.1, 0.25, 63, n=500, semilla=42)
    b = gbm.simular_gbm(100, 0.1, 0.25, 63, n=500, semilla=42)
    assert np.array_equal(a, b)


def test_media_simulada_cerca_de_la_teorica():
    a, sigma = 0.10, 0.25
    tr = gbm.simular_gbm(100, a, sigma, 126, n=5000, semilla=7)
    r = gbm.resumir(tr, a)
    assert r.media_teorica == pytest.approx(100 * np.exp(a * 0.5))
    assert abs(r.z) < 3
    assert r.todos_positivos


def test_cuantil_teorico_coincide_con_la_simulacion():
    a, sigma, t = 0.08, 0.2, 0.5
    tr = gbm.simular_gbm(100, a, sigma, 126, n=20000, semilla=11)
    for beta in (0.25, 0.5, 0.75):
        empirico = np.percentile(tr[:, -1], beta * 100)
        assert empirico == pytest.approx(gbm.cuantil_gbm(100, a, sigma, t, beta), rel=0.01)


def test_sin_volatilidad_crece_exactamente_a_la_tasa_a():
    tr = gbm.simular_gbm(100, 0.05, 0.0, 252, n=3, semilla=0)
    assert tr[0, -1] == pytest.approx(100 * np.exp(0.05))


def test_entradas_invalidas():
    with pytest.raises(ValueError):
        gbm.simular_gbm(0, 0.1, 0.2, 10)
