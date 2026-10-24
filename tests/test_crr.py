"""Pruebas del árbol CRR (Paso 8)."""
import numpy as np
import pytest

from src import crr

CONTROL = dict(s0=100, k=100, t=1, r=0.05, q=0.0, sigma=0.20)


def test_control_del_instructivo_10_4506():
    v = crr.precio_crr(**CONTROL, n=1000, tipo="call", estilo="europea")
    assert v == pytest.approx(10.4506, abs=0.005)
    assert crr.black_scholes(**CONTROL) == pytest.approx(10.4506, abs=1e-4)


def test_converge_hacia_black_scholes():
    bs = crr.black_scholes(**CONTROL)
    err_10 = abs(crr.precio_crr(**CONTROL, n=10) - bs)
    err_500 = abs(crr.precio_crr(**CONTROL, n=500) - bs)
    assert err_500 < err_10 and err_500 < 0.01


def test_paridad_put_call_europea():
    c = crr.precio_crr(**CONTROL, n=400, tipo="call")
    p = crr.precio_crr(**CONTROL, n=400, tipo="put")
    s0, k, t, r, q = 100, 100, 1, 0.05, 0.0
    assert c - p == pytest.approx(s0 * np.exp(-q * t) - k * np.exp(-r * t), abs=1e-8)


def test_americana_vale_al_menos_la_europea():
    for tipo in ("call", "put"):
        e = crr.precio_crr(**CONTROL, n=300, tipo=tipo, estilo="europea")
        a = crr.precio_crr(**CONTROL, n=300, tipo=tipo, estilo="americana")
        assert a >= e - 1e-12
    # put americana: el ejercicio anticipado vale algo
    assert crr.precio_crr(**CONTROL, n=300, tipo="put", estilo="americana") > \
        crr.precio_crr(**CONTROL, n=300, tipo="put", estilo="europea") + 0.1


def test_call_americana_sin_dividendos_igual_a_europea():
    e = crr.precio_crr(**CONTROL, n=300, tipo="call", estilo="europea")
    a = crr.precio_crr(**CONTROL, n=300, tipo="call", estilo="americana")
    assert a == pytest.approx(e, abs=1e-10)


def test_t_cero_devuelve_el_payoff():
    assert crr.precio_crr(110, 100, 0, 0.05, 0, 0.2, 10, "call") == 10
    assert crr.precio_crr(110, 100, 0, 0.05, 0, 0.2, 10, "put") == 0


def test_probabilidad_neutral_valida():
    par = crr.parametros_crr(1, 3, 0.05, 0.0, 0.2)
    assert 0 < par.p < 1 and par.u * par.d == pytest.approx(1)
    with pytest.raises(ValueError):
        crr.parametros_crr(1, 1, 0.9, 0.0, 0.05)   # e^{(r−q)Δt} > u  ->  p^Q > 1


def test_arbol_de_tres_pasos_coincide_con_el_precio():
    arb = crr.arbol_crr(**CONTROL, n=3, tipo="call", estilo="europea")
    assert len(arb.precios) == 4 and len(arb.precios[3]) == 4
    assert arb.valores[0][0] == pytest.approx(crr.precio_crr(**CONTROL, n=3))
    assert arb.precios[2][1] == pytest.approx(100)   # sube y baja: vuelve a S0


def test_entradas_invalidas():
    with pytest.raises(ValueError):
        crr.precio_crr(-1, 100, 1, 0.05, 0, 0.2, 10)
    with pytest.raises(ValueError):
        crr.precio_crr(100, 100, 1, 0.05, 0, 0.2, 10, tipo="swap")
