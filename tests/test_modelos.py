"""Pruebas de los modelos adicionales: browniano estándar, Merton, browniano fraccional, Ornstein-Uhlenbeck /
Vasicek, herding y opciones reales."""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from src import crr, datos, gbm, modelos, riesgo


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


# ---------- Ornstein-Uhlenbeck · Vasicek ----------
def test_ou_recupera_los_parametros_simulados():
    x = modelos.simular_ou(0.03, 2.0, 0.04, 0.01, 2520, n=1, semilla=1)[0]
    c = modelos.calibrar_ou(pd.Series(x))
    assert c.theta == pytest.approx(0.04, abs=0.003)
    assert c.sigma == pytest.approx(0.01, rel=0.05)
    assert 0.8 < c.kappa < 4
    assert c.revierte and c.evidencia == "fuerte"
    assert c.vida_media == pytest.approx(np.log(2) / c.kappa)


def test_ou_sin_reversion_en_un_camino_al_azar():
    x = np.cumsum(np.random.default_rng(5).normal(0, 0.01, 1000))
    c = modelos.calibrar_ou(pd.Series(x))
    assert not c.revierte


def test_media_y_desviacion_ou_coinciden_con_la_simulacion():
    x0, k, th, s, t = 0.05, 1.5, 0.03, 0.02, 0.5
    tr = modelos.simular_ou(x0, k, th, s, 126, n=20_000, semilla=2)
    assert tr[:, -1].mean() == pytest.approx(modelos.media_ou(x0, k, th, t), abs=3e-4)
    assert tr[:, -1].std() == pytest.approx(modelos.desv_ou(k, s, t), rel=0.03)


def test_tasa_promedio_vasicek_frente_a_la_simulacion():
    r0, k, th, s, t = 0.03, 2.0, 0.04, 0.01, 0.5
    media, desv = modelos.tasa_promedio_vasicek(r0, k, th, s, t)
    prom = modelos.simular_ou(r0, k, th, s, 126, n=20_000, semilla=3)[:, 1:].mean(axis=1)
    assert prom.mean() == pytest.approx(media, abs=1e-4)
    assert prom.std() == pytest.approx(desv, rel=0.05)


def test_bono_vasicek_sin_volatilidad_y_con_tasa_constante():
    # Con σ = 0 y r0 = θ la tasa no se mueve: P = e^{−rT}
    assert modelos.bono_vasicek(0.04, 1.0, 0.04, 0.0, 2) == pytest.approx(np.exp(-0.08))
    curva = modelos.curva_vasicek(0.04, 1.0, 0.04, 0.0)
    assert np.allclose(curva["Tasa"], 0.04)


def test_irx_se_guarda_y_se_carga(tmp_path):
    serie = pd.Series([4.0, 4.1, 3.9], index=pd.bdate_range("2026-01-05", periods=3), name="IRX (%)")
    datos.guardar_irx(serie, tmp_path)
    leida = datos.cargar_irx(tmp_path)
    assert np.allclose(leida.to_numpy(), serie.to_numpy())
    assert datos.cargar_irx(tmp_path / "no_existe") is None
    assert datos.r_continua_desde_irx(leida).iloc[0] == pytest.approx(np.log(1.04))


# ---------- Herding ----------
def _cov_ejemplo():
    s = np.array([0.01, 0.02, 0.015])
    c = np.array([[1, 0.3, 0.1], [0.3, 1, 0.2], [0.1, 0.2, 1]])
    t = ["A", "B", "C"]
    return pd.DataFrame(np.outer(s, s) * c, index=t, columns=t), pd.Series([0.5, 0.3, 0.2], index=t)


def test_cov_manada_extremos():
    cov, _ = _cov_ejemplo()
    assert np.allclose(modelos.cov_manada(cov, 0), cov)
    s = np.sqrt(np.diag(cov))
    assert np.allclose(modelos.cov_manada(cov, 1), np.outer(s, s))
    c_h = modelos.cov_manada(cov, 0.5).to_numpy() / np.outer(s, s)
    assert c_h[0, 1] == pytest.approx(0.5 + 0.5 * 0.3)


def test_var_por_manada_crece_y_coincide_con_riesgo():
    cov, w = _cov_ejemplo()
    tabla = modelos.var_por_manada(cov, w, 100_000, 0.99, 10)
    assert tabla["VaR"].iloc[0] == pytest.approx(riesgo.var_normal(cov, w, 100_000, 0.99, 10)["var"])
    assert (np.diff(tabla["VaR"]) > 0).all()
    # con h = 1 la cartera es una sola acción: σ_p = Σ w_i σ_i
    s = np.sqrt(np.diag(cov))
    assert tabla["VaR"].iloc[-1] == pytest.approx(100_000 * norm.ppf(0.99) * np.sqrt(10) * (w.to_numpy() @ s))


def test_simulacion_de_manada_cerca_del_var_normal():
    cov, w = _cov_ejemplo()
    perd = modelos.simular_manada(cov, w, 100_000, 0.4, 10, n=100_000, semilla=1)
    var_mc, _ = riesgo.var_es(perd, 0.99)
    var_n = riesgo.var_normal(modelos.cov_manada(cov, 0.4), w, 100_000, 0.99, 10)["var"]
    assert var_mc == pytest.approx(var_n, rel=0.03)


def test_correlacion_movil_y_csad():
    rng = np.random.default_rng(4)
    m = rng.standard_normal(500)
    x = pd.DataFrame({f"T{i}": 0.6 * m + 0.8 * rng.standard_normal(500) for i in range(5)},
                     index=pd.bdate_range("2024-01-01", periods=500)) * 0.01
    rho = modelos.correlacion_promedio_movil(x, 63)
    assert len(rho) == 500 - 62
    assert rho.mean() == pytest.approx(0.36, abs=0.06)
    assert modelos.h_para_correlacion(0.2, 0.6) == pytest.approx(0.5)
    assert not modelos.csad(x).hay_manada      # datos normales sin manada


def test_csad_detecta_manada_sembrada():
    # Sin manada la dispersión crecería en línea recta con |R_m|; aquí se frena en los días extremos (cóncava)
    rng = np.random.default_rng(6)
    m = rng.standard_normal(2000) * 0.015
    propio = np.maximum(0.004 + 0.5 * np.abs(m) - 10 * m ** 2, 0.001)
    x = pd.DataFrame({f"T{i}": m + propio * rng.standard_normal(2000) for i in range(8)})
    c = modelos.csad(x)
    assert c.g2 < 0 and c.hay_manada


# ---------- Opciones reales ----------
ARGS_OR = dict(v0=100, sigma=0.3, r=0.04, delta=0.03, t=3, n=36)


def test_esperar_es_una_call_americana():
    o = modelos.opcion_real(**ARGS_OR, tipo="esperar", inversion=110)
    assert o.opcion == pytest.approx(crr.precio_crr(100, 110, 3, 0.04, 0.03, 0.3, 36, "call", "americana"))
    assert o.vpn == -10 and o.vpn_estrategico == o.opcion
    assert modelos.flexibilidad(o) == pytest.approx(o.opcion)


def test_abandonar_es_una_put_y_expandir_una_call_escalada():
    a = modelos.opcion_real(**ARGS_OR, tipo="abandonar", inversion=110, rescate=60)
    assert a.opcion == pytest.approx(crr.precio_crr(100, 60, 3, 0.04, 0.03, 0.3, 36, "put", "americana"))
    e = modelos.opcion_real(**ARGS_OR, tipo="expandir", inversion=110, expansion=0.5, costo_expansion=40)
    assert e.opcion == pytest.approx(0.5 * crr.precio_crr(100, 80, 3, 0.04, 0.03, 0.3, 36, "call", "americana"))
    assert e.vpn_estrategico == pytest.approx(-10 + e.opcion)


def test_las_dos_opciones_juntas():
    kw = dict(inversion=110, expansion=0.5, costo_expansion=40, rescate=60)
    e, a, j = (modelos.opcion_real(**ARGS_OR, tipo=t, **kw) for t in ("expandir", "abandonar", "expandir o abandonar"))
    assert max(e.opcion, a.opcion) <= j.opcion <= e.opcion + a.opcion + 1e-9
    assert set(np.concatenate(j.decision)) <= {"Seguir", "Expandir", "Abandonar"}


def test_flexibilidad_crece_con_la_volatilidad():
    tabla = modelos.sensibilidad_volatilidad(100, 0.04, 0.03, 3, 36, 110, 0.5, 40, 60).dropna()
    for tipo in modelos.TIPOS_OPCION_REAL:
        assert (np.diff(tabla[tipo]) >= -1e-9).all()


def test_opcion_real_entradas_invalidas():
    with pytest.raises(ValueError):
        modelos.opcion_real(**ARGS_OR, tipo="vender")
    with pytest.raises(ValueError):
        modelos.opcion_real(0, 0.3, 0.04, 0.03, 3, 36)
