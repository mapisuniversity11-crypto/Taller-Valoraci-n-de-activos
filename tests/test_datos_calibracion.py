"""Pruebas de los Pasos 2 y 3 con datos simulados (no requieren internet)."""
import numpy as np
import pandas as pd
import pytest

from src import calibracion as cal
from src import datos

TICKERS = ["A", "B", "C"]


def _precios_sinteticos(n=800, semilla=0):
    rng = np.random.default_rng(semilla)
    fechas = pd.bdate_range("2022-01-03", periods=n)
    x = rng.normal(0.0004, 0.015, size=(n, len(TICKERS)))
    return pd.DataFrame(100 * np.exp(np.cumsum(x, axis=0)), index=fechas, columns=TICKERS)


def _crudos(aj):
    return datos.DatosCrudos(aj, aj * 1.01, {"fuente": "prueba"})


def test_limpieza_descarta_fechas_incompletas_sin_inventar_ceros():
    aj = _precios_sinteticos()
    aj.iloc[10, 0] = np.nan       # faltante
    aj.iloc[20, 1] = -5.0         # precio no positivo
    aj = pd.concat([aj, aj.iloc[[30]]])  # fecha duplicada
    limpio = datos.limpiar(_crudos(aj), TICKERS)

    assert limpio.resumen["duplicados_eliminados"] == 1
    assert limpio.resumen["sesiones_descartadas"] == 2
    assert limpio.ajustados.notna().all().all()
    assert limpio.ajustados.index.is_monotonic_increasing
    assert limpio.calidad_por_ticker.loc["B", "Precios ≤ 0"] == 1
    # no hay retornos exactamente cero inventados
    assert (cal.log_retornos(limpio.ajustados) == 0).sum().sum() == 0


def test_ventanas_504_y_252():
    aj = _precios_sinteticos(n=800)
    v = datos.separar_ventanas(aj)
    assert len(v["calibracion"]) - 1 == 504
    assert len(v["evaluacion"]) - 1 == 252
    # comparten el precio frontera, sin solaparse en retornos
    assert v["calibracion"].index[-1] == v["evaluacion"].index[0]
    assert v["evaluacion"].index[-1] == aj.index[-1]


def test_ventanas_insuficientes_avisa():
    with pytest.raises(ValueError, match="Amplíen"):
        datos.separar_ventanas(_precios_sinteticos(n=500))


def test_calibracion_formulas_ec2():
    aj = _precios_sinteticos()
    p = cal.calibrar(aj, q={"A": 0.02, "B": 0.0, "C": 0.01})
    x = np.log(aj / aj.shift(1)).dropna()
    s = x["A"].std(ddof=1)
    assert p.loc["A", "σ anual"] == pytest.approx(np.sqrt(252) * s)
    assert p.loc["A", "μ total anual"] == pytest.approx(252 * x["A"].mean() + 0.5 * 252 * s**2)
    assert p.loc["A", "a = μ − q"] == pytest.approx(p.loc["A", "μ total anual"] - 0.02)


def test_divisor_n_menos_1():
    precios = pd.DataFrame({"A": [100.0, 110.0, 99.0, 105.0]})
    r = cal.retornos_simples(precios)["A"].to_numpy()
    esperado = ((r - r.mean()) ** 2).sum() / (len(r) - 1)
    assert cal.covarianza_diaria(precios).loc["A", "A"] == pytest.approx(esperado)


def test_respaldo_ida_y_vuelta(tmp_path):
    aj = _precios_sinteticos(n=50)
    crudos = _crudos(aj)
    datos.guardar_respaldo(crudos, tmp_path)
    leido = datos.cargar_respaldo(tmp_path)
    pd.testing.assert_frame_equal(leido.ajustados, aj, check_freq=False, check_names=False)
    assert leido.meta["fuente"] == "prueba"


def test_q_desde_dividendos():
    # 4 USD de dividendos sobre un precio de 100 -> y = 4 %, q = ln(1.04)
    assert datos.q_desde_dividendos(4.0, 100.0) == pytest.approx(np.log(1.04))
    assert datos.q_desde_dividendos(0.0, 50.0) == 0.0
    with pytest.raises(ValueError):
        datos.q_desde_dividendos(1.0, 0.0)
