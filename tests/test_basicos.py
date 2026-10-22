"""Pruebas automáticas. Ejecutar con:  pytest"""
from src import config


def test_universo_tiene_diez_acciones_y_cinco_sectores():
    assert len(config.TICKERS_SECTORES) == 10
    assert len(set(config.TICKERS_SECTORES.values())) >= 4


def test_asignacion_es_factible():
    # La suma de topes sectoriales debe permitir invertir el 100 %.
    n_sectores = len(set(config.TICKERS_SECTORES.values()))
    assert n_sectores * config.MAX_SECTOR >= 1
    assert 10 * config.W_MIN <= 1 <= 10 * config.W_MAX
