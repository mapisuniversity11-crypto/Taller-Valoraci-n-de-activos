"""Árbol binomial Cox-Ross-Rubinstein (Paso 8).

    Δt = T/N,   u = e^{σ√Δt},   d = 1/u,   p^Q = (e^{(r−q)Δt} − d) / (u − d)      (Ec. 6)
    S_{k,j} = S0 · u^j · d^{k−j}            (k = paso, j = número de subidas)
    Vencimiento:  H_C(S) = (S − K)⁺,   H_P(S) = (K − S)⁺
    Retroceso:    V^cont_{k,j} = e^{−rΔt} [p^Q V_{k+1,j+1} + (1 − p^Q) V_{k+1,j}]  (Ec. 7)
    Europea:  V = V^cont.     Americana:  V = máx{H(S_{k,j}), V^cont}.

Tasas en decimal y continuas, tiempo en años.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

TIPOS = ("call", "put")
ESTILOS = ("europea", "americana")


def _validar(s0, k, t, r, q, sigma, n, tipo, estilo):
    if s0 <= 0 or k <= 0:
        raise ValueError("S0 y K deben ser positivos.")
    if t < 0:
        raise ValueError("T no puede ser negativo.")
    if sigma <= 0:
        raise ValueError("σ debe ser positiva.")
    if int(n) != n or n < 1:
        raise ValueError("N debe ser un entero mayor o igual a 1.")
    if tipo not in TIPOS:
        raise ValueError(f"tipo debe ser uno de {TIPOS}.")
    if estilo not in ESTILOS:
        raise ValueError(f"estilo debe ser uno de {ESTILOS}.")


def payoff(s, k: float, tipo: str):
    s = np.asarray(s, dtype=float)
    return np.maximum(s - k, 0.0) if tipo == "call" else np.maximum(k - s, 0.0)


@dataclass
class ParametrosCRR:
    dt: float
    u: float
    d: float
    p: float


def parametros_crr(t: float, n: int, r: float, q: float, sigma: float) -> ParametrosCRR:
    """Ec. 6. Lanza un error si p^Q no está entre 0 y 1 (el árbol no sería válido)."""
    dt = t / n
    u = float(np.exp(sigma * np.sqrt(dt)))
    d = 1 / u
    p = float((np.exp((r - q) * dt) - d) / (u - d))
    if not 0 < p < 1:
        raise ValueError(f"p^Q = {p:.4f} está fuera de (0, 1): aumente N o revise r, q y σ.")
    return ParametrosCRR(dt, u, d, p)


def precio_crr(s0: float, k: float, t: float, r: float, q: float, sigma: float, n: int,
               tipo: str = "call", estilo: str = "europea") -> float:
    """Prima de la opción por el árbol CRR (Ec. 6 y 7). Si T = 0, devuelve el payoff."""
    _validar(s0, k, t, r, q, sigma, n, tipo, estilo)
    if t == 0:
        return float(payoff(s0, k, tipo))
    par = parametros_crr(t, n, r, q, sigma)
    j = np.arange(n + 1)
    precios = s0 * par.u ** j * par.d ** (n - j)
    v = payoff(precios, k, tipo)
    desc = np.exp(-r * par.dt)
    for paso in range(n - 1, -1, -1):
        v = desc * (par.p * v[1:] + (1 - par.p) * v[:-1])
        if estilo == "americana":
            j = np.arange(paso + 1)
            v = np.maximum(v, payoff(s0 * par.u ** j * par.d ** (paso - j), k, tipo))
    return float(v[0])


@dataclass
class Arbol:
    parametros: ParametrosCRR
    precios: list[np.ndarray]      # precios[k][j]
    valores: list[np.ndarray]      # valores[k][j]
    ejercicio: list[np.ndarray]    # True si conviene ejercer en ese nodo (solo americana)


def arbol_crr(s0: float, k: float, t: float, r: float, q: float, sigma: float, n: int = 3,
              tipo: str = "call", estilo: str = "europea") -> Arbol:
    """Árbol completo nodo por nodo, para dibujarlo (se usa con N pequeño, p. ej. 3)."""
    _validar(s0, k, t, r, q, sigma, n, tipo, estilo)
    if t == 0:
        raise ValueError("Con T = 0 no hay árbol que dibujar: la prima es el payoff.")
    par = parametros_crr(t, n, r, q, sigma)
    precios = [s0 * par.u ** np.arange(m + 1) * par.d ** (m - np.arange(m + 1)) for m in range(n + 1)]
    valores = [None] * (n + 1)
    ejercicio = [None] * (n + 1)
    valores[n] = payoff(precios[n], k, tipo)
    ejercicio[n] = valores[n] > 0
    desc = np.exp(-r * par.dt)
    for m in range(n - 1, -1, -1):
        cont = desc * (par.p * valores[m + 1][1:] + (1 - par.p) * valores[m + 1][:-1])
        inmediato = payoff(precios[m], k, tipo)
        if estilo == "americana":
            valores[m] = np.maximum(cont, inmediato)
            ejercicio[m] = (inmediato > cont) & (inmediato > 0)
        else:
            valores[m] = cont
            ejercicio[m] = np.zeros_like(cont, dtype=bool)
    return Arbol(par, precios, valores, ejercicio)


def convergencia(s0, k, t, r, q, sigma, tipo="call", estilo="europea",
                 pasos=(1, 2, 3, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500)) -> list[tuple[int, float]]:
    """Prima para distintos N: al crecer N, el árbol se estabiliza."""
    return [(n, precio_crr(s0, k, t, r, q, sigma, n, tipo, estilo)) for n in pasos]


def black_scholes(s0: float, k: float, t: float, r: float, q: float, sigma: float, tipo: str = "call") -> float:
    """Precio exacto de la opción europea (referencia a la que converge el CRR)."""
    if t == 0:
        return float(payoff(s0, k, tipo))
    d1 = (np.log(s0 / k) + (r - q + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    if tipo == "call":
        return float(s0 * np.exp(-q * t) * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2))
    return float(k * np.exp(-r * t) * norm.cdf(-d2) - s0 * np.exp(-q * t) * norm.cdf(-d1))
