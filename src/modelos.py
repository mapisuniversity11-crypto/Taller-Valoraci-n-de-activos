"""Modelos adicionales de la pestaña Modelos.

Cada modelo relaja un supuesto del GBM y se aplica a los datos del caso:

* Browniano estándar (aritmético): P_t = P_0 + m·t + s·W_t. Los movimientos se miden en
  dólares, no en porcentaje; sirve para mostrar por qué el GBM usa el logaritmo.
* Merton (difusión con saltos): el log-precio sigue un GBM más saltos que llegan como un
  proceso de Poisson de intensidad λ, con tamaño normal N(μ_J, δ_J²) en logaritmo.
* Browniano fraccional: el exponente de Hurst H mide si los retornos tienen memoria
  (H = 0,5 sin memoria, H > 0,5 tendencia, H < 0,5 reversión).
* Ornstein-Uhlenbeck · Vasicek: dx = κ(θ − x)dt + σ dW. La variable vuelve a un nivel de largo
  plazo θ con velocidad κ. Se aplica a la tasa del Tesoro (Vasicek) y a la diferencia de
  log-precios de dos acciones del mismo sector.
* Herding (comportamiento en manada): cuando los inversionistas se copian, las acciones se mueven
  juntas. Se mide con la correlación promedio en el tiempo y con la dispersión CSAD, y se simula
  su efecto en el VaR.
* Opciones reales: la flexibilidad de esperar, expandir o abandonar un proyecto se valora como una
  opción americana sobre el valor del proyecto, con el mismo árbol CRR del Paso 8.

Unidades: tasas y volatilidades anuales en decimal, tiempo en años, D = 252 sesiones.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm, poisson

from .crr import black_scholes, parametros_crr

D = 252


# ---------------------------------------------------------------------------
# Browniano estándar (aritmético)
# ---------------------------------------------------------------------------
@dataclass
class BrownianoCalibrado:
    m: float   # drift en USD por año
    s: float   # volatilidad en USD por año (√252 · desviación de ΔP diario)


def calibrar_browniano(precios: pd.Series, d: int = D) -> BrownianoCalibrado:
    """Calibra con los cambios diarios en dólares ΔP_t = P_t − P_{t−1} (divisor n − 1)."""
    dp = precios.diff().dropna()
    return BrownianoCalibrado(m=float(d * dp.mean()), s=float(np.sqrt(d) * dp.std(ddof=1)))


def simular_browniano(p0: float, m: float, s: float, sesiones: int, n: int = 5_000,
                      semilla: int = 42, dt: float = 1 / D) -> np.ndarray:
    """Matriz (n, sesiones + 1): P_{t+Δt} = P_t + m·Δt + s·√Δt·Z. Usa la misma Z que gbm.simular_gbm
    con la misma semilla, para que la comparación sea con los mismos choques."""
    rng = np.random.default_rng(semilla)
    z = rng.standard_normal((n, sesiones))
    pasos = m * dt + s * np.sqrt(dt) * z
    return np.hstack([np.full((n, 1), p0), p0 + np.cumsum(pasos, axis=1)])


def prob_negativo(p0: float, m: float, s: float, t: float) -> float:
    """P(P_T < 0) del browniano aritmético = Φ((−P0 − m·T) / (s·√T))."""
    if t <= 0 or s <= 0:
        return 0.0
    return float(norm.cdf((-p0 - m * t) / (s * np.sqrt(t))))


def volatilidad_por_nivel(precios: pd.Series, bloque: int = 63) -> pd.DataFrame:
    """En bloques trimestrales sin traslape: precio medio, desviación del cambio diario en USD y en %.

    Si la desviación en USD crece con el precio mientras la de % se mantiene, los movimientos son
    proporcionales al precio (como supone el GBM) y no fijos en dólares (como el browniano estándar).
    """
    dp = precios.diff()
    rp = precios.pct_change()
    filas = []
    for i in range(1, len(precios) - bloque + 1, bloque):
        tramo = slice(i, i + bloque)
        filas.append({
            "Desde": precios.index[i], "Precio medio": float(precios.iloc[tramo].mean()),
            "Movimiento diario USD": float(dp.iloc[tramo].std(ddof=1)),
            "Movimiento diario %": float(rp.iloc[tramo].std(ddof=1)),
        })
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Merton: difusión con saltos
# ---------------------------------------------------------------------------
@dataclass
class MertonCalibrado:
    sigma: float          # volatilidad de los días normales (anual)
    lam: float            # saltos por año
    mu_j: float           # tamaño medio del salto (log-retorno)
    delta_j: float        # dispersión del tamaño del salto
    fechas: pd.DatetimeIndex
    tamanos: pd.Series    # log-retorno de cada día de salto
    umbral: float         # |x − mediana| > umbral se considera salto
    sigma_total: float    # volatilidad total implícita: √(σ² + λ(μ_J² + δ_J²))


def calibrar_merton(x: pd.Series, k: float = 3.0, d: int = D) -> MertonCalibrado:
    """Detecta saltos como días con |x − mediana| > k · σ robusta (σ robusta = 1,4826 · MAD).

    La MAD no se infla con los propios saltos, así que el umbral no depende de ellos.
    σ se calcula con los días normales; λ = saltos / años de la muestra; μ_J y δ_J con los
    días de salto, descontando el drift de un día normal.
    """
    x = x.dropna()
    med = float(x.median())
    s_rob = 1.4826 * float((x - med).abs().median())
    umbral = k * s_rob
    es_salto = (x - med).abs() > umbral
    normales, saltos = x[~es_salto], x[es_salto]
    sigma = float(np.sqrt(d) * normales.std(ddof=1))
    anios = len(x) / d
    lam = len(saltos) / anios
    tam = saltos - normales.mean()
    mu_j = float(tam.mean()) if len(saltos) else 0.0
    delta_j = float(tam.std(ddof=1)) if len(saltos) > 1 else 0.0
    sigma_total = float(np.sqrt(sigma ** 2 + lam * (mu_j ** 2 + delta_j ** 2)))
    return MertonCalibrado(sigma, lam, mu_j, delta_j, saltos.index, saltos, umbral, sigma_total)


def precio_merton(s0: float, k: float, t: float, r: float, q: float, sigma: float, lam: float,
                  mu_j: float, delta_j: float, tipo: str = "call", n_terminos: int = 80) -> float:
    """Fórmula de Merton (1976) para la opción europea: promedio de Black-Scholes ponderado por
    la probabilidad de que ocurran n saltos antes del vencimiento.

        κ = exp(μ_J + ½δ_J²) − 1,   λ' = λ(1 + κ)
        σ_n² = σ² + n·δ_J²/T,       r_n = r − λκ + n·ln(1 + κ)/T
        V = Σ_n  Poisson(n; λ'T) · BS(S0, K, T, r_n, q, σ_n)
    """
    if t <= 0:
        return float(max(s0 - k, 0.0) if tipo == "call" else max(k - s0, 0.0))
    kappa = np.exp(mu_j + 0.5 * delta_j ** 2) - 1
    lam_p = lam * (1 + kappa)
    total = 0.0
    for n in range(n_terminos):
        peso = poisson.pmf(n, lam_p * t)
        if n > 5 and peso < 1e-14:
            break
        sig_n = np.sqrt(sigma ** 2 + n * delta_j ** 2 / t)
        r_n = r - lam * kappa + n * np.log1p(kappa) / t
        total += peso * black_scholes(s0, k, t, r_n, q, sig_n, tipo)      # Hull, cap. 27
    return float(total)


def prob_algun_salto(lam: float, t: float) -> float:
    """Probabilidad de al menos un salto en T años: 1 − e^{−λT}."""
    return float(1 - np.exp(-lam * t))


# ---------------------------------------------------------------------------
# Browniano fraccional y exponente de Hurst
# ---------------------------------------------------------------------------
ESCALAS_HURST = (1, 2, 4, 8, 16, 32)


def _desv_agregadas(arr: np.ndarray, escalas) -> np.ndarray:
    """Desviación (n − 1) de las sumas móviles de m retornos, para cada m."""
    cs = np.concatenate([[0.0], np.cumsum(arr)])
    return np.array([np.std(cs[m:] - cs[:-m], ddof=1) for m in escalas])


def volatilidad_por_horizonte(x: pd.Series, escalas=ESCALAS_HURST) -> pd.DataFrame:
    """Desviación de los retornos acumulados a m sesiones (sumas móviles) frente a la regla √m del GBM."""
    arr = np.asarray(pd.Series(x).dropna(), dtype=float)
    reales = _desv_agregadas(arr, escalas)
    s1 = float(np.std(arr, ddof=1))
    return pd.DataFrame({"Sesiones": list(escalas), "Real": reales,
                         "Regla √m (GBM)": [s1 * np.sqrt(m) for m in escalas]})


def hurst(x, escalas=ESCALAS_HURST) -> float:
    """Exponente de Hurst por varianza agregada: desv(suma de m retornos) ∝ m^H.
    H es la pendiente de log(desviación) contra log(m)."""
    arr = np.asarray(pd.Series(x).dropna(), dtype=float)
    return float(np.polyfit(np.log(escalas), np.log(_desv_agregadas(arr, escalas)), 1)[0])


def intervalo_sin_memoria(x, n_perm: int = 200, semilla: int = 42, escalas=ESCALAS_HURST) -> tuple[float, float]:
    """Rango P5–P95 de H al desordenar los días al azar (lo que daría una serie sin memoria del mismo
    tamaño). Incluye el sesgo de muestra finita del estimador."""
    rng = np.random.default_rng(semilla)
    arr = np.asarray(pd.Series(x).dropna(), dtype=float)
    hs = [hurst(rng.permutation(arr), escalas) for _ in range(n_perm)]
    return float(np.percentile(hs, 5)), float(np.percentile(hs, 95))


def tabla_hurst(x: pd.DataFrame, semilla: int = 42, n_perm: int = 200) -> pd.DataFrame:
    filas = []
    for i, t in enumerate(x.columns):
        h = hurst(x[t])
        lo, hi = intervalo_sin_memoria(x[t], n_perm, semilla + i)
        if h > hi:
            veredicto = "Tendencia"
        elif h < lo:
            veredicto = "Reversión"
        else:
            veredicto = "Sin memoria"
        filas.append({"Ticker": t, "H": h, "Sin memoria desde": lo, "Sin memoria hasta": hi, "Lectura": veredicto})
    return pd.DataFrame(filas)


def simular_fbm(h: float, sesiones: int = 252, n: int = 1, semilla: int = 42) -> np.ndarray:
    """Trayectorias (n, sesiones + 1) de browniano fraccional con B_H(1 año) de varianza 1.

    Se generan incrementos con covarianza γ(k) = ½(|k+1|^{2H} − 2|k|^{2H} + |k−1|^{2H}) por Cholesky.
    Con la misma semilla, los tres valores de H usan los mismos choques.
    """
    k = np.arange(sesiones)
    gam = 0.5 * (np.abs(k + 1) ** (2 * h) - 2 * np.abs(k) ** (2 * h) + np.abs(k - 1) ** (2 * h))
    cov = gam[np.abs(k[:, None] - k[None, :])]
    chol = np.linalg.cholesky(cov + 1e-12 * np.eye(sesiones))
    z = np.random.default_rng(semilla).standard_normal((n, sesiones))
    incr = z @ chol.T * (1 / sesiones) ** h
    return np.hstack([np.zeros((n, 1)), np.cumsum(incr, axis=1)])


def factor_var_hurst(h_var: int, h: float) -> float:
    """Cuánto cambia el VaR a h_var días si se escala con h^H en lugar de √h: h^H / h^0,5."""
    return float(h_var ** h / np.sqrt(h_var))


# ---------------------------------------------------------------------------
# Ornstein-Uhlenbeck · Vasicek
# ---------------------------------------------------------------------------
VALOR_CRITICO_DF = -2.86     # Dickey-Fuller con constante, 5 % (muestras grandes)
VALOR_CRITICO_DF_10 = -2.57  # ídem, 10 %


@dataclass
class OUCalibrado:
    kappa: float          # velocidad de reversión (por año)
    theta: float          # nivel de largo plazo
    sigma: float          # volatilidad (por año, en las unidades de x)
    b: float              # coeficiente AR(1) diario: x_{t+1} = a + b·x_t + ε
    t_df: float           # estadístico de Dickey-Fuller: (b − 1) / ee(b)
    n: int                # observaciones usadas
    dt: float             # paso en años (1/252)
    x0: float             # último valor observado

    @property
    def revierte(self) -> bool:
        """Hay evidencia estadística (5 %) de que la serie vuelve a su nivel de largo plazo."""
        return self.t_df < VALOR_CRITICO_DF and self.kappa > 0

    @property
    def evidencia(self) -> str:
        """"fuerte" (5 %), "débil" (solo al 10 %) o "ninguna": qué tan clara es la reversión a θ."""
        if self.kappa <= 0 or self.t_df >= VALOR_CRITICO_DF_10:
            return "ninguna"
        return "fuerte" if self.t_df < VALOR_CRITICO_DF else "débil"

    @property
    def vida_media(self) -> float:
        """Años que tarda en cerrarse la mitad de la distancia a θ: ln 2 / κ (infinito si κ = 0)."""
        return float(np.log(2) / self.kappa) if self.kappa > 0 else float("inf")

    @property
    def desv_largo_plazo(self) -> float:
        """Desviación estacionaria σ / √(2κ): cuánto se aleja normalmente x de θ."""
        return float(self.sigma / np.sqrt(2 * self.kappa)) if self.kappa > 0 else float("inf")

    @property
    def z_hoy(self) -> float:
        """A cuántas desviaciones de largo plazo está hoy x de θ."""
        return float((self.x0 - self.theta) / self.desv_largo_plazo) if self.kappa > 0 else 0.0


def calibrar_ou(x: pd.Series, d: int = D) -> OUCalibrado:
    """Calibración exacta por regresión AR(1) de la discretización de Ornstein-Uhlenbeck:

        x_{t+Δt} = a + b·x_t + ε,   b = e^{−κΔt},   a = θ(1 − b),   Var(ε) = σ²(1 − b²) / (2κ)

    Si b ≥ 1 (sin reversión en la muestra) se toma κ = 0, θ = último valor y σ = √d · desv(Δx).
    """
    x = pd.Series(x).dropna().astype(float)
    if len(x) < 30:
        raise ValueError("Se necesitan al menos 30 observaciones para calibrar Ornstein-Uhlenbeck.")
    dt = 1 / d
    y, z = x.to_numpy()[1:], x.to_numpy()[:-1]
    n = len(y)
    zc = z - z.mean()
    b = float((zc * (y - y.mean())).sum() / (zc ** 2).sum())
    a = float(y.mean() - b * z.mean())
    resid = y - a - b * z
    s_e = float(np.sqrt((resid ** 2).sum() / (n - 2)))
    ee_b = s_e / float(np.sqrt((zc ** 2).sum()))
    t_df = (b - 1) / ee_b if ee_b > 0 else 0.0
    if 0 < b < 1:
        kappa = float(-np.log(b) / dt)
        theta = float(a / (1 - b))
        sigma = float(s_e * np.sqrt(2 * kappa / (1 - b ** 2)))
    else:
        kappa, theta = 0.0, float(x.iloc[-1])
        sigma = float(np.sqrt(d) * np.diff(x.to_numpy()).std(ddof=1))
    return OUCalibrado(kappa, theta, sigma, b, float(t_df), n, dt, float(x.iloc[-1]))


def simular_ou(x0: float, kappa: float, theta: float, sigma: float, sesiones: int, n: int = 5_000,
               semilla: int = 42, dt: float = 1 / D) -> np.ndarray:
    """Matriz (n, sesiones + 1) con la transición exacta de Ornstein-Uhlenbeck:
    x_{t+Δt} = θ + (x_t − θ)e^{−κΔt} + σ·√((1 − e^{−2κΔt}) / (2κ))·Z.  Con κ = 0 es un browniano."""
    rng = np.random.default_rng(semilla)
    z = rng.standard_normal((n, sesiones))
    if kappa > 0:
        b = np.exp(-kappa * dt)
        sd = sigma * np.sqrt((1 - b ** 2) / (2 * kappa))
    else:
        b, sd = 1.0, sigma * np.sqrt(dt)
    tray = np.empty((n, sesiones + 1))
    tray[:, 0] = x0
    for j in range(sesiones):
        tray[:, j + 1] = theta + (tray[:, j] - theta) * b + sd * z[:, j]
    return tray


def media_ou(x0: float, kappa: float, theta: float, t: float) -> float:
    """E[x_T] = θ + (x0 − θ)e^{−κT}."""
    return float(theta + (x0 - theta) * np.exp(-kappa * t))


def desv_ou(kappa: float, sigma: float, t: float) -> float:
    """Desv(x_T) = σ·√((1 − e^{−2κT}) / (2κ));  con κ = 0, σ√T."""
    if kappa <= 0:
        return float(sigma * np.sqrt(t))
    return float(sigma * np.sqrt((1 - np.exp(-2 * kappa * t)) / (2 * kappa)))


def tasa_promedio_vasicek(r0: float, kappa: float, theta: float, sigma: float, t: float) -> tuple[float, float]:
    """Media y desviación de la tasa promedio r̄_T = (1/T)∫₀ᵀ r_s ds bajo Vasicek (es normal).

        E[r̄] = θ + (r0 − θ)(1 − e^{−κT}) / (κT)
        Var(∫r) = σ²/κ² · [T − 2(1 − e^{−κT})/κ + (1 − e^{−2κT})/(2κ)]
    Es la tasa que importa para descontar un pago en T.  Con κ = 0: media r0 y Var(∫r) = σ²T³/3.
    """
    if t <= 0:
        return float(r0), 0.0
    if kappa <= 1e-10:
        return float(r0), float(sigma * np.sqrt(t ** 3 / 3) / t)
    e1, e2 = np.exp(-kappa * t), np.exp(-2 * kappa * t)
    media = theta + (r0 - theta) * (1 - e1) / (kappa * t)
    var_int = sigma ** 2 / kappa ** 2 * (t - 2 * (1 - e1) / kappa + (1 - e2) / (2 * kappa))
    return float(media), float(np.sqrt(max(var_int, 0.0)) / t)


def bono_vasicek(r0: float, kappa: float, theta: float, sigma: float, t: float) -> float:
    """Precio de un bono cero cupón que paga 1 en T (Vasicek, 1977), sin prima por riesgo de tasa:
    P = A·e^{−B·r0},  B = (1 − e^{−κT})/κ,  ln A = (θ − σ²/(2κ²))(B − T) − σ²B²/(4κ)."""
    if t <= 0:
        return 1.0
    if kappa <= 1e-10:
        return float(np.exp(-r0 * t + sigma ** 2 * t ** 3 / 6))
    b = (1 - np.exp(-kappa * t)) / kappa
    ln_a = (theta - sigma ** 2 / (2 * kappa ** 2)) * (b - t) - sigma ** 2 * b ** 2 / (4 * kappa)
    return float(np.exp(ln_a - b * r0))


def curva_vasicek(r0: float, kappa: float, theta: float, sigma: float,
                  plazos=(0.25, 0.5, 1, 2, 3, 5, 7, 10)) -> pd.DataFrame:
    """Tasa continua implícita y(T) = −ln P(T) / T para varios plazos."""
    return pd.DataFrame({"Plazo (años)": list(plazos),
                         "Tasa": [-np.log(bono_vasicek(r0, kappa, theta, sigma, t)) / t for t in plazos]})


# ---------------------------------------------------------------------------
# Herding: comportamiento en manada
# ---------------------------------------------------------------------------
def correlacion_promedio_movil(x: pd.DataFrame, ventana: int = 63) -> pd.Series:
    """Promedio de las correlaciones entre todos los pares de acciones, en ventanas móviles de `ventana`
    sesiones. Si los inversionistas actúan en manada, las acciones se mueven juntas y sube."""
    x = x.dropna()
    arr = x.to_numpy()
    k = arr.shape[1]
    iu = np.triu_indices(k, 1)
    valores = [np.corrcoef(arr[i - ventana:i].T)[iu].mean() for i in range(ventana, len(arr) + 1)]
    return pd.Series(valores, index=x.index[ventana - 1:], name="Correlación promedio")


def correlacion_promedio(corr: pd.DataFrame) -> float:
    """Promedio de las correlaciones fuera de la diagonal."""
    c = np.asarray(corr, dtype=float)
    return float(c[np.triu_indices(len(c), 1)].mean())


@dataclass
class CSADResultado:
    datos: pd.DataFrame   # columnas: R_m, CSAD
    a: float
    g1: float             # coeficiente de |R_m|
    g2: float             # coeficiente de R_m² (negativo y significativo = manada)
    t_g2: float
    r2: float

    @property
    def hay_manada(self) -> bool:
        return self.g2 < 0 and self.t_g2 < -1.96


def csad(x: pd.DataFrame) -> CSADResultado:
    """Prueba de Chang, Cheng y Khorana (2000).

        CSAD_t = (1/N) Σ_i |R_i,t − R_m,t|,     R_m,t = promedio simple de los retornos del día
        CSAD_t = a + γ1·|R_m,t| + γ2·R_m,t² + e_t

    Sin manada, la dispersión crece en línea recta con el tamaño del movimiento del mercado (γ2 ≥ 0).
    Con manada, en los días de movimientos grandes todos hacen lo mismo y la dispersión crece menos o
    cae: γ2 < 0. Errores estándar de Newey-West (5 rezagos), porque la dispersión tiene persistencia.
    """
    x = x.dropna()
    rm = x.mean(axis=1)
    disp = x.sub(rm, axis=0).abs().mean(axis=1)
    X = np.column_stack([np.ones(len(rm)), rm.abs(), rm ** 2])
    y = disp.to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    n = len(y)
    xtx_inv = np.linalg.inv(X.T @ X)
    s = (X * e[:, None]).T @ (X * e[:, None])
    for l in range(1, 6):
        w = 1 - l / 6
        g = (X[l:] * e[l:, None]).T @ (X[:-l] * e[:-l, None])
        s += w * (g + g.T)
    cov_b = xtx_inv @ s @ xtx_inv
    r2 = 1 - float((e ** 2).sum() / ((y - y.mean()) ** 2).sum())
    return CSADResultado(pd.DataFrame({"R_m": rm, "CSAD": disp}), float(beta[0]), float(beta[1]), float(beta[2]),
                         float(beta[2] / np.sqrt(cov_b[2, 2])), r2)


def cov_manada(cov_diaria: pd.DataFrame, h: float) -> pd.DataFrame:
    """Covarianza cuando una fracción h de cada movimiento viene de copiar a la multitud.

    Cada acción: r_i = σ_i(√h·M + √(1 − h)·ε_i), con M común a todas y ε con la correlación real C.
    Entonces la correlación pasa a ser ρ_ij(h) = h + (1 − h)·C_ij, y las volatilidades no cambian.
    """
    if not 0 <= h <= 1:
        raise ValueError("h debe estar entre 0 y 1.")
    s = np.sqrt(np.diag(cov_diaria.to_numpy()))
    c = cov_diaria.to_numpy() / np.outer(s, s)
    c_h = h + (1 - h) * c
    np.fill_diagonal(c_h, 1.0)
    return pd.DataFrame(np.outer(s, s) * c_h, index=cov_diaria.index, columns=cov_diaria.columns)


def h_para_correlacion(rho_actual: float, rho_objetivo: float) -> float:
    """Intensidad h que lleva la correlación promedio de rho_actual a rho_objetivo: (ρ* − ρ̄) / (1 − ρ̄)."""
    if rho_actual >= 1:
        return 0.0
    return float(np.clip((rho_objetivo - rho_actual) / (1 - rho_actual), 0, 1))


def var_por_manada(cov_diaria: pd.DataFrame, w: pd.Series, capital: float, alpha: float, horizonte: int,
                   pasos=np.linspace(0, 1, 21)) -> pd.DataFrame:
    """VaR y ES normales (media cero, como en la pestaña Riesgo) para cada intensidad de manada h."""
    from .riesgo import var_normal

    filas = []
    sd = np.sqrt(np.diag(cov_diaria.to_numpy()))
    corr_media = correlacion_promedio(cov_diaria.to_numpy() / np.outer(sd, sd))
    for h in pasos:
        r = var_normal(cov_manada(cov_diaria, float(h)), w, capital, alpha, horizonte)
        filas.append({"h": float(h), "Correlación promedio": h + (1 - h) * corr_media,
                      "VaR": r["var"], "ES": r["es"]})
    return pd.DataFrame(filas)


def simular_manada(cov_diaria: pd.DataFrame, w: pd.Series, capital: float, h: float, horizonte: int,
                   n: int = 20_000, semilla: int = 42) -> np.ndarray:
    """Pérdidas simuladas de la cartera a `horizonte` días con intensidad de manada h (media cero).
    Con la misma semilla, todas las intensidades usan los mismos choques M y ε."""
    tick = list(cov_diaria.index)
    s = np.sqrt(np.diag(cov_diaria.to_numpy()))
    c = cov_diaria.to_numpy() / np.outer(s, s)
    chol = np.linalg.cholesky(c + 1e-12 * np.eye(len(c)))
    rng = np.random.default_rng(semilla)
    m = rng.standard_normal((n, 1))
    eps = rng.standard_normal((n, len(tick))) @ chol.T
    r = s * (np.sqrt(h) * m + np.sqrt(1 - h) * eps) * np.sqrt(horizonte)
    return -capital * (r @ w[tick].to_numpy())


# ---------------------------------------------------------------------------
# Opciones reales
# ---------------------------------------------------------------------------
TIPOS_OPCION_REAL = ("esperar", "expandir", "abandonar", "expandir o abandonar")


@dataclass
class OpcionReal:
    tipo: str
    opcion: float           # valor de la flexibilidad (la opción americana)
    vpn: float              # VPN estático: V0 − I
    vpn_estrategico: float  # VPN con la flexibilidad
    tiempos: list           # tiempo (años) de cada paso del árbol
    valores_v: list         # valor del proyecto en cada nodo, por paso
    decision: list          # decisión óptima en cada nodo: "Esperar", "Invertir", "Expandir", "Abandonar", "Seguir"
    parametros: object      # u, d, p del árbol (crr.ParametrosCRR)


def _ejercicio(tipo: str, v: np.ndarray, inversion: float, expansion: float, costo_expansion: float,
               rescate: float) -> tuple[np.ndarray, np.ndarray]:
    """Valor de ejercer ya en cada nodo y la etiqueta de esa decisión."""
    if tipo == "esperar":
        return v - inversion, np.full(v.shape, "Invertir", dtype=object)
    if tipo == "expandir":
        return expansion * v - costo_expansion, np.full(v.shape, "Expandir", dtype=object)
    if tipo == "abandonar":
        return rescate - v, np.full(v.shape, "Abandonar", dtype=object)
    exp_, aba = expansion * v - costo_expansion, rescate - v
    return np.maximum(exp_, aba), np.where(exp_ >= aba, "Expandir", "Abandonar").astype(object)


def opcion_real(v0: float, sigma: float, r: float, delta: float, t: float, n: int, tipo: str = "esperar",
                inversion: float = 0.0, expansion: float = 0.0, costo_expansion: float = 0.0,
                rescate: float = 0.0) -> OpcionReal:
    """Valora la flexibilidad de un proyecto con un árbol CRR sobre su valor V (Ec. 6 y 7 del Paso 8).

    V0: valor presente de los flujos del proyecto; δ: flujos que se dejan de recibir por esperar (hace el
    papel del dividendo q). La flexibilidad es una opción americana, con valor de ejercer:
        esperar:   V − I                  (call sobre V con strike I; el proyecto aún no se ha hecho)
        expandir:  e·V − I_e              (crecer un e·100 % pagando I_e)
        abandonar: A − V                  (put: vender los activos por A)
        expandir o abandonar: el mayor de los dos (se ejerce una sola de las dos)
    VPN estratégico = VPN estático + valor de la flexibilidad; en "esperar", es el valor de la opción
    (todavía no se ha pagado I), y la flexibilidad es lo que vale por encima de máx(VPN, 0).
    """
    if tipo not in TIPOS_OPCION_REAL:
        raise ValueError(f"tipo debe ser uno de {TIPOS_OPCION_REAL}.")
    if v0 <= 0 or sigma <= 0 or t <= 0 or int(n) != n or n < 1:
        raise ValueError("Se requiere V0 > 0, σ > 0, T > 0 y N entero ≥ 1.")
    par = parametros_crr(t, int(n), r, delta, sigma)
    desc = np.exp(-r * par.dt)
    valores_v = [v0 * par.u ** np.arange(k + 1) * par.d ** (k - np.arange(k + 1)) for k in range(n + 1)]
    ej, et = _ejercicio(tipo, valores_v[n], inversion, expansion, costo_expansion, rescate)
    val = np.maximum(ej, 0.0)
    espera = "Esperar" if tipo == "esperar" else "Seguir"
    decision = [None] * (n + 1)
    decision[n] = np.where(ej > 0, et, "No invertir" if tipo == "esperar" else "Seguir").astype(object)
    for k in range(n - 1, -1, -1):
        cont = desc * (par.p * val[1:] + (1 - par.p) * val[:-1])
        ej, et = _ejercicio(tipo, valores_v[k], inversion, expansion, costo_expansion, rescate)
        ejercer = (ej > cont) & (ej > 0)
        val = np.where(ejercer, ej, cont)
        decision[k] = np.where(ejercer, et, espera).astype(object)
    opcion = float(val[0])
    vpn = v0 - inversion
    if tipo == "esperar":
        estrategico = opcion
    else:
        estrategico = vpn + opcion
    return OpcionReal(tipo, opcion, vpn, estrategico, list(np.arange(n + 1) * par.dt), valores_v, decision, par)


def flexibilidad(o: OpcionReal) -> float:
    """Cuánto vale tener la opción por encima de decidir hoy: en "esperar", opción − máx(VPN, 0);
    en las demás, el valor de la opción."""
    return o.opcion - max(o.vpn, 0.0) if o.tipo == "esperar" else o.opcion


def sensibilidad_volatilidad(v0: float, r: float, delta: float, t: float, n: int, inversion: float,
                             expansion: float, costo_expansion: float, rescate: float,
                             sigmas=np.linspace(0.05, 0.6, 23)) -> pd.DataFrame:
    """Valor de cada flexibilidad para distintas volatilidades del proyecto."""
    filas = []
    for s in sigmas:
        fila = {"σ": float(s)}
        for tipo in TIPOS_OPCION_REAL:
            try:
                fila[tipo] = flexibilidad(opcion_real(v0, float(s), r, delta, t, n, tipo, inversion, expansion,
                                                      costo_expansion, rescate))
            except ValueError:      # p^Q fuera de (0, 1) con σ muy baja
                fila[tipo] = np.nan
        filas.append(fila)
    return pd.DataFrame(filas)
