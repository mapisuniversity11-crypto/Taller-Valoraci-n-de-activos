"""Retornos y calibración del modelo (Paso 3). Entrega 1.

Funciones previstas:
- retornos_simples(P): R = P_t / P_{t-1} - 1  (Ec. 1)
- log_retornos(P): x = ln(P_t / P_{t-1})  (Ec. 1)
- calibrar(x): sigma = sqrt(D) * s_x (divisor n-1); mu_total = D * mean(x) + 0.5 * sigma^2  (Ec. 2)
- drift_exdividendo(mu_total, q): a = mu_total - q
- covarianza / correlación en las mismas fechas, divisor n-1
"""
