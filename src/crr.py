"""Árbol binomial Cox-Ross-Rubinstein (Paso 8). Entrega 1.

Funciones previstas:
- parametros_crr(T, N, r, q, sigma): dt, u, d, pQ  (Ec. 6); validar 0 < pQ < 1
- precio_crr(S0, K, T, r, q, sigma, N, tipo='call'|'put', estilo='europea'|'americana')  (Ec. 7); T = 0 -> payoff
- arbol_crr(...): matriz de precios y valores para dibujar el árbol de 3 pasos
- Control: call europea S0=K=100, T=1, r=0.05, q=0, sigma=0.20 -> ~10.4506
"""
