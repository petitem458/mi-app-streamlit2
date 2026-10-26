"""Modelo de datos: perfiles de CPU por tipo, tráfico de los enlaces (Mbps) y costo de las rutas.

Las ARISTAS de la red son los caminos por donde se comunican los equipos y su
peso es el TRÁFICO MEDIDO EN Mbps. El costo de atravesar el enlace u→v es:

    c(u→v) = tráfico(u,v) [Mbps] + 30 Mbps·[v está Anómalo]

(el tráfico que ya circula por el camino, más 30 Mbps equivalentes a las
retransmisiones que provoca un equipo en observación). Los equipos Bloqueados
no se pueden atravesar. Con este costo Dijkstra produce las rutas de menor
tráfico acumulado con las que se entrena la red neuronal; la red no lo conoce: lo aprende.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

TIPOS = ("Servidor", "PC", "Portátil", "Impresora", "IoT")

# Uso de CPU esperado por tipo: (media, desviación) en %
PERFIL_CPU = {
    "Servidor": (55.0, 10.0),
    "PC": (40.0, 12.0),
    "Portátil": (35.0, 12.0),
    "Impresora": (18.0, 7.0),
    "IoT": (25.0, 9.0),
}

# Rango de tráfico medido (Mbps) y latencia (ms) por tipo de enlace
ENLACES = {
    "troncal":  {"trafico": (60.0, 140.0), "latencia": (0.5, 2.0)},
    "cableado": {"trafico": (10.0, 70.0),  "latencia": (1.0, 5.0)},
    "wifi":     {"trafico": (5.0, 45.0),   "latencia": (4.0, 15.0)},
}
TRAFICO_MAXIMO = 300.0          # tope del tráfico de un enlace (Mbps)

# Costo real de una ruta (en Mbps)
PENALIZACION_ANOMALO = 30.0

# Clasificación del tráfico de un enlace
TRAFICO_MEDIO = 30.0            # ≥ 30 Mbps: tráfico medio
TRAFICO_ALTO = 60.0             # ≥ 60 Mbps: enlace congestionado

# Eventos de comunicación: P(ERR) = 0,02 + 0,05·máx(d − 1, 0)
PROB_ERR_BASE = 0.02
PROB_ERR_PENDIENTE = 0.05
PROB_ERR_SIN_SERVICIO = 0.85
TRAFICO_NORMAL = 45.0           # tráfico medio esperado en los enlaces de un equipo (Mbps)
DISPERSION_TRAFICO = 10.0       # Mbps por unidad de desviación


def desviacion(g: nx.Graph, nodo: str) -> float:
    """d_i = máx(z_CPU, (T̄ − 45)/10, 0): carga del equipo y tráfico medio (Mbps) de sus enlaces."""
    d = g.nodes[nodo]
    media, sigma = PERFIL_CPU[d["tipo"]]
    z_cpu = (d["cpu"] - media) / sigma
    trafico = [a["trafico"] for _, _, a in g.edges(nodo, data=True) if a.get("activa", True)]
    z_red = (float(np.mean(trafico)) - TRAFICO_NORMAL) / DISPERSION_TRAFICO if trafico else 0.0
    return max(z_cpu, z_red, 0.0)


def prob_error(g: nx.Graph, nodo: str) -> float:
    """Probabilidad de que un intervalo de comunicación termine en ERR."""
    if g.nodes[nodo].get("sin_servicio"):
        return PROB_ERR_SIN_SERVICIO
    d = desviacion(g, nodo)
    return float(np.clip(PROB_ERR_BASE + PROB_ERR_PENDIENTE * max(d - 1.0, 0.0), PROB_ERR_BASE, 0.85))


def costo_real(g: nx.Graph, u: str, v: str) -> float:
    """Costo (Mbps) de pasar de u a v: tráfico del enlace + penalización si v está Anómalo."""
    anomalo = g.nodes[v].get("estado_afd") == "Anómalo"
    return g.edges[u, v]["trafico"] + PENALIZACION_ANOMALO * anomalo


NOMBRES_CARACTERISTICAS = (
    "trafico/100", "(trafico/100)²", "latencia/15", "inalambrico",
    "llegada_anomalo", "cpu_llegada/100",
    "es_Servidor", "es_PC", "es_Portátil", "es_Impresora", "es_IoT",
)


def caracteristicas_arista(g: nx.Graph, u: str, v: str) -> np.ndarray:
    """Vector φ(u→v) que recibe la red neuronal para estimar el costo del enlace."""
    a, dv = g.edges[u, v], g.nodes[v]
    tipo = np.zeros(len(TIPOS))
    tipo[TIPOS.index(dv["tipo"])] = 1.0
    t = a["trafico"] / 100.0
    return np.concatenate([[t, t * t, a["latencia"] / 15.0, float(a["clase"] == "wifi"),
                            float(dv.get("estado_afd") == "Anómalo"), dv["cpu"] / 100.0], tipo])
