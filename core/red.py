"""Red fija de SoftCast: los 18 equipos son los nodos y los 24 caminos de comunicación las aristas.

G = (V, E, w): grafo no dirigido y ponderado. El peso w de cada arista es el
tráfico medido en Mbps. Cada equipo tiene dos o tres caminos diferentes
(grado 2 o 3) y la red es biconexa: entre cualquier par de equipos existen al
menos dos rutas sin nodos intermedios comunes. La topología es fija; el
tráfico, la CPU, las latencias y los eventos son aleatorios.
"""

from __future__ import annotations

import copy
import zlib
from collections import deque

import networkx as nx
import numpy as np

from . import automata as au
from . import datos as dt

AFD = au.afd_dispositivo()

# --------------------------------------------------------------------------
# 1. Inventario y topología fijos
# --------------------------------------------------------------------------
SERVIDORES = ["S1", "S2"]
INVENTARIO = {
    "S1": ("Servidor", "Servidor de aplicaciones y archivos", "Administrativa"),
    "S2": ("Servidor", "Servidor de bases de datos y respaldo", "Operativa"),
    "PC1": ("PC", "Computador de escritorio", "Administrativa"),
    "PC2": ("PC", "Computador de escritorio", "Administrativa"),
    "PC3": ("PC", "Computador de escritorio", "Administrativa"),
    "PC4": ("PC", "Computador de escritorio", "Operativa"),
    "PC5": ("PC", "Computador de escritorio", "Operativa"),
    "PT1": ("Portátil", "Computador portátil (wifi)", "Administrativa"),
    "PT2": ("Portátil", "Computador portátil (wifi)", "Administrativa"),
    "PT3": ("Portátil", "Computador portátil (wifi)", "Operativa"),
    "PT4": ("Portátil", "Computador portátil (wifi)", "Operativa"),
    "PT5": ("Portátil", "Computador portátil (wifi)", "Administrativa"),
    "IM1": ("Impresora", "Impresora de red", "Administrativa"),
    "IM2": ("Impresora", "Impresora de red", "Operativa"),
    "IoT1": ("IoT", "Cámara de seguridad", "Administrativa"),
    "IoT2": ("IoT", "Sensor de temperatura", "Administrativa"),
    "IoT3": ("IoT", "Control de acceso", "Operativa"),
    "IoT4": ("IoT", "Cámara de seguridad", "Operativa"),
}

# Anillo de 16 equipos (sentido horario, desde la cámara IoT1 en la parte superior)
ANILLO = ("IoT1", "PC4", "PT3", "IM2", "IoT3", "PT4", "IoT4", "PC5",
          "PT5", "PC3", "IoT2", "PT2", "PC2", "IM1", "PT1", "PC1")
# Troncal entre servidores, cuatro acometidas desde los servidores y tres enlaces transversales
ENLACES_EXTRA = (("S1", "S2"), ("S1", "PC1"), ("S1", "PC3"), ("S2", "PC4"), ("S2", "PC5"),
                 ("PT1", "PC2"), ("PT3", "IoT3"), ("PT2", "PT4"))


def clase_enlace(u: str, v: str) -> str:
    """troncal entre servidores; wifi si participa un portátil o un IoT; si no, cableado."""
    tipos = {INVENTARIO[u][0], INVENTARIO[v][0]}
    if tipos == {"Servidor"}:
        return "troncal"
    return "wifi" if tipos & {"Portátil", "IoT"} else "cableado"


ENLACES_FIJOS = [(u, v, clase_enlace(u, v)) for u, v in
                 [(ANILLO[i], ANILLO[(i + 1) % len(ANILLO)]) for i in range(len(ANILLO))]
                 + list(ENLACES_EXTRA)]

ORDEN = list(INVENTARIO)


def ordenar_nodos(nodos) -> list[str]:
    return sorted(nodos, key=ORDEN.index)


def _rng(*partes) -> np.random.Generator:
    """Generador determinista a partir de varias claves (reproducibilidad)."""
    return np.random.default_rng(zlib.crc32("|".join(str(p) for p in partes).encode()))


def _fijar_trafico(a: dict, trafico: float) -> None:
    """El peso de la arista: tráfico medido en Mbps (1 decimal)."""
    a["trafico"] = round(float(np.clip(trafico, 1.0, dt.TRAFICO_MAXIMO)), 1)


def construir_topologia(semilla: int) -> nx.Graph:
    """Grafo fijo de 18 nodos y 24 aristas con tráfico (Mbps) y latencia aleatorios."""
    rng = _rng("enlaces", semilla)
    g = nx.Graph(semilla=semilla)
    for v in ORDEN:
        tipo, rol, area = INVENTARIO[v]
        g.add_node(v, tipo=tipo, rol=rol, area=area)
    for u, v, clase in ENLACES_FIJOS:
        p = dt.ENLACES[clase]
        a = {"clase": clase, "activa": True, "latencia": round(float(rng.uniform(*p["latencia"])), 2)}
        _fijar_trafico(a, rng.uniform(*p["trafico"]))
        g.add_edge(u, v, **a)
    return g


# --------------------------------------------------------------------------
# 2. Métricas aleatorias y fallas
# --------------------------------------------------------------------------
def inyectar_falla(g: nx.Graph, nodo: str, rng: np.random.Generator) -> None:
    """Satura el equipo: CPU alta y +30 a +50 Mbps de tráfico en cada uno de sus caminos."""
    g.nodes[nodo]["cpu"] = float(rng.uniform(90, 99.5))
    g.nodes[nodo]["falla_inyectada"] = True
    for v in g.neighbors(nodo):
        a = g.edges[nodo, v]
        _fijar_trafico(a, a["trafico"] + rng.uniform(30.0, 50.0))


def generar_red(semilla: int = 5, proporcion_fallas: float = 0.12) -> nx.Graph:
    """Red fija con CPU y tráfico aleatorios y fallas iniciales en equipos que no son servidores."""
    g = construir_topologia(semilla)
    rng = _rng("metricas", semilla)
    for v in g.nodes:
        media, sigma = dt.PERFIL_CPU[g.nodes[v]["tipo"]]
        g.nodes[v].update(cpu=float(np.clip(rng.normal(media, sigma), 1, 100)), falla_inyectada=False)
    candidatos = [v for v in ORDEN if g.nodes[v]["tipo"] != "Servidor"]
    k = round(proporcion_fallas * len(candidatos)) if proporcion_fallas > 0 else 0
    for nodo in rng.choice(candidatos, size=k, replace=False) if k else []:
        inyectar_falla(g, str(nodo), rng)
    return g


# --------------------------------------------------------------------------
# 3. Recorrido BFS con evaluación del AFD
# --------------------------------------------------------------------------
def _evaluar_nodo(g: nx.Graph, v: str, habilitar: set[str]) -> None:
    d = g.nodes[v]
    p = dt.prob_error(g, v)
    rng = _rng("eventos", g.graph["semilla"], v, round(p, 2), d.get("bloqueo_manual", False))
    cadena = au.generar_eventos(rng, p, bloqueo_manual=d.get("bloqueo_manual", False),
                                habilitar=v in habilitar)
    n_obs = sum(s in (au.OK, au.ERR) for s in cadena)
    d.update(cadena=cadena, prob_error=round(p, 3), estado_afd=AFD.estado_final(cadena),
             tasa_fallas=cadena.count(au.ERR) / max(1, n_obs),
             traza_estados=[AFD.inicial] + [t[2] for t in AFD.recorrer(cadena)])


def recorrer_red(g: nx.Graph, habilitar: set[str] = frozenset()) -> list[dict]:
    """ALGORITMO DE RECORRIDO: búsqueda en anchura (BFS) desde los servidores.

    Cada nodo visitado genera su cadena de eventos y el AFD fija su estado.
    Si queda Bloqueado no se exploran sus vecinos. Los nodos no alcanzados
    quedan sin servicio. Complejidad O(|V| + |E|).
    """
    for v in g.nodes:
        g.nodes[v].update(visitado=False, nivel=None, padre=None, sin_servicio=False)
    cola = deque()
    for s in SERVIDORES:
        if not g.nodes[s].get("bloqueo_manual"):
            g.nodes[s].update(visitado=True, nivel=0)
            cola.append(s)
    orden = []
    while cola:
        u = cola.popleft()
        _evaluar_nodo(g, u, habilitar)
        du = g.nodes[u]
        expande = du["estado_afd"] != au.BLOQUEADO
        orden.append({"paso": len(orden) + 1, "nodo": u, "nivel": du["nivel"], "padre": du["padre"],
                      "estado": du["estado_afd"], "expande": expande})
        if not expande:
            continue
        for v in ordenar_nodos(g.neighbors(u)):
            dv = g.nodes[v]
            if dv["visitado"] or dv.get("bloqueo_manual") or not g.edges[u, v]["activa"]:
                continue
            dv.update(visitado=True, nivel=du["nivel"] + 1, padre=u)
            cola.append(v)
    for v in ORDEN:
        dv = g.nodes[v]
        if dv["visitado"]:
            continue
        if not dv.get("bloqueo_manual"):
            dv["sin_servicio"] = True
        _evaluar_nodo(g, v, habilitar)
        orden.append({"paso": None, "nodo": v, "nivel": None, "padre": None,
                      "estado": dv["estado_afd"], "expande": False})
    return orden


# --------------------------------------------------------------------------
# 4. Simulación de escenarios
# --------------------------------------------------------------------------
def _desactivar(g: nx.Graph, nodo: str) -> None:
    for v in g.neighbors(nodo):
        g.edges[nodo, v]["activa"] = False


def simular(g_cruda: nx.Graph, nodos_bloqueados=(), enlaces_bloqueados=(), fallas=(),
            habilitar=()) -> nx.Graph:
    """Aplica un escenario sobre una copia de la red y la evalúa.

    Orden: fallas → bloqueo de enlaces (sus Mbps se suman a la ruta alterna de
    menor tráfico) → bloqueo manual de nodos (si es un servidor, el otro absorbe
    su carga: +20 % de CPU y +15 Mbps en sus caminos) → recorrido BFS con el AFD → aislamiento de los bloqueados.
    """
    g = copy.deepcopy(g_cruda)
    semilla = g.graph["semilla"]
    for v in g.nodes:
        g.nodes[v].update(bloqueo_manual=False, estado_afd=au.NORMAL)
    for nodo in fallas:
        inyectar_falla(g, nodo, _rng("falla_manual", semilla, nodo))

    for u, v in enlaces_bloqueados:
        if not g.has_edge(u, v):
            continue
        trafico = g.edges[u, v]["trafico"]
        g.edges[u, v]["activa"] = False
        try:
            ruta = nx.shortest_path(subgrafo_activo(g), u, v, weight="trafico")
            for a, b in zip(ruta, ruta[1:]):
                e = g.edges[a, b]
                _fijar_trafico(e, e["trafico"] + trafico)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass

    for b in nodos_bloqueados:
        g.nodes[b]["bloqueo_manual"] = True
        _desactivar(g, b)
        if g.nodes[b]["tipo"] == "Servidor":
            for s in SERVIDORES:
                if s != b and s not in nodos_bloqueados:
                    g.nodes[s]["cpu"] = float(min(100.0, g.nodes[s]["cpu"] + 20))
                    for v in g.neighbors(s):
                        e = g.edges[s, v]
                        if e["activa"]:
                            _fijar_trafico(e, e["trafico"] + 15.0)

    orden = recorrer_red(g, set(habilitar))
    for v in ORDEN:
        d = g.nodes[v]
        d["motivo_bloqueo"] = None
        if d["estado_afd"] == au.BLOQUEADO:
            d["motivo_bloqueo"] = ("Manual (BLQ)" if d["bloqueo_manual"] else
                                   "Sin servicio" if d["sin_servicio"] else "AFD: 2 fallas seguidas")
            _desactivar(g, v)
        d["estado"] = d["estado_afd"]
    g.graph["orden_recorrido"] = orden
    g.graph["escenario"] = {"nodos": list(nodos_bloqueados), "enlaces": [list(e) for e in enlaces_bloqueados],
                            "fallas": list(fallas), "habilitar": list(habilitar)}
    return g


def nodos_activos(g: nx.Graph) -> list[str]:
    return [v for v in ORDEN if g.nodes[v].get("estado_afd") != au.BLOQUEADO]


def subgrafo_activo(g: nx.Graph) -> nx.Graph:
    h = g.subgraph(nodos_activos(g)).copy()
    h.remove_edges_from([(u, v) for u, v, a in h.edges(data=True) if not a["activa"]])
    return h


# --------------------------------------------------------------------------
# 5. Rutas de referencia (maestro y línea base)
# --------------------------------------------------------------------------
def grafo_costos(g: nx.Graph) -> nx.DiGraph:
    """Grafo dirigido de los enlaces utilizables con el costo real de cada sentido."""
    h = subgrafo_activo(g)
    dg = nx.DiGraph()
    dg.add_nodes_from(h.nodes)
    for u, v in h.edges:
        dg.add_edge(u, v, costo=dt.costo_real(g, u, v))
        dg.add_edge(v, u, costo=dt.costo_real(g, v, u))
    return dg


def costo_ruta(g: nx.Graph, ruta: list[str]) -> float:
    return float(sum(dt.costo_real(g, a, b) for a, b in zip(ruta, ruta[1:])))


def ruta_optima(g: nx.Graph, origen: str, destino: str) -> list[str] | None:
    """Dijkstra sobre el costo real (Mbps): el camino óptimo que la GNN debe aprender."""
    try:
        return nx.dijkstra_path(grafo_costos(g), origen, destino, weight="costo")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def ruta_saltos(g: nx.Graph, origen: str, destino: str) -> list[str] | None:
    """Línea base: el camino con menos saltos, que ignora el tráfico (Mbps)."""
    try:
        return nx.shortest_path(grafo_costos(g), origen, destino)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def metricas_red(g: nx.Graph) -> dict:
    h = subgrafo_activo(g)
    clientes = [v for v in ORDEN if g.nodes[v]["tipo"] != "Servidor"]
    con_servicio = [v for v in clientes if v in h and not g.nodes[v]["sin_servicio"]]
    traf = [a["trafico"] for _, _, a in h.edges(data=True)]
    estados = [g.nodes[v]["estado"] for v in g]
    return {
        **{e: estados.count(e) for e in au.ESTADOS},
        "enlaces_activos": h.number_of_edges(),
        "componentes": nx.number_connected_components(h) if len(h) else 0,
        "servicio_pct": 100.0 * len(con_servicio) / len(clientes),
        "trafico_medio": float(np.mean(traf)) if traf else 0.0,
        "congestionados": int(sum(t >= dt.TRAFICO_ALTO for t in traf)),
        "trafico_total": float(sum(traf)),
    }
