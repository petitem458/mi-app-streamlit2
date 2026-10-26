"""Pruebas de la red fija, el recorrido BFS, la simulación y la GNN de rutas."""

import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import automata as au  # noqa: E402
from core import gnn  # noqa: E402
from core import red as rd  # noqa: E402

MODELO = gnn.cargar(Path(__file__).resolve().parents[1] / "modelos" / "gnn_rutas").modelo


def test_inventario_fijo():
    g = rd.construir_topologia(5)
    tipos = [g.nodes[v]["tipo"] for v in g]
    assert [tipos.count(t) for t in ("Servidor", "PC", "Portátil", "Impresora", "IoT")] == [2, 5, 5, 2, 4]
    assert g.number_of_edges() == 24 and nx.is_connected(g)


def test_cada_nodo_tiene_dos_o_tres_caminos():
    g = rd.construir_topologia(5)
    assert all(2 <= g.degree(v) <= 3 for v in g)
    assert nx.is_biconnected(g)                      # sin puntos únicos de falla
    assert nx.edge_connectivity(g) == 2              # dos caminos independientes como mínimo


def test_aristas_con_trafico_en_mbps():
    from core import datos as dt
    g = rd.construir_topologia(5)
    for _, _, a in g.edges(data=True):
        lo, hi = dt.ENLACES[a["clase"]]["trafico"]
        assert lo <= a["trafico"] <= hi
        assert set(a) == {"clase", "activa", "latencia", "trafico"}
    s = rd.simular(rd.generar_red(5, 0.0))
    u, v = "PC1", "S1"
    if s.nodes[v]["estado"] != au.ANOMALO:
        assert dt.costo_real(s, u, v) == s.edges[u, v]["trafico"]   # el costo es el tráfico (Mbps)


def test_misma_topologia_distinto_trafico():
    a, b = rd.generar_red(1), rd.generar_red(2)
    assert set(a.edges) == set(b.edges)
    assert [a.edges[e]["trafico"] for e in a.edges] != [b.edges[e]["trafico"] for e in a.edges]


def test_recorrido_bfs():
    g = rd.simular(rd.generar_red(5, 0.0))
    visitados = [o for o in g.graph["orden_recorrido"] if o["paso"] is not None]
    assert [o["nodo"] for o in visitados[:2]] == ["S1", "S2"]
    assert [o["nivel"] for o in visitados] == sorted(o["nivel"] for o in visitados)
    assert g.nodes["IoT2"]["padre"] == "PC3" and g.nodes["IoT2"]["nivel"] == 2


def test_bloqueo_de_servidor():
    cruda = rd.generar_red(5, 0.0)
    s = rd.simular(cruda, ["S1"])
    assert s.nodes["S1"]["estado"] == au.BLOQUEADO
    assert not s.nodes["PC1"]["sin_servicio"]          # sigue llegando a S2 por PC4
    assert rd.ruta_optima(s, "PT1", "S1") is None
    assert "bloqueo_manual" not in cruda.nodes["S1"]   # la red original no cambia


def test_ruta_optima_esquiva_bloqueados():
    s = rd.simular(rd.generar_red(5, 0.0), ["PC1"])
    ruta = rd.ruta_optima(s, "PT1", "S2")
    assert ruta[0] == "PT1" and ruta[-1] == "S2" and "PC1" not in ruta


def test_gradiente_analitico_coincide_con_numerico():
    esc = gnn.preparar(rd.simular(rd.generar_red(3)))
    m = gnn.GNNRutas(semilla=1)
    _, grads = m.perdida_y_gradientes(esc)
    eps = 1e-6
    for clave in ("W1", "w2", "b1"):
        p = m.params[clave]
        idx = tuple(np.unravel_index(np.argmax(np.abs(grads[clave])), p.shape))
        orig = p[idx]
        p[idx] = orig + eps
        mas, _ = m.perdida_y_gradientes(esc)
        p[idx] = orig - eps
        menos, _ = m.perdida_y_gradientes(esc)
        p[idx] = orig
        assert abs((mas - menos) / (2 * eps) - grads[clave][idx]) < 1e-5


def test_gnn_entrenada_encuentra_rutas_optimas():
    escs = [gnn.preparar(rd.simular(rd.generar_red(s))) for s in range(900, 910)]
    m = gnn.evaluar(MODELO, escs)
    assert m["optimas"] > 0.95 and m["sobrecosto_medio"] < 0.01
    assert m["optimas"] > gnn.evaluar(None, escs)["optimas"]


def test_buscar_ruta_y_rendimiento():
    g = rd.simular(rd.generar_red(5))
    inicio = time.perf_counter()
    r = gnn.buscar_ruta(MODELO, g, "PT1", "S2")
    assert time.perf_counter() - inicio < 2.0                       # RNF-02
    assert r["ruta"][0] == "PT1" and r["ruta"][-1] == "S2"
    assert all(g.has_edge(a, b) for a, b in zip(r["ruta"], r["ruta"][1:]))
