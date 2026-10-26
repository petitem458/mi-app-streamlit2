"""Red neuronal de grafos que busca el camino óptimo (Bellman-Ford neuronal).

Idea (Veličković et al., 2020; Zhu et al., 2021): cada capa de la GNN es un paso
de paso de mensajes en el que cada nodo recibe de sus vecinos la distancia que
ellos conocen hacia el destino más el costo del enlace, y se queda con el mínimo:

    h_v^(k+1) = min( h_v^(k),  min_{u ∈ N(v)} [ ĉ_θ(v→u) + h_u^(k) ] ),  h_t^(0) = 0

El costo ĉ_θ de cada enlace lo calcula una red neuronal (función de mensaje)
a partir de las características de la arista φ(v→u): tráfico (Mbps), latencia,
tipo de enlace y estado/tipo del equipo de llegada.

    ĉ_θ(φ) = softplus( w2 · ReLU(W1·φ + b1) + b2 )

Tras |V| − 1 capas, h_v es la distancia estimada de v al destino y el vecino que
dio el mínimo es el siguiente salto; seguirlo desde el origen da la ruta.

ENTRENAMIENTO (aprendizaje supervisado por imitación de un experto):
    * Datos: escenarios aleatorios de la red fija, todos los pares (origen, destino).
    * Etiqueta: distancia óptima d*(v, t) calculada con Dijkstra sobre el costo real.
    * Pérdida: error cuadrático medio entre h_v (estimada) y d*(v, t), en unidades /100.
    * Gradiente: ∂h_v/∂ĉ(e) = 1 si e está en la ruta elegida desde v; luego
      retropropagación por la MLP. Optimizador Adam.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import numpy as np

from . import datos as dt
from . import red as rd

ESCALA = 100.0          # los costos (Mbps) se manejan en unidades de 100 Mbps


# --------------------------------------------------------------------------
# Preparación de un escenario
# --------------------------------------------------------------------------
@dataclass
class Escenario:
    nodos: list[str]
    origen_e: np.ndarray          # índice del nodo de salida de cada arista dirigida
    destino_e: np.ndarray         # índice del nodo de llegada
    phi: np.ndarray               # características (m × F)
    costo: np.ndarray             # costo real de cada arista dirigida (para métricas)
    d_opt: np.ndarray             # distancias óptimas n × n (inf si no hay ruta)


def preparar(g: nx.Graph) -> Escenario:
    dg = rd.grafo_costos(g)
    nodos = rd.ordenar_nodos(dg.nodes)
    idx = {v: i for i, v in enumerate(nodos)}
    aristas = list(dg.edges)
    phi = np.array([dt.caracteristicas_arista(g, u, v) for u, v in aristas]) if aristas else \
        np.zeros((0, len(dt.NOMBRES_CARACTERISTICAS)))
    costo = np.array([dg.edges[u, v]["costo"] for u, v in aristas]) / ESCALA
    n = len(nodos)
    d_opt = np.full((n, n), np.inf)
    for t, dist in nx.all_pairs_dijkstra_path_length(dg.reverse(copy=False), weight="costo"):
        for v, d in dist.items():
            d_opt[idx[v], idx[t]] = d / ESCALA
    return Escenario(nodos, np.array([idx[u] for u, _ in aristas], dtype=int),
                     np.array([idx[v] for _, v in aristas], dtype=int), phi, costo, d_opt)


# --------------------------------------------------------------------------
# Modelo
# --------------------------------------------------------------------------
def _softplus(z):
    return np.logaddexp(0.0, z)


def _sigmoide(z):
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class GNNRutas:
    entradas: int = len(dt.NOMBRES_CARACTERISTICAS)
    ocultas: int = 16
    semilla: int = 0
    params: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.params:
            rng = np.random.default_rng(self.semilla)
            lim = np.sqrt(6.0 / (self.entradas + self.ocultas))
            self.params = {"W1": rng.uniform(-lim, lim, (self.entradas, self.ocultas)),
                           "b1": np.zeros(self.ocultas),
                           "w2": rng.uniform(-lim, lim, self.ocultas) * 0.5,
                           "b2": np.array([-1.0])}

    # --- función de mensaje: costo aprendido de cada arista
    def costos(self, phi: np.ndarray):
        pre = phi @ self.params["W1"] + self.params["b1"]
        oculta = np.maximum(pre, 0.0)
        z = oculta @ self.params["w2"] + self.params["b2"][0]
        return _softplus(z), {"phi": phi, "pre": pre, "oculta": oculta, "z": z}

    # --- paso de mensajes con agregación mínima (Bellman-Ford)
    @staticmethod
    def propagar(n: int, origen_e, destino_e, c, capas: int | None = None, historia: bool = False):
        """Devuelve H (n × n: distancia estimada de v a t) y NH (siguiente salto)."""
        if n == 0:
            return np.zeros((0, 0)), np.zeros((0, 0), dtype=int), []
        C = np.full((n, n), np.inf)
        C[origen_e, destino_e] = c
        H = np.full((n, n), np.inf)
        np.fill_diagonal(H, 0.0)
        hist = [H.copy()] if historia else None
        for _ in range(capas or max(1, n - 1)):
            cand = (C[:, :, None] + H[None, :, :]).min(axis=1)        # mensajes de los vecinos
            nuevo = np.minimum(H, cand)
            if historia:
                hist.append(nuevo.copy())
            if np.array_equal(nuevo, H) and not historia:
                break
            H = nuevo
        NH = (C[:, :, None] + H[None, :, :]).argmin(axis=1)
        np.fill_diagonal(NH, -1)
        NH[~np.isfinite(H)] = -1
        return H, NH, hist

    def rutas(self, esc: Escenario):
        c, cache = self.costos(esc.phi)
        H, NH, _ = self.propagar(len(esc.nodos), esc.origen_e, esc.destino_e, c)
        return c, cache, H, NH

    # --- pérdida y gradientes
    def perdida_y_gradientes(self, esc: Escenario):
        c, cache, H, NH = self.rutas(esc)
        n = len(esc.nodos)
        mascara = np.isfinite(esc.d_opt) & ~np.eye(n, dtype=bool) & np.isfinite(H)
        N = max(1, mascara.sum())
        err = np.where(mascara, H - esc.d_opt, 0.0)
        perdida = float((err ** 2).sum() / N)
        g_h = 2.0 * err / N
        # gradiente de cada costo de arista: se acumula a lo largo de las rutas
        pos = {(int(a), int(b)): k for k, (a, b) in enumerate(zip(esc.origen_e, esc.destino_e))}
        g_c = np.zeros_like(c)
        for t in range(n):
            orden = np.argsort(-np.where(np.isfinite(H[:, t]), H[:, t], -1.0))
            flujo = g_h[:, t].copy()
            for v in orden:
                u = NH[v, t]
                if u < 0 or flujo[v] == 0.0:
                    continue
                g_c[pos[(int(v), int(u))]] += flujo[v]
                flujo[u] += flujo[v]
        g_z = g_c * _sigmoide(cache["z"])
        grads = {"w2": cache["oculta"].T @ g_z, "b2": np.array([g_z.sum()])}
        g_pre = np.outer(g_z, self.params["w2"]) * (cache["pre"] > 0)
        grads["W1"] = cache["phi"].T @ g_pre
        grads["b1"] = g_pre.sum(axis=0)
        return perdida, grads


class Adam:
    def __init__(self, params: dict, lr: float = 0.01, b1: float = 0.9, b2: float = 0.999):
        self.lr, self.b1, self.b2, self.t = lr, b1, b2, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def paso(self, params: dict, grads: dict) -> None:
        self.t += 1
        for k in params:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * grads[k] ** 2
            m_hat = self.m[k] / (1 - self.b1 ** self.t)
            v_hat = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)


# --------------------------------------------------------------------------
# Datos de entrenamiento
# --------------------------------------------------------------------------
def escenario_aleatorio(semilla: int, rng: np.random.Generator) -> nx.Graph:
    cruda = rd.generar_red(semilla, float(rng.uniform(0.0, 0.25)))
    if rng.random() < 0.5:
        return rd.simular(cruda)
    enlaces = list(cruda.edges)
    bloq = [str(v) for v in rng.choice(rd.ORDEN, size=int(rng.integers(0, 2)), replace=False)]
    bloq_e = [enlaces[i] for i in rng.choice(len(enlaces), size=int(rng.integers(0, 3)), replace=False)]
    libres = [v for v in rd.ORDEN if v not in bloq]
    fallas = [str(v) for v in rng.choice(libres, size=int(rng.integers(0, 2)), replace=False)]
    return rd.simular(cruda, bloq, bloq_e, fallas)


def generar_dataset(n_escenarios: int = 300, semilla: int = 2026) -> list[nx.Graph]:
    rng = np.random.default_rng(semilla)
    return [escenario_aleatorio(int(semilla * 1000 + i), rng) for i in range(n_escenarios)]


# --------------------------------------------------------------------------
# Evaluación
# --------------------------------------------------------------------------
def _camino(NH, o, t):
    ruta, v = [o], o
    while v != t:
        v = int(NH[v, t])
        if v < 0 or v in ruta:
            return None
        ruta.append(v)
    return ruta


def evaluar(modelo: GNNRutas | None, escenarios: list[Escenario]) -> dict:
    """% de rutas óptimas y sobrecosto. modelo=None evalúa la línea base de menos saltos."""
    optimas = total = 0
    sobrecostos, mae = [], []
    for esc in escenarios:
        n = len(esc.nodos)
        costo = np.full((n, n), np.inf)
        costo[esc.origen_e, esc.destino_e] = esc.costo
        if modelo is not None:
            c, _, H, NH = modelo.rutas(esc)
            mae.append(np.abs(c - esc.costo).mean() * ESCALA)
        else:   # menos saltos: Bellman-Ford con costo 1 por arista
            H, NH, _ = GNNRutas.propagar(n, esc.origen_e, esc.destino_e, np.ones(len(esc.costo)))
        for o in range(n):
            for t in range(n):
                if o == t or not np.isfinite(esc.d_opt[o, t]):
                    continue
                total += 1
                ruta = _camino(NH, o, t)
                if ruta is None:
                    sobrecostos.append(1.0)
                    continue
                c_ruta = sum(costo[a, b] for a, b in zip(ruta, ruta[1:]))
                rel = (c_ruta - esc.d_opt[o, t]) / esc.d_opt[o, t]
                sobrecostos.append(rel)
                optimas += rel < 1e-9
    return {"rutas": total, "optimas": optimas / max(1, total),
            "sobrecosto_medio": float(np.mean(sobrecostos)) if sobrecostos else 0.0,
            "sobrecosto_max": float(np.max(sobrecostos)) if sobrecostos else 0.0,
            "error_costo": float(np.mean(mae)) if mae else None}


# --------------------------------------------------------------------------
# Entrenamiento
# --------------------------------------------------------------------------
@dataclass
class ResultadoEntrenamiento:
    modelo: GNNRutas
    historial: list[dict]
    prueba: dict
    linea_base: dict
    sin_entrenar: dict
    segundos: float
    n_escenarios: tuple[int, int, int]


def entrenar(grafos: list[nx.Graph], ocultas: int = 32, epocas: int = 300, lr: float = 0.01,
             semilla: int = 0) -> ResultadoEntrenamiento:
    """División por escenarios 70/15/15; se conserva el modelo con más rutas óptimas en validación."""
    inicio = time.perf_counter()
    escs = [e for e in (preparar(g) for g in grafos) if len(e.nodos) >= 2]
    n = len(escs)
    i_tr, i_va = int(0.7 * n), int(0.85 * n)
    tr, va, te = escs[:i_tr], escs[i_tr:i_va], escs[i_va:]
    modelo = GNNRutas(ocultas=ocultas, semilla=semilla)
    sin_entrenar = evaluar(modelo, te)
    opt = Adam(modelo.params, lr=lr)
    historial, mejor, mejores = [], -1.0, None
    for epoca in range(1, epocas + 1):
        total = {k: np.zeros_like(v) for k, v in modelo.params.items()}
        perdidas = []
        for esc in tr:
            p, g = modelo.perdida_y_gradientes(esc)
            perdidas.append(p)
            for k in total:
                total[k] += g[k] / len(tr)
        opt.paso(modelo.params, total)
        if epoca % 5 == 0 or epoca == 1:
            perd_va = float(np.mean([modelo.perdida_y_gradientes(e)[0] for e in va]))
            m_va = evaluar(modelo, va)
            historial.append({"epoca": epoca, "perdida": float(np.mean(perdidas)), "perdida_val": perd_va,
                              "optimas_val": m_va["optimas"], "sobrecosto_val": m_va["sobrecosto_medio"]})
            if m_va["optimas"] > mejor:
                mejor, mejores = m_va["optimas"], {k: v.copy() for k, v in modelo.params.items()}
    modelo.params = mejores
    return ResultadoEntrenamiento(modelo, historial, evaluar(modelo, te), evaluar(None, te),
                                  sin_entrenar, time.perf_counter() - inicio, (len(tr), len(va), len(te)))


# --------------------------------------------------------------------------
# Inferencia sobre una red concreta
# --------------------------------------------------------------------------
def buscar_ruta(modelo: GNNRutas, g: nx.Graph, origen: str, destino: str) -> dict:
    """Ruta de la GNN, distancias por capa (para visualizar) y costos aprendidos."""
    inicio = time.perf_counter()
    esc = preparar(g)
    idx = {v: i for i, v in enumerate(esc.nodos)}
    if origen not in idx or destino not in idx:
        return {"ruta": None, "ms": 0.0, "motivo": "El origen o el destino está bloqueado."}
    c, _ = modelo.costos(esc.phi)
    n = len(esc.nodos)
    H, NH, hist = GNNRutas.propagar(n, esc.origen_e, esc.destino_e, c, historia=True)
    camino = _camino(NH, idx[origen], idx[destino])
    ms = (time.perf_counter() - inicio) * 1000
    t = idx[destino]
    capas = [{v: float(h[idx[v], t]) * ESCALA for v in esc.nodos} for h in hist]
    aprendido = {(esc.nodos[a], esc.nodos[b]): float(x) * ESCALA
                 for a, b, x in zip(esc.origen_e, esc.destino_e, c)}
    return {"ruta": [esc.nodos[i] for i in camino] if camino else None, "ms": ms,
            "estimado": float(H[idx[origen], t]) * ESCALA, "capas": capas, "costos_aprendidos": aprendido,
            "motivo": None if camino else "No existe ruta entre los dos equipos."}


# --------------------------------------------------------------------------
# Persistencia
# --------------------------------------------------------------------------
def guardar(r: ResultadoEntrenamiento, ruta: Path) -> None:
    np.savez(ruta.with_suffix(".npz"), **r.modelo.params)
    meta = {"ocultas": r.modelo.ocultas, "historial": r.historial, "prueba": r.prueba,
            "linea_base": r.linea_base, "sin_entrenar": r.sin_entrenar, "segundos": r.segundos,
            "n_escenarios": r.n_escenarios}
    ruta.with_suffix(".json").write_text(json.dumps(meta, indent=1), encoding="utf-8")


def cargar(ruta: Path) -> ResultadoEntrenamiento:
    meta = json.loads(ruta.with_suffix(".json").read_text(encoding="utf-8"))
    with np.load(ruta.with_suffix(".npz")) as p:
        params = {k: p[k] for k in p.files}
    modelo = GNNRutas(ocultas=meta["ocultas"], params=params)
    return ResultadoEntrenamiento(modelo, meta["historial"], meta["prueba"], meta["linea_base"],
                                  meta["sin_entrenar"], meta["segundos"], tuple(meta["n_escenarios"]))
