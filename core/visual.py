"""Grafo interactivo con Pyvis: las aristas muestran el tráfico de cada enlace."""

from __future__ import annotations

import json
import math

import networkx as nx
from pyvis.network import Network

from .datos import TRAFICO_ALTO, TRAFICO_MEDIO

COLORES = {
    "Normal": "#22C55E", "Anómalo": "#EF4444", "Bloqueado": "#64748B",
    "pendiente": "#1E293B", "actual": "#2DD4BF", "ruta": "#22D3EE", "falla": "#E879F9",
    "sin_servicio": "#F59E0B", "cambio": "#FACC15", "fondo": "#0B1220", "texto": "#E6EDF3",
    "traf_bajo": "#3B82F6", "traf_medio": "#F59E0B", "traf_alto": "#EF4444", "inactiva": "#475569",
}
FORMAS = {"Servidor": "diamond", "PC": "dot", "Portátil": "dot", "Impresora": "square", "IoT": "triangle"}
TAMANO = {"Servidor": 34, "PC": 22, "Portátil": 18, "Impresora": 18, "IoT": 18}

# Posiciones fijas (x, y): los 16 equipos del anillo sobre una elipse y los servidores al centro.
from .red import ANILLO  # noqa: E402

POSICIONES = {"S1": (-150, 0), "S2": (150, 0)}
for _i, _v in enumerate(ANILLO):
    _t = math.radians(-90 + 22.5 * _i)
    POSICIONES[_v] = (round(430 * math.cos(_t)), round(300 * math.sin(_t)))
# Las impresoras se desplazan hacia afuera para que no queden sobre los caminos PT1–PC2 y PT3–IoT3.
POSICIONES["IM1"], POSICIONES["IM2"] = (-520, -140), (520, -140)

OPCIONES = {
    "physics": {"enabled": False},
    "interaction": {"hover": True, "tooltipDelay": 80, "zoomView": True, "dragNodes": True},
    "nodes": {"font": {"face": "Inter, Segoe UI, sans-serif", "size": 14, "color": COLORES["texto"],
                       "strokeWidth": 3, "strokeColor": COLORES["fondo"]},
              "borderWidth": 2, "shadow": {"enabled": True, "size": 6, "x": 0, "y": 2}},
    "edges": {"smooth": False,
              "font": {"size": 13, "color": "#CBD5E1", "strokeWidth": 3, "strokeColor": COLORES["fondo"],
                       "align": "top"}},
}


def color_trafico(trafico: float) -> str:
    """Azul < 30 Mbps ≤ ámbar < 60 Mbps ≤ rojo."""
    if trafico < TRAFICO_MEDIO:
        return COLORES["traf_bajo"]
    return COLORES["traf_medio"] if trafico < TRAFICO_ALTO else COLORES["traf_alto"]


def ancho_trafico(trafico: float) -> float:
    """Grosor de la arista proporcional al tráfico (Mbps)."""
    return 1.5 + 7.0 * min(trafico / 100.0, 1.0)


def _tooltip(v: str, d: dict) -> str:
    lineas = [f"{v} · {d['tipo']}", d["rol"], f"Área: {d['area']}", f"Estado (AFD): {d.get('estado', '—')}"]
    if d.get("motivo_bloqueo"):
        lineas.append(f"Motivo: {d['motivo_bloqueo']}")
    lineas.append(f"CPU: {d['cpu']:.1f} %")
    if d.get("cadena") is not None:
        lineas.append("Cadena: " + " ".join(d["cadena"]))
    if d.get("nivel") is not None:
        lineas.append(f"BFS: nivel {d['nivel']}" + (f", desde {d['padre']}" if d.get("padre") else ""))
    if d.get("falla_inyectada"):
        lineas.append("⚡ Falla inyectada")
    if d.get("sin_servicio"):
        lineas.append("⚠ Sin ruta hacia un servidor")
    return "\n".join(lineas)


def grafo_html(g: nx.Graph, altura: int = 560, modo: str = "estado", ruta: list[str] | None = None,
               cambios: set[str] | None = None, visitados: list[str] | None = None,
               etiquetas: bool = True) -> str:
    """modo: 'estado' (color por estado del AFD) o 'recorrido' (avance del BFS)."""
    cambios = cambios or set()
    en_ruta = set(zip(ruta, ruta[1:])) | set(zip(ruta[1:], ruta)) if ruta else set()
    red = Network(height=f"{altura}px", width="100%", bgcolor=COLORES["fondo"],
                  font_color=COLORES["texto"], cdn_resources="in_line")
    for v, d in g.nodes(data=True):
        color = COLORES[d.get("estado", "Normal")]
        borde, ancho = color, 2
        if modo == "recorrido":
            if visitados is not None and v not in visitados:
                color = borde = COLORES["pendiente"]
            else:
                color = borde = COLORES[d.get("estado_afd") or "Normal"]
                if visitados and v == visitados[-1]:
                    borde, ancho = COLORES["actual"], 6
        else:
            if d.get("falla_inyectada") and d.get("estado") != "Bloqueado":
                borde, ancho = COLORES["falla"], 4
            if d.get("sin_servicio"):
                borde, ancho = COLORES["sin_servicio"], 4
            if v in cambios:
                borde, ancho = COLORES["cambio"], 6
            if ruta and v in (ruta[0], ruta[-1]):
                borde, ancho = COLORES["ruta"], 6
        x, y = POSICIONES[v]
        red.add_node(v, label=v, title=_tooltip(v, d), shape=FORMAS[d["tipo"]], size=TAMANO[d["tipo"]],
                     borderWidth=ancho, x=x, y=y, physics=False,
                     color={"background": color, "border": borde,
                            "highlight": {"background": color, "border": "#FFFFFF"},
                            "hover": {"background": color, "border": "#FFFFFF"}})
    for u, v, a in g.edges(data=True):
        activa = a["activa"]
        titulo = (f"{u} — {v} ({a['clase']})\nTráfico (peso): {a['trafico']:.1f} Mbps\n"
                  f"Latencia: {a['latencia']} ms\nEstado: {'Activo' if activa else 'Inactivo'}")
        if (u, v) in en_ruta:
            color, ancho = COLORES["ruta"], 9
        elif not activa:
            color, ancho = COLORES["inactiva"], 1.5
        else:
            color, ancho = color_trafico(a["trafico"]), ancho_trafico(a["trafico"])
        red.add_edge(u, v, title=titulo, width=ancho, color=color, dashes=not activa,
                     label=f"{a['trafico']:.0f} Mbps" if (etiquetas and activa) else None)
    red.set_options(json.dumps(OPCIONES))
    html = red.generate_html(notebook=False).replace("border: 1px solid lightgray;", "border: none;")
    estilo = ("<style>body{margin:0;background:%s;} .card{border:none!important;background:%s!important;}"
              "div.vis-tooltip{font-family:Inter,Segoe UI,sans-serif;background:#111A2B;color:#E6EDF3;"
              "border:1px solid #2DD4BF;border-radius:8px;padding:8px 10px;white-space:pre-line;font-size:13px;}"
              "#mynetwork{border:0!important;outline:0!important;} html,body{overflow:hidden;}</style>"
              % (COLORES["fondo"], COLORES["fondo"]))
    ajuste = ("<script>(function(){var t=null;function fit(){if(window.network){network.redraw();"
              "network.fit({animation:false});}}new ResizeObserver(function(){clearTimeout(t);"
              "t=setTimeout(fit,120);}).observe(document.body);setTimeout(fit,60);})();</script>")
    html = html.replace("</body>", ajuste + "</body>", 1)
    return html.replace("</head>", estilo + "</head>", 1)
