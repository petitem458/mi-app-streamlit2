"""SoftCast · Red de computadores: estados con AFD y camino óptimo con una GNN.

Prototipo funcional (Streamlit):
    * Red fija: los 18 equipos son los nodos y las 24 aristas los caminos de comunicación,
      ponderados por el tráfico medido en Mbps; cada equipo tiene 2 o 3 caminos.
    * Recorrido en anchura (BFS) desde los servidores.
    * AFD de tres estados (Normal, Anómalo, Bloqueado) para cada dispositivo.
    * GNN tipo Bellman-Ford neuronal que encuentra el camino óptimo entre dos equipos.
    * Simulación de bloqueo y habilitación de nodos y enlaces e inyección de fallas.

Ejecutar:  streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from core import automata as au
from core import datos as dt
from core import gnn
from core import red as rd
from core import visual

st.set_page_config(page_title="SoftCast · Camino óptimo", page_icon="🛰️", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
html, body, [class*="css"], .stMarkdown, .stText { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1400px; }
.hero { background: linear-gradient(120deg, #0F2A3D 0%, #111A2B 55%, #1A1530 100%);
        border: 1px solid #1F2B40; border-radius: 16px; padding: 22px 26px; margin-bottom: 18px; }
.hero h1 { font-size: 1.7rem; margin: 0 0 4px 0; color: #F1F5F9; font-weight: 700; }
.hero p { color: #94A3B8; margin: 0; font-size: 0.95rem; }
.hero .tag { display:inline-block; background:#0E3B3A; color:#5EEAD4; border:1px solid #115E59;
             padding:2px 10px; border-radius:999px; font-size:0.75rem; margin-right:6px; }
.kpi { background:#111A2B; border:1px solid #1F2B40; border-radius:14px; padding:14px 16px; }
.kpi .lbl { color:#94A3B8; font-size:0.76rem; text-transform:uppercase; letter-spacing:.06em; }
.kpi .val { color:#F1F5F9; font-size:1.6rem; font-weight:700; line-height:1.2; }
.kpi .sub { color:#64748B; font-size:0.78rem; }
.kpi.ok .val { color:#4ADE80; } .kpi.alerta .val { color:#F87171; } .kpi.gris .val { color:#94A3B8; }
.kpi.info .val { color:#5EEAD4; } .kpi.ambar .val { color:#FBBF24; }
.leyenda { display:flex; flex-wrap:wrap; gap:14px; font-size:0.85rem; color:#CBD5E1; margin:6px 0 10px; }
.leyenda span.c { display:inline-block; width:12px; height:12px; border-radius:3px; margin-right:6px; vertical-align:-1px; }
.leyenda span.l { display:inline-block; width:22px; height:4px; border-radius:2px; margin-right:6px; vertical-align:3px; }
.chip { display:inline-block; padding:3px 12px; border-radius:999px; font-weight:600; font-size:0.9rem; }
.chip.Normal { background:#052E1A; color:#4ADE80; border:1px solid #14532D; }
.chip.Anómalo { background:#3B0A0A; color:#F87171; border:1px solid #7F1D1D; }
.chip.Bloqueado { background:#1E293B; color:#CBD5E1; border:1px solid #475569; }
.ruta { font-family:'JetBrains Mono', monospace; font-size:1.0rem; color:#22D3EE; }
.mono { font-family:'JetBrains Mono', monospace; font-size:0.86rem; }
.card { background:#111A2B; border:1px solid #1F2B40; border-radius:14px; padding:16px 18px; }
div[data-testid="stTabs"] button p { font-size:0.95rem; font-weight:600; }
</style>
""", unsafe_allow_html=True)

CARPETA_MODELOS = Path(__file__).parent / "modelos"
COLOR_ESTADO = {"Normal": "color:#4ADE80", "Anómalo": "color:#F87171;font-weight:600",
                "Bloqueado": "color:#94A3B8;font-weight:600"}
NODOS = rd.ORDEN


@st.cache_resource(show_spinner="Cargando la GNN entrenada…")
def obtener_modelo() -> gnn.ResultadoEntrenamiento:
    try:
        return gnn.cargar(CARPETA_MODELOS / "gnn_rutas")
    except (FileNotFoundError, KeyError, ValueError):
        return gnn.entrenar(gnn.generar_dataset(300, 2026))


@st.cache_data
def red_cruda(semilla: int, fallas: float):
    return rd.generar_red(semilla, fallas)


def kpi(etiqueta, valor, sub="", clase=""):
    return (f'<div class="kpi {clase}"><div class="lbl">{etiqueta}</div><div class="val">{valor}</div>'
            f'<div class="sub">{sub}</div></div>')


def leyenda(recorrido: bool = False) -> None:
    c = visual.COLORES
    nodos = (f'<span><span class="c" style="background:{c["Normal"]}"></span>Normal</span>'
             f'<span><span class="c" style="background:{c["Anómalo"]}"></span>Anómalo</span>'
             f'<span><span class="c" style="background:{c["Bloqueado"]}"></span>Bloqueado</span>')
    if recorrido:
        extra = (f'<span><span class="c" style="background:{c["pendiente"]};border:1px solid #475569"></span>Aún no visitado</span>'
                 f'<span><span class="c" style="background:transparent;border:3px solid {c["actual"]}"></span>Nodo actual</span>')
    else:
        extra = (f'<span><span class="l" style="background:{c["traf_bajo"]}"></span>Tráfico &lt; 30 Mbps</span>'
                 f'<span><span class="l" style="background:{c["traf_medio"]}"></span>30–60 Mbps</span>'
                 f'<span><span class="l" style="background:{c["traf_alto"]}"></span>≥ 60 Mbps</span>'
                 f'<span><span class="l" style="background:{c["ruta"]}"></span>Ruta de la GNN</span>'
                 f'<span><span class="c" style="background:transparent;border:3px solid {c["falla"]}"></span>Falla inyectada</span>'
                 f'<span><span class="c" style="background:transparent;border:3px solid {c["cambio"]}"></span>Cambió de estado</span>')
    st.markdown('<div class="leyenda">' + nodos + extra +
                '<span>◆ Servidor · ● PC / Portátil · ■ Impresora · ▲ IoT · grosor y número de la arista = tráfico (Mbps)</span></div>',
                unsafe_allow_html=True)


def tema(fig: go.Figure, alto: int = 330) -> go.Figure:
    fig.update_layout(template="plotly_dark", height=alto, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=40, b=30),
                      font=dict(family="Inter, sans-serif", color="#CBD5E1"),
                      legend=dict(orientation="h", y=-0.3, x=0))
    fig.update_xaxes(gridcolor="#1F2B40")
    fig.update_yaxes(gridcolor="#1F2B40")
    return fig


def tabla_enlaces(g) -> pd.DataFrame:
    return pd.DataFrame([{"Camino": f"{u} — {v}", "Tipo": a["clase"], "Tráfico (Mbps)": a["trafico"],
                          "Latencia (ms)": a["latencia"], "Activo": "Sí" if a["activa"] else "No"}
                         for u, v, a in g.edges(data=True)])


def comparar_rutas(g, origen, destino, modelo):
    """Ruta de la GNN frente al óptimo (Dijkstra) y a la de menos saltos."""
    r = gnn.buscar_ruta(modelo, g, origen, destino)
    opt = rd.ruta_optima(g, origen, destino)
    filas = []
    for nombre, ruta in (("GNN (red neuronal)", r["ruta"]), ("Óptimo · Dijkstra", opt),
                         ("Menos saltos · BFS", rd.ruta_saltos(g, origen, destino))):
        if ruta is None:
            filas.append({"Método": nombre, "Ruta": "—", "Saltos": None, "Tráfico acumulado (Mbps)": None, "¿Óptima?": "—"})
            continue
        costo = rd.costo_ruta(g, ruta)
        filas.append({"Método": nombre, "Ruta": " → ".join(ruta), "Saltos": len(ruta) - 1,
                      "Tráfico acumulado (Mbps)": costo,
                      "¿Óptima?": "Sí" if opt and abs(costo - rd.costo_ruta(g, opt)) < 1e-6 else "No"})
    return r, pd.DataFrame(filas)


# --------------------------------------------------------------------------
# Barra lateral
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🛰️ SoftCast NOC")
    st.caption("Lenguajes y Autómatas · UdeC Fusagasugá · 802M")
    st.markdown("### Red fija · 18 equipos")
    st.markdown("2 servidores · 5 PC · 5 portátiles · 2 impresoras · 4 IoT · 24 caminos (2 o 3 por equipo)")
    semilla = st.number_input("Semilla aleatoria", 0, 9999, 3, step=1,
                              help="Cambia el tráfico, la CPU, las latencias y los eventos; la topología es fija.")
    prop_fallas = st.slider("Proporción de fallas iniciales", 0.0, 0.3, 0.12, step=0.02)
    st.markdown("### Ruta")
    origen = st.selectbox("Origen", NODOS, index=NODOS.index("PC1"))
    destino = st.selectbox("Destino", NODOS, index=NODOS.index("S2"))
    st.caption("GNN Bellman-Ford neuronal · función de mensaje MLP de 32 neuronas · 300 épocas.")
    st.divider()
    st.caption("Paula Andrea Merchán Suaza · Victor Hernando Molina Rodríguez")

resultado = obtener_modelo()
modelo = resultado.modelo
cruda = red_cruda(int(semilla), float(prop_fallas))
base = rd.simular(cruda)
m_base = rd.metricas_red(base)
clave = (int(semilla), float(prop_fallas))
if st.session_state.get("clave_red") != clave:
    st.session_state["clave_red"] = clave
    for k in ("sim_nodos", "sim_enlaces", "sim_fallas", "sim_hab"):
        st.session_state[k] = []

st.markdown("""
<div class="hero">
  <span class="tag">Grafo · tráfico</span><span class="tag">BFS</span><span class="tag">AFD · 3 estados</span><span class="tag">GNN · camino óptimo</span>
  <h1>SoftCast: estado de los equipos y camino óptimo en la red</h1>
  <p>Los nodos son los 18 equipos y las aristas los caminos por donde se comunican, con su tráfico medido en Mbps.
  Un recorrido BFS visita cada equipo y el AFD decide si está Normal, Anómalo o Bloqueado; la red neuronal de grafos
  busca la ruta de menor tráfico acumulado esquivando equipos bloqueados.</p>
</div>""", unsafe_allow_html=True)

c = st.columns(6)
c[0].markdown(kpi("Normal", m_base["Normal"], "de 18 equipos", "ok"), unsafe_allow_html=True)
c[1].markdown(kpi("Anómalo", m_base["Anómalo"], "en observación", "alerta"), unsafe_allow_html=True)
c[2].markdown(kpi("Bloqueado", m_base["Bloqueado"], "decididos por el AFD", "gris"), unsafe_allow_html=True)
c[3].markdown(kpi("Tráfico total", f"{m_base['trafico_total']:.0f}", "Mbps en la red", "info"),
              unsafe_allow_html=True)
c[4].markdown(kpi("Tráfico medio", f"{m_base['trafico_medio']:.1f}", "Mbps por camino"), unsafe_allow_html=True)
c[5].markdown(kpi("Congestionados", m_base["congestionados"], "caminos ≥ 60 Mbps", "ambar"), unsafe_allow_html=True)
st.write("")

tab_red, tab_ruta, tab_bfs, tab_sim, tab_afd, tab_gnn = st.tabs([
    "🌐 Red y tráfico", "🧭 Camino óptimo (GNN)", "🔎 Recorrido BFS", "🛠️ Simulación",
    "🧩 Autómata (AFD)", "🧠 Entrenamiento de la GNN"])

# ----------------------------------------------------------------- Red y tráfico
with tab_red:
    etiquetas = st.toggle("Mostrar el tráfico (Mbps) sobre las aristas", value=True)
    leyenda()
    components.html(visual.grafo_html(base, 560, etiquetas=etiquetas), height=575)
    izq, der = st.columns([1, 1], gap="large")
    with izq:
        st.markdown("#### Caminos (aristas = tráfico en Mbps)")
        st.dataframe(tabla_enlaces(base).sort_values("Tráfico (Mbps)", ascending=False)
                     .style.format({"Tráfico (Mbps)": "{:.1f}", "Latencia (ms)": "{:.1f}"})
                     .bar(subset=["Tráfico (Mbps)"], color="#7F1D1D", vmin=0, vmax=150),
                     hide_index=True, height=420, width="stretch")
    with der:
        st.markdown("#### Equipos (nodos)")
        st.dataframe(pd.DataFrame([{"Nodo": v, "Tipo": base.nodes[v]["tipo"], "Caminos": base.degree(v),
                                    "CPU (%)": base.nodes[v]["cpu"],
                                    "Cadena de eventos": " ".join(base.nodes[v]["cadena"]),
                                    "Estado (AFD)": base.nodes[v]["estado"],
                                    "Motivo": base.nodes[v]["motivo_bloqueo"] or ""} for v in NODOS])
                     .style.map(lambda x: COLOR_ESTADO.get(x, ""), subset=["Estado (AFD)"])
                     .format({"CPU (%)": "{:.1f}"}), hide_index=True, height=420, width="stretch")

# ----------------------------------------------------------------- Camino óptimo
with tab_ruta:
    if origen == destino:
        st.warning("Elija un origen y un destino distintos en la barra lateral.")
    else:
        r, comp = comparar_rutas(base, origen, destino, modelo)
        if r["ruta"]:
            st.markdown(f'<div class="card">Ruta encontrada por la GNN de <b>{origen}</b> a <b>{destino}</b>: '
                        f'<span class="ruta">{" → ".join(r["ruta"])}</span><br>Tráfico acumulado estimado por la red: '
                        f'<b>{r["estimado"]:.1f} Mbps</b> · real: <b>{rd.costo_ruta(base, r["ruta"]):.1f} Mbps</b> · '
                        f'inferencia en <b>{r["ms"]:.1f} ms</b></div>', unsafe_allow_html=True)
        else:
            st.error(r["motivo"])
        st.write("")
        leyenda()
        components.html(visual.grafo_html(base, 540, ruta=r["ruta"]), height=555)
        izq, der = st.columns([3, 2], gap="large")
        with izq:
            st.markdown("#### Comparación de rutas")
            st.dataframe(comp.style.format({"Tráfico acumulado (Mbps)": "{:.1f}", "Saltos": "{:.0f}"}, na_rep="—")
                         .map(lambda x: "color:#4ADE80" if x == "Sí" else ("color:#F87171" if x == "No" else ""),
                              subset=["¿Óptima?"]), hide_index=True, width="stretch")
            if r["ruta"]:
                st.markdown("#### Costo de cada camino de la ruta (Mbps)")
                filas = [{"Camino": f"{a} → {b}", "Tráfico (Mbps)": base.edges[a, b]["trafico"],
                          "Llegada": base.nodes[b]["estado"], "Costo real (Mbps)": dt.costo_real(base, a, b),
                          "Costo aprendido (GNN)": r["costos_aprendidos"][(a, b)]}
                         for a, b in zip(r["ruta"], r["ruta"][1:])]
                st.dataframe(pd.DataFrame(filas).style.format({"Tráfico (Mbps)": "{:.1f}", "Costo real (Mbps)": "{:.1f}",
                                                               "Costo aprendido (GNN)": "{:.1f}"}),
                             hide_index=True, width="stretch")
                st.caption("Costo real = tráfico del camino (Mbps) + 30 Mbps si el equipo de llegada está Anómalo. "
                           "La GNN no conoce esta fórmula: la aprendió de las rutas óptimas.")
        with der:
            st.markdown("#### Paso de mensajes capa por capa")
            capas = r.get("capas") or []
            if capas:
                k = st.slider("Capa de la GNN", 0, len(capas) - 1, min(3, len(capas) - 1))
                st.caption(f"Distancia estimada de cada equipo hasta {destino} después de {k} capas "
                           "(∞ = el mensaje aún no ha llegado).")
                dist = pd.DataFrame([{"Equipo": v, "Distancia estimada": d} for v, d in capas[k].items()])
                st.dataframe(dist.style.format({"Distancia estimada": lambda x: "∞" if x == float("inf") else f"{x:.1f}"}),
                             hide_index=True, height=420, width="stretch")

# ----------------------------------------------------------------- Recorrido BFS
with tab_bfs:
    orden = base.graph["orden_recorrido"]
    visitables = [o for o in orden if o["paso"] is not None]
    paso = st.slider("Paso del recorrido", 1, len(visitables), len(visitables))
    actual = visitables[paso - 1]
    d = base.nodes[actual["nodo"]]
    st.markdown(f'<div class="card">Paso <b>{paso}</b> · visitando <b>{actual["nodo"]}</b> ({d["tipo"]}) · nivel '
                f'<b>{actual["nivel"]}</b>' + (f' · desde <b>{actual["padre"]}</b>' if actual["padre"] else " · origen")
                + f'<br><span class="mono">{" ".join(d["cadena"])}</span> → AFD: '
                f'<span class="chip {d["estado_afd"]}">{d["estado_afd"]}</span>'
                + ("" if actual["expande"] else " · <b>no se exploran sus vecinos</b>") + '</div>',
                unsafe_allow_html=True)
    leyenda(recorrido=True)
    components.html(visual.grafo_html(base, 520, "recorrido", visitados=[o["nodo"] for o in visitables[:paso]],
                                      etiquetas=False), height=535)
    izq, der = st.columns([2, 3], gap="large")
    with izq:
        st.markdown("#### Algoritmo de recorrido (BFS)")
        st.code("""ALGORITMO RecorrerRed(G, M)
  Q ← cola con los servidores no bloqueados (nivel 0)
  mientras Q no esté vacía:
     u ← desencolar(Q)
     estado[u] ← δ̂(Normal, cadena(u))      ▸ AFD
     si estado[u] = Bloqueado: continuar
     para cada vecino v de u (orden fijo):
        si enlace activo y v no visitado:
           nivel[v] ← nivel[u] + 1; encolar(Q, v)
  no visitados → sin servicio → AFD
Complejidad: O(|V| + |E|)""", language=None)
    with der:
        st.markdown("#### Orden de visita")
        st.dataframe(pd.DataFrame([{"Paso": o["paso"] or "—", "Nodo": o["nodo"],
                                    "Nivel": "—" if o["nivel"] is None else o["nivel"], "Desde": o["padre"] or "—",
                                    "AFD": o["estado"]} for o in orden])
                     .style.map(lambda x: COLOR_ESTADO.get(x, ""), subset=["AFD"]),
                     hide_index=True, height=400, width="stretch")

# ----------------------------------------------------------------- Simulación
with tab_sim:
    st.markdown("#### Escenario de simulación")
    st.caption("Bloquee (BLQ) o habilite (HAB) equipos, bloquee caminos o inyecte fallas. Los Mbps se redistribuyen, "
               "el BFS y el AFD reevalúan la red y la GNN recalcula la ruta "
               f"de {origen} a {destino}.")
    enlaces = {f"{u} — {v}": (u, v) for u, v in base.edges}
    bloq_afd = [v for v in NODOS if base.nodes[v]["motivo_bloqueo"]]
    c1, c2, c3, c4 = st.columns(4)
    sel_n = c1.multiselect("🚫 Bloquear equipos (BLQ)", NODOS, key="sim_nodos", placeholder="Seleccione…")
    sel_e = c2.multiselect("✂️ Bloquear caminos", list(enlaces), key="sim_enlaces", placeholder="Seleccione…")
    sel_f = c3.multiselect("⚡ Inyectar falla en", NODOS, key="sim_fallas", placeholder="Seleccione…")
    sel_h = c4.multiselect("✅ Habilitar (HAB)", bloq_afd, key="sim_hab", placeholder="Bloqueados por el AFD…")

    def fijar(nodos=(), enlaces_=(), fallas=(), hab=()):
        st.session_state["sim_nodos"] = list(nodos)
        st.session_state["sim_enlaces"] = list(enlaces_)
        st.session_state["sim_fallas"] = list(fallas)
        st.session_state["sim_hab"] = list(hab)

    tramo = rd.ruta_optima(base, origen, destino) or []
    b = st.columns(5)
    b[0].button("Bloquear servidor S1", on_click=fijar, kwargs={"nodos": ["S1"]}, width="stretch")
    b[1].button("Bloquear troncal S1 — S2", on_click=fijar,
                kwargs={"enlaces_": [k for k, e in enlaces.items() if set(e) == {"S1", "S2"}]}, width="stretch")
    b[2].button("Falla en la ruta actual", on_click=fijar,
                kwargs={"fallas": tramo[1:2] if len(tramo) > 2 else []}, width="stretch")
    b[3].button("Habilitar bloqueados", on_click=fijar, kwargs={"hab": bloq_afd}, width="stretch")
    b[4].button("Restablecer", on_click=fijar, width="stretch")

    esc = rd.simular(cruda, sel_n, [enlaces[e] for e in sel_e], sel_f, sel_h)
    m_esc = rd.metricas_red(esc)
    cambios = {v for v in NODOS if esc.nodes[v]["estado"] != base.nodes[v]["estado"]}

    def delta(a, b_, fmt="{:+d}"):
        dd = a - b_
        return None if abs(dd) < 0.05 else fmt.format(dd)

    k = st.columns(6)
    k[0].metric("Normal", m_esc["Normal"], delta(m_esc["Normal"], m_base["Normal"]))
    k[1].metric("Anómalo", m_esc["Anómalo"], delta(m_esc["Anómalo"], m_base["Anómalo"]), delta_color="inverse")
    k[2].metric("Bloqueado", m_esc["Bloqueado"], delta(m_esc["Bloqueado"], m_base["Bloqueado"]), delta_color="inverse")
    k[3].metric("Con servicio", f"{m_esc['servicio_pct']:.0f} %",
                delta(m_esc["servicio_pct"], m_base["servicio_pct"], "{:+.0f} pp"))
    k[4].metric("Tráfico medio (Mbps)", f"{m_esc['trafico_medio']:.1f}",
                delta(m_esc["trafico_medio"], m_base["trafico_medio"], "{:+.1f} Mbps"), delta_color="inverse")
    k[5].metric("Congestionados", m_esc["congestionados"], delta(m_esc["congestionados"], m_base["congestionados"]),
                delta_color="inverse")
    r_esc, comp_esc = (comparar_rutas(esc, origen, destino, modelo) if origen != destino else ({"ruta": None}, None))
    leyenda()
    components.html(visual.grafo_html(esc, 540, ruta=r_esc["ruta"], cambios=cambios), height=555)
    izq, der = st.columns([3, 2], gap="large")
    with izq:
        st.markdown(f"#### Ruta de {origen} a {destino} en el escenario")
        if comp_esc is not None:
            st.dataframe(comp_esc.style.format({"Tráfico acumulado (Mbps)": "{:.1f}", "Saltos": "{:.0f}"}, na_rep="—"),
                         hide_index=True, width="stretch")
            antes = rd.ruta_optima(base, origen, destino)
            if antes and r_esc["ruta"] and antes != r_esc["ruta"]:
                st.info("La GNN cambió la ruta: antes " + " → ".join(antes) + ".")
            elif r_esc["ruta"] is None:
                st.error(r_esc.get("motivo") or "No hay ruta.")
    with der:
        st.markdown("#### Equipos que cambiaron de estado")
        if cambios:
            st.dataframe(pd.DataFrame([{"Nodo": v, "Antes": base.nodes[v]["estado"], "Después": esc.nodes[v]["estado"],
                                        "Motivo": esc.nodes[v]["motivo_bloqueo"] or "AFD"} for v in NODOS if v in cambios])
                         .style.map(lambda x: COLOR_ESTADO.get(x, ""), subset=["Antes", "Después"]),
                         hide_index=True, width="stretch")
        else:
            st.info("Ningún equipo cambió de estado.")

# ----------------------------------------------------------------- AFD
afd = rd.AFD
with tab_afd:
    izq, der = st.columns(2, gap="large")
    with izq:
        st.markdown("#### Definición formal  M = (Q, Σ, δ, q₀, F)")
        st.markdown("- **Q** = {Normal, Anómalo, Bloqueado}\n- **Σ** = {OK, ERR, BLQ, HAB}\n"
                    "- **q₀** = Normal\n- **F** = {Normal}\n- **δ**: Q × Σ → Q, total (12 transiciones)")
        st.dataframe(pd.DataFrame([{"Símbolo": s, "Significado": au.DESCRIPCION_SIMBOLOS[s]} for s in au.ALFABETO]),
                     hide_index=True, width="stretch")
        st.dataframe(pd.DataFrame(afd.tabla()), hide_index=True, width="stretch")
        st.caption("→ inicial · * aceptación. Dos ERR seguidos bloquean el equipo; solo HAB lo saca de Bloqueado.")
        st.markdown('<div class="card mono">L(M) = ( OK + HAB + ERR·HAB*·OK + (ERR·HAB*·(ERR + BLQ) + BLQ)'
                    '·(OK + ERR + BLQ)*·HAB )*</div>', unsafe_allow_html=True)
    with der:
        st.markdown("#### Probar una cadena de eventos")
        ejemplos = {"Comunicación sana": "OK OK OK OK", "Falla aislada y recuperación": "OK ERR OK OK",
                    "Termina con una falla": "OK OK ERR", "Dos fallas seguidas (bloqueo)": "OK ERR ERR OK",
                    "Bloqueo manual": "OK OK BLQ OK", "Bloqueo y habilitación": "OK ERR ERR HAB OK"}
        ej = st.selectbox("Ejemplos", list(ejemplos))
        texto = st.text_input("Cadena (símbolos separados por espacios)", ejemplos[ej], key=f"cad_{ej}")
        cadena = au.parsear_cadena(texto)
        malos = [s for s in cadena if s not in au.ALFABETO]
        if malos:
            st.error(f"Símbolos fuera de Σ: {', '.join(malos)}")
        else:
            traza = afd.recorrer(cadena)
            p = st.slider("Paso de la simulación", 0, len(traza), len(traza)) if traza else 0
            act = traza[p - 1][2] if p else afd.inicial
            st.graphviz_chart(afd.a_dot(resaltar=act, arista=(traza[p - 1][0], traza[p - 1][2]) if p else None),
                              width="stretch")
            if p == len(traza):
                st.markdown(f'<span class="chip {act}">{"ACEPTADA" if afd.acepta(cadena) else "RECHAZADA"} · '
                            f'estado final {act}</span>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame([{"Paso": i + 1, "Estado": a, "Símbolo": s, "Siguiente": b}
                                       for i, (a, s, b) in enumerate(traza)])
                         .style.map(lambda x: COLOR_ESTADO.get(x, ""), subset=["Estado", "Siguiente"]),
                         hide_index=True, width="stretch")

# ----------------------------------------------------------------- Entrenamiento
with tab_gnn:
    izq, der = st.columns([1, 1], gap="large")
    with izq:
        st.markdown("#### Cómo funciona la GNN")
        st.latex(r"\hat c_\theta(v\to u) = \mathrm{softplus}\big(w_2^\top \mathrm{ReLU}(W_1\,\phi_{vu} + b_1) + b_2\big)")
        st.latex(r"h_v^{(k+1)} = \min\Big(h_v^{(k)},\ \min_{u\in N(v)} \big[\hat c_\theta(v\to u) + h_u^{(k)}\big]\Big),\quad h_t^{(0)} = 0")
        st.markdown("- **φ(v→u)**: tráfico del camino (Mbps), su cuadrado, latencia, si es inalámbrico, estado, CPU y tipo del equipo de llegada.\n"
                    "- Cada capa es un paso de **paso de mensajes**: el nodo recibe la distancia que conoce cada vecino "
                    "más el costo del camino y conserva el mínimo (Bellman-Ford).\n"
                    "- Tras |V| − 1 capas, el vecino que dio el mínimo es el **siguiente salto**; se sigue desde el origen.")
    with der:
        st.markdown("#### Definición del entrenamiento")
        st.markdown(f"""
| Elemento | Valor |
|---|---|
| Datos | 300 escenarios aleatorios de la red fija · todos los pares origen–destino |
| Etiqueta | distancia óptima d*(v, t) calculada con **Dijkstra** (maestro) |
| Pérdida | error cuadrático medio entre h_v y d*(v, t) |
| Optimizador | Adam, η = 0,01 · 300 épocas · 32 neuronas ocultas |
| División | {resultado.n_escenarios[0]} / {resultado.n_escenarios[1]} / {resultado.n_escenarios[2]} escenarios (70/15/15) |
| Selección | modelo con más rutas óptimas en validación |
""")
    st.markdown("#### Resultados en el conjunto de prueba")
    filas = []
    for nombre, mtr in (("GNN entrenada", resultado.prueba), ("GNN sin entrenar", resultado.sin_entrenar),
                        ("Menos saltos (BFS)", resultado.linea_base)):
        filas.append({"Método": nombre, "Rutas evaluadas": mtr["rutas"], "Rutas óptimas": 100 * mtr["optimas"],
                      "Sobrecosto medio (%)": 100 * mtr["sobrecosto_medio"],
                      "Sobrecosto máximo (%)": 100 * mtr["sobrecosto_max"]})
    st.dataframe(pd.DataFrame(filas).style.format({"Rutas óptimas": "{:.1f} %", "Sobrecosto medio (%)": "{:.2f}",
                                                   "Sobrecosto máximo (%)": "{:.1f}"}), hide_index=True, width="stretch")
    h = pd.DataFrame(resultado.historial)
    c1, c2 = st.columns(2)
    fig = go.Figure()
    fig.add_scatter(x=h["epoca"], y=h["perdida"], name="Pérdida entrenamiento", line=dict(color="#2DD4BF", width=2.5))
    fig.add_scatter(x=h["epoca"], y=h["perdida_val"], name="Pérdida validación", line=dict(color="#F59E0B", width=2.5, dash="dot"))
    fig.update_layout(title="Pérdida (error cuadrático medio)", xaxis_title="Época", yaxis_type="log")
    c1.plotly_chart(tema(fig), width="stretch")
    fig = go.Figure()
    fig.add_scatter(x=h["epoca"], y=100 * h["optimas_val"], name="Rutas óptimas (validación)",
                    line=dict(color="#A78BFA", width=2.5))
    fig.add_hline(y=100 * resultado.linea_base["optimas"], line_dash="dash", line_color="#94A3B8",
                  annotation_text="Menos saltos", annotation_position="bottom right")
    fig.update_layout(title="Rutas óptimas por época (%)", xaxis_title="Época", yaxis_range=[60, 100])
    c2.plotly_chart(tema(fig), width="stretch")
