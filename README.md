# SoftCast · Estado de los equipos (AFD) y camino óptimo (GNN)

Experiencia final de **Lenguajes y Autómatas**, Universidad de Cundinamarca, Fusagasugá (802M).
Paula Andrea Merchán Suaza · Victor Hernando Molina Rodríguez.

Red fija: los **nodos son los 18 equipos** (2 servidores, 5 PC de escritorio, 5 portátiles, 2 impresoras y 4 IoT)
y las **aristas son los 24 caminos** por donde se comunican, con **peso = tráfico medido en Mbps**.
Cada equipo tiene **dos o tres caminos diferentes** y la red es biconexa (sin puntos únicos de falla).

- **Recorrido BFS** desde los servidores: define el orden en que se evalúa cada equipo.
- **AFD de tres estados** (Normal, Anómalo, Bloqueado) sobre Σ = {OK, ERR, BLQ, HAB}: dos fallas
  seguidas bloquean el equipo y solo HAB lo habilita.
- **GNN de camino óptimo** (Bellman-Ford neuronal): una MLP aprende el costo de cada enlace y el paso de
  mensajes con mínimo encuentra la ruta de **menor tráfico acumulado (Mbps)**, esquivando equipos bloqueados.
  Costo real de cada arista: c(u→v) = tráfico (Mbps) + 30 Mbps si el equipo de llegada está Anómalo.
  Se entrena por imitación de Dijkstra (300 escenarios, pérdida MSE, Adam).
- **Simulación**: bloqueo y habilitación de equipos, bloqueo de caminos e inyección de fallas.

## Estructura

```
softcast_gnn/
├── app.py               Interfaz Streamlit (6 pestañas)
├── entrenar_modelo.py   Entrena la GNN y guarda los pesos (≈ 1 min)
├── modelos/             gnn_rutas.npz y gnn_rutas.json (pesos y métricas)
├── core/
│   ├── automata.py      AFD de 3 estados, minimización, generador de eventos
│   ├── datos.py         Perfiles, tráfico (Mbps), costo real de las rutas, características
│   ├── red.py           Red fija, recorrido BFS, simulación, Dijkstra y ruta de menos saltos
│   ├── gnn.py           GNN Bellman-Ford neuronal: entrenamiento, evaluación y búsqueda de rutas
│   └── visual.py        Grafo interactivo con Pyvis (aristas en Mbps)
├── tests/               31 pruebas unitarias (pytest)
├── requirements.txt
└── .streamlit/config.toml
```

## Ejecución local

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    ·    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Pruebas: `pip install pytest` y luego `python -m pytest -q`.

## Despliegue en Streamlit Community Cloud (gratuito)

1. Suba esta carpeta a un repositorio público de GitHub, con `app.py` en la raíz **e incluyendo `modelos/`**.
2. Ingrese a <https://share.streamlit.io> con su cuenta de GitHub.
3. *Create app* → repositorio, rama `main`, archivo `app.py`.
4. En *Advanced settings* elija Python 3.11 o 3.12 y pulse *Deploy*.
