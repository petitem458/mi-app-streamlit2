"""Núcleo de SoftCast: estados de los equipos con un AFD y camino óptimo con una GNN.

Módulos:
    automata  -- AFD de 3 estados (Normal, Anómalo, Bloqueado) y generador de eventos.
    datos     -- Perfiles de CPU, tráfico de los caminos (Mbps), costo real y características.
    red       -- Red fija (18 equipos, 24 caminos de grado 2–3), recorrido BFS, simulación y rutas de referencia.
    gnn       -- GNN Bellman-Ford neuronal que aprende a encontrar el camino óptimo.
    visual    -- Grafo interactivo con Pyvis (las aristas muestran el tráfico en Mbps).
"""
