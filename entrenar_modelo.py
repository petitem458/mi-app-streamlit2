"""Entrena la GNN de rutas (Bellman-Ford neuronal) y guarda los pesos en modelos/.

Uso:  python entrenar_modelo.py      (≈ 1 minuto)
La aplicación carga estos archivos al iniciar; si no existen, entrena en el arranque.
"""

from pathlib import Path

from core import gnn

ESCENARIOS, SEMILLA, OCULTAS, EPOCAS, TASA = 300, 2026, 32, 300, 0.01
CARPETA = Path(__file__).parent / "modelos"


def main() -> None:
    CARPETA.mkdir(exist_ok=True)
    grafos = gnn.generar_dataset(ESCENARIOS, SEMILLA)
    r = gnn.entrenar(grafos, ocultas=OCULTAS, epocas=EPOCAS, lr=TASA)
    gnn.guardar(r, CARPETA / "gnn_rutas")
    print("GNN       ", r.prueba)
    print("Por saltos", r.linea_base)
    print(f"Entrenamiento: {r.segundos:.1f} s")


if __name__ == "__main__":
    main()
