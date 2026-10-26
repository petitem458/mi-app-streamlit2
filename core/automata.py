"""Autómata finito determinista (AFD) que sigue el estado de cada dispositivo.

M = (Q, Σ, δ, q0, F)
    Q  = {Normal, Anómalo, Bloqueado}
    Σ  = {OK, ERR, BLQ, HAB}
    q0 = Normal
    F  = {Normal}

Cada dispositivo produce, durante una ventana de comunicación, una cadena de
eventos sobre Σ. El AFD la lee y el estado en que termina es el estado del
dispositivo. Regla de tolerancia: dos fallas (ERR) seguidas bloquean el equipo
y solo una orden HAB (habilitar) lo devuelve a Normal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

# --------------------------------------------------------------------------
# Alfabeto y estados
# --------------------------------------------------------------------------
OK, ERR, BLQ, HAB = "OK", "ERR", "BLQ", "HAB"
ALFABETO = (OK, ERR, BLQ, HAB)

NORMAL, ANOMALO, BLOQUEADO = "Normal", "Anómalo", "Bloqueado"
ESTADOS = (NORMAL, ANOMALO, BLOQUEADO)
ABREV = {NORMAL: "N", ANOMALO: "A", BLOQUEADO: "B"}

DESCRIPCION_SIMBOLOS = {
    OK: "Comunicación exitosa: el dispositivo respondió y sus métricas están en rango",
    ERR: "Falla de comunicación: pérdida de paquetes, tiempo agotado o métricas fuera de rango",
    BLQ: "Orden de bloqueo: el administrador aísla el dispositivo de la red",
    HAB: "Orden de habilitación: el administrador desbloquea el dispositivo",
}

DESCRIPCION_ESTADOS = {
    NORMAL: "Opera dentro de lo esperado (estado inicial y de aceptación)",
    ANOMALO: "Registró una falla y está en observación",
    BLOQUEADO: "Aislado de la red; solo sale con HAB",
}


@dataclass
class AFD:
    """Autómata finito determinista genérico M = (Q, Σ, δ, q0, F)."""

    estados: tuple[str, ...]
    alfabeto: tuple[str, ...]
    delta: dict[tuple[str, str], str]
    inicial: str
    finales: set[str]
    nombres: dict[str, str] = field(default_factory=dict)

    def validar(self) -> None:
        """Comprueba que δ sea una función total Q × Σ → Q."""
        for q in self.estados:
            for a in self.alfabeto:
                if (q, a) not in self.delta:
                    raise ValueError(f"δ no está definida para ({q}, {a})")
                if self.delta[(q, a)] not in self.estados:
                    raise ValueError(f"δ({q}, {a}) apunta a un estado inexistente")

    def paso(self, estado: str, simbolo: str) -> str:
        if simbolo not in self.alfabeto:
            raise ValueError(f"Símbolo '{simbolo}' fuera del alfabeto Σ")
        return self.delta[(estado, simbolo)]

    def recorrer(self, cadena: Iterable[str]) -> list[tuple[str, str, str]]:
        """Traza [(estado, símbolo, siguiente), ...] = función δ̂ paso a paso."""
        traza, actual = [], self.inicial
        for simbolo in cadena:
            siguiente = self.paso(actual, simbolo)
            traza.append((actual, simbolo, siguiente))
            actual = siguiente
        return traza

    def estado_final(self, cadena: Iterable[str]) -> str:
        actual = self.inicial
        for simbolo in cadena:
            actual = self.paso(actual, simbolo)
        return actual

    def acepta(self, cadena: Iterable[str]) -> bool:
        return self.estado_final(cadena) in self.finales

    def tabla(self) -> list[dict[str, str]]:
        filas = []
        for q in self.estados:
            marca = ("→" if q == self.inicial else "") + ("*" if q in self.finales else "")
            fila = {"Estado": f"{marca}{q}"}
            for a in self.alfabeto:
                fila[a] = self.delta[(q, a)]
            filas.append(fila)
        return filas

    def a_dot(self, resaltar: str | None = None, arista: tuple[str, str] | None = None,
              tema_oscuro: bool = True) -> str:
        """Diagrama Graphviz; agrupa en una flecha los símbolos con igual destino."""
        texto = "#E6EDF3" if tema_oscuro else "#1F2937"
        borde = "#7D8FA9" if tema_oscuro else "#4B5563"
        colores = ({NORMAL: "#14532D", ANOMALO: "#7F1D1D", BLOQUEADO: "#334155"} if tema_oscuro
                   else {NORMAL: "#DCFCE7", ANOMALO: "#FEE2E2", BLOQUEADO: "#E2E8F0"})
        lineas = [
            "digraph AFD {",
            '  graph [rankdir=LR, bgcolor="transparent", pad=0.2, nodesep=0.45, ranksep=0.65];',
            f'  node [shape=circle, fontname="Helvetica", fontsize=19, color="{borde}", '
            f'fontcolor="{texto}", style=filled, width=1.6, fixedsize=true, penwidth=1.4];',
            f'  edge [fontname="Helvetica", fontsize=17, color="{borde}", fontcolor="{texto}", penwidth=1.2];',
            '  inicio [shape=point, width=0.1, color="#0EA5A4"];',
            f'  inicio -> "{self.inicial}";',
        ]
        for q in self.estados:
            forma = "doublecircle" if q in self.finales else "circle"
            relleno = "#0EA5A4" if q == resaltar else colores.get(q, "#F3F4F6")
            lineas.append(f'  "{q}" [shape={forma}, fillcolor="{relleno}"];')
        agrupado: dict[tuple[str, str], list[str]] = {}
        for (q, a), destino in self.delta.items():
            agrupado.setdefault((q, destino), []).append(a)
        for (q, destino), simbolos in agrupado.items():
            orden = [s for s in self.alfabeto if s in simbolos]
            etiqueta = ", ".join(orden)
            estilo = ""
            if arista and arista == (q, destino):
                estilo = ', color="#0EA5A4", penwidth=3, fontcolor="#0EA5A4"'
            lineas.append(f'  "{q}" -> "{destino}" [label=" {etiqueta} "{estilo}];')
        lineas.append("}")
        return "\n".join(lineas)


def minimizar(afd: AFD) -> list[set[str]]:
    """Refinamiento de particiones (Moore, 1956). Devuelve las clases de equivalencia.

    Si hay tantas clases como estados, el AFD ya es mínimo.
    """
    clase = {q: int(q in afd.finales) for q in afd.estados}
    while True:
        firmas = {q: (clase[q], *(clase[afd.delta[(q, a)]] for a in afd.alfabeto))
                  for q in afd.estados}
        indices: dict[tuple, int] = {}
        nueva = {q: indices.setdefault(firmas[q], len(indices)) for q in afd.estados}
        if len(set(nueva.values())) == len(set(clase.values())):
            break
        clase = nueva
    grupos: dict[int, set[str]] = {}
    for q in afd.estados:
        grupos.setdefault(nueva[q], set()).add(q)
    return list(grupos.values())


# --------------------------------------------------------------------------
# AFD del proyecto
# --------------------------------------------------------------------------
TRANSICIONES = {
    (NORMAL, OK): NORMAL,         # sigue sano
    (NORMAL, ERR): ANOMALO,       # primera falla: queda en observación
    (NORMAL, BLQ): BLOQUEADO,     # bloqueo manual
    (NORMAL, HAB): NORMAL,        # habilitar algo ya habilitado no cambia nada
    (ANOMALO, OK): NORMAL,        # se recupera
    (ANOMALO, ERR): BLOQUEADO,    # segunda falla seguida: bloqueo automático
    (ANOMALO, BLQ): BLOQUEADO,    # bloqueo manual
    (ANOMALO, HAB): ANOMALO,      # no está bloqueado: HAB no tiene efecto
    (BLOQUEADO, OK): BLOQUEADO,   # un equipo aislado no participa
    (BLOQUEADO, ERR): BLOQUEADO,
    (BLOQUEADO, BLQ): BLOQUEADO,
    (BLOQUEADO, HAB): NORMAL,     # única salida: habilitación del administrador
}

EXPRESION_REGULAR = ("((OK|HAB)|ERR HAB*OK|(ERR HAB*(ERR|BLQ)|BLQ)(OK|ERR|BLQ)*HAB)*")


def afd_dispositivo() -> AFD:
    afd = AFD(ESTADOS, ALFABETO, dict(TRANSICIONES), NORMAL, {NORMAL},
              {q: ABREV[q] for q in ESTADOS})
    afd.validar()
    return afd


# --------------------------------------------------------------------------
# Generación aleatoria de cadenas de eventos
# --------------------------------------------------------------------------
INTERVALOS = 10   # longitud de la ventana de observación de cada dispositivo


def generar_eventos(rng: np.random.Generator, prob_error: float,
                    bloqueo_manual: bool = False, habilitar: bool = False,
                    n: int = INTERVALOS) -> list[str]:
    """Cadena de n eventos OK/ERR (ERR con probabilidad prob_error).

    Si el usuario bloquea el equipo se agrega BLQ en una posición aleatoria de
    la segunda mitad; si pide habilitarlo, se agrega HAB al final.
    """
    cadena = [ERR if rng.random() < prob_error else OK for _ in range(n)]
    if bloqueo_manual:
        cadena.insert(int(rng.integers(n // 2, n + 1)), BLQ)
    if habilitar:
        cadena.append(HAB)
    return cadena


def parsear_cadena(texto: str) -> list[str]:
    """'OK, ERR-BLQ hab' → ['OK', 'ERR', 'BLQ', 'HAB']."""
    for separador in (",", "-", ";", "·", "|"):
        texto = texto.replace(separador, " ")
    return [t.strip().upper() for t in texto.split() if t.strip()]
