"""Pruebas unitarias del AFD de tres estados (Normal, Anómalo, Bloqueado)."""

import itertools
import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import automata as au  # noqa: E402

AFD = au.afd_dispositivo()
N, A, B = au.NORMAL, au.ANOMALO, au.BLOQUEADO


@pytest.mark.parametrize("cadena, final", [
    ("", N),
    ("OK OK OK", N),
    ("OK ERR", A),
    ("OK ERR OK", N),
    ("ERR OK ERR OK", N),
    ("OK ERR ERR", B),
    ("ERR ERR OK OK", B),
    ("ERR ERR HAB", N),
    ("OK BLQ", B),
    ("OK BLQ OK ERR", B),
    ("OK BLQ HAB OK", N),
    ("ERR HAB", A),
    ("HAB HAB OK", N),
    ("ERR BLQ HAB ERR", A),
])
def test_estado_final(cadena, final):
    assert AFD.estado_final(au.parsear_cadena(cadena)) == final


def test_acepta_solo_si_termina_en_normal():
    assert AFD.acepta(au.parsear_cadena("OK ERR OK"))
    assert not AFD.acepta(au.parsear_cadena("OK ERR"))
    assert not AFD.acepta(au.parsear_cadena("OK BLQ"))


def test_delta_total_y_tres_estados():
    AFD.validar()
    assert AFD.estados == (N, A, B)
    assert len(AFD.delta) == 12


def test_bloqueado_solo_sale_con_hab():
    for a in au.ALFABETO:
        esperado = N if a == au.HAB else B
        assert AFD.paso(B, a) == esperado


def test_simbolo_fuera_del_alfabeto():
    with pytest.raises(ValueError):
        AFD.estado_final(["OK", "REQ"])


def test_afd_es_minimo():
    """El refinamiento de particiones produce tres clases: ningún estado sobra."""
    assert len(au.minimizar(AFD)) == 3


def test_expresion_regular_equivale_al_afd():
    """L(M) = expresión regular, para las 21 845 cadenas de longitud ≤ 7."""
    letra = {au.OK: "o", au.ERR: "e", au.BLQ: "b", au.HAB: "h"}
    patron = au.EXPRESION_REGULAR
    for simbolo, l in letra.items():
        patron = patron.replace(simbolo, l)
    patron = re.compile(patron.replace(" ", ""))
    for n in range(8):
        for cadena in itertools.product(au.ALFABETO, repeat=n):
            texto = "".join(letra[s] for s in cadena)
            assert bool(patron.fullmatch(texto)) == AFD.acepta(cadena), cadena


def test_generador_de_eventos():
    rng = np.random.default_rng(0)
    for _ in range(500):
        c = au.generar_eventos(rng, 0.3, bloqueo_manual=True)
        assert c.count(au.BLQ) == 1 and AFD.estado_final(c) == B
        c = au.generar_eventos(rng, 0.3, bloqueo_manual=True, habilitar=True)
        assert c[-1] == au.HAB and AFD.estado_final(c) == N
