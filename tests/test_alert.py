import pytest

from pmaint.alert import ALERTE, NORMAL, SURVEILLANCE, AlertManager


def test_une_anomalie_isolee_ne_declenche_pas_l_alerte():
    m = AlertManager(3)
    assert [m.update(x) for x in (False, True, False, True, False)] == [
        NORMAL, SURVEILLANCE, NORMAL, SURVEILLANCE, NORMAL]


def test_alerte_apres_trois_anomalies_consecutives():
    m = AlertManager(3)
    assert [m.update(True) for _ in range(4)] == [SURVEILLANCE, SURVEILLANCE, ALERTE, ALERTE]


def test_retour_a_la_normale():
    m = AlertManager(2)
    m.update(True), m.update(True)
    assert m.update(False) == NORMAL


def test_parametre_invalide():
    with pytest.raises(ValueError):
        AlertManager(0)
