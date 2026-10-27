import numpy as np
import pytest

from pmaint.simulate import WIN, make_scenario, make_window


def test_forme_et_reproductibilite():
    a = make_window(np.random.default_rng(1))
    b = make_window(np.random.default_rng(1))
    assert a[0].shape == (WIN,)
    assert np.array_equal(a[0], b[0]) and a[1] == b[1]


def test_echauffement_augmente_la_temperature():
    sain = np.mean([make_window(np.random.default_rng(i))[1] for i in range(30)])
    chaud = np.mean([make_window(np.random.default_rng(i), "overheat")[1] for i in range(30)])
    assert chaud > sain + 10


def test_condition_inconnue():
    with pytest.raises(ValueError):
        make_window(np.random.default_rng(0), "inconnue")


def test_scenario_structure():
    windows, sev, onset = make_scenario(np.random.default_rng(0), "bearing", 10, 20)
    assert len(windows) == 30 and onset == 10
    assert sev[:10].max() == 0 and sev[-1] == 1.0
