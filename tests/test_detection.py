import numpy as np
import pytest

from pmaint.features import extract_many
from pmaint.pipeline import first_alert, run_stream
from pmaint.simulate import make_dataset, make_scenario
from pmaint.train import train_detector


@pytest.fixture(scope="module")
def detector():
    return train_detector(seed=0, n_train=300, n_val=200)[0]


def test_peu_de_fausses_alarmes_sur_moteur_sain(detector):
    X = extract_many(make_dataset(np.random.default_rng(50), 200))
    assert detector.predict(X).mean() < 0.05


@pytest.mark.parametrize("fault", ["imbalance", "bearing", "overheat"])
def test_defaut_franc_detecte(detector, fault):
    X = extract_many(make_dataset(np.random.default_rng(51), 100, fault, 1.0))
    assert detector.predict(X).mean() > 0.9


@pytest.mark.parametrize("fault", ["imbalance", "bearing", "overheat"])
def test_alerte_apres_le_debut_du_defaut(detector, fault):
    windows, _, onset = make_scenario(np.random.default_rng(52), fault)
    _, states = run_stream(detector, windows)
    alert = first_alert(states)
    assert alert is not None and alert >= onset
