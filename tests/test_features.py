import numpy as np

from pmaint.features import FEATURE_NAMES, extract
from pmaint.simulate import FS


def test_nombre_d_indicateurs():
    assert extract(np.zeros(FS) + np.random.default_rng(0).normal(size=FS), 60).size == len(FEATURE_NAMES)


def test_rms_d_une_sinusoide():
    t = np.arange(FS) / FS
    f = extract(2.0 * np.sin(2 * np.pi * 50 * t), 60)
    assert abs(f[0] - 2.0 / np.sqrt(2)) < 0.01


def test_kurtosis_bruit_gaussien():
    x = np.random.default_rng(0).normal(size=200_000)
    assert abs(extract(x, 60, fs=FS)[2] - 3.0) < 0.2


def test_energie_1x_detectee():
    t = np.arange(FS) / FS
    f = extract(np.sin(2 * np.pi * 50 * t), 60)
    assert f[3] > f[4] + 3          # bien plus d'énergie à 50 Hz qu'à 100 Hz
