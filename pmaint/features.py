"""Indicateurs calculés sur chaque fenêtre de mesure."""
import numpy as np

from pmaint.simulate import FS

FEATURE_NAMES = ["rms", "facteur_crete", "kurtosis",
                 "log_energie_1x", "log_energie_2x", "log_energie_HF", "temperature"]
_EPS = 1e-12


def extract(vib, temp, fs=FS):
    """Vecteur de 7 indicateurs : 6 pour les vibrations, 1 pour la température."""
    x = np.asarray(vib, dtype=float)
    x = x - x.mean()
    n = x.size
    p2 = np.mean(x ** 2)
    rms = np.sqrt(p2)
    crest = np.max(np.abs(x)) / rms
    kurt = np.mean(x ** 4) / (p2 ** 2)          # vaut ~3 pour un bruit gaussien

    spec = np.abs(np.fft.rfft(x * np.hanning(n))) ** 2 / n
    freqs = np.fft.rfftfreq(n, 1.0 / fs)

    def band(lo, hi):
        return spec[(freqs >= lo) & (freqs <= hi)].sum()

    return np.array([rms, crest, kurt,
                     np.log10(band(45, 55) + _EPS),
                     np.log10(band(95, 105) + _EPS),
                     np.log10(band(1000, 2000) + _EPS),
                     float(temp)])


def extract_many(windows):
    return np.vstack([extract(v, t) for v, t in windows])
