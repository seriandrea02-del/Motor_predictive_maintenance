"""Chaîne complète : fenêtres -> indicateurs -> score -> état d'alerte."""
import numpy as np

from pmaint.alert import AlertManager
from pmaint.features import extract_many


def run_stream(detector, windows, n_confirm=3):
    """Traite une suite de fenêtres. Retourne (scores, états)."""
    X = extract_many(windows)
    scores = detector.score(X)
    manager = AlertManager(n_confirm)
    states = [manager.update(s > detector.threshold) for s in scores]
    return scores, states


def first_alert(states):
    for i, s in enumerate(states):
        if s == "ALERTE":
            return i
    return None
