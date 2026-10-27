"""Entraîne le détecteur sur un moteur sain simulé et le sauvegarde."""
import numpy as np

from pmaint.features import extract_many
from pmaint.model import AnomalyDetector
from pmaint.simulate import make_dataset


def train_detector(seed=0, n_train=600, n_val=300):
    rng = np.random.default_rng(seed)
    X_train = extract_many(make_dataset(rng, n_train))
    X_val = extract_many(make_dataset(rng, n_val))
    return AnomalyDetector(seed=seed).fit(X_train, X_val), X_val


if __name__ == "__main__":
    det, _ = train_detector()
    det.save("results/detector.joblib")
    print(f"Modèle sauvegardé (alarme si indice d'anomalie > {det.threshold:.1f})")
