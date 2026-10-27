"""Détecteur d'anomalies appris uniquement sur des mesures de moteur sain.

Deux voies complémentaires, combinées par un « indice d'anomalie » (alarme si > 1) :
 - vibrations : Isolation Forest sur les 6 indicateurs vibratoires ;
 - température : écart au niveau normal, rapporté à une limite calibrée sur le sain.
La température est traitée à part car, noyée parmi 7 indicateurs, une dérive sur une
seule variable n'est presque pas vue par une Isolation Forest (voir README).
"""
import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

N_VIB = 6          # les 6 premiers indicateurs sont vibratoires, le 7e est la température


class AnomalyDetector:
    def __init__(self, quantile=0.995, seed=0):
        self.quantile = quantile
        self.seed = seed
        self.scaler = StandardScaler()
        self.forest = IsolationForest(n_estimators=200, random_state=seed)
        self.threshold = 1.0                      # l'indice est normalisé : alarme si > 1

    def _vib_score(self, X):
        return -self.forest.score_samples(self.scaler.transform(X[:, :N_VIB]))

    def fit(self, X_train, X_val):
        self.scaler.fit(X_train[:, :N_VIB])
        self.forest.fit(self.scaler.transform(X_train[:, :N_VIB]))
        self.vib_limit = float(np.quantile(self._vib_score(X_val), self.quantile))
        self.temp_ref = float(np.mean(X_train[:, N_VIB]))
        self.temp_limit = float(np.quantile(X_val[:, N_VIB], self.quantile))
        return self

    def components(self, X):
        """(indice vibrations, indice température), chacun égal à 1 à son seuil."""
        X = np.atleast_2d(X)
        s_vib = self._vib_score(X) / self.vib_limit
        s_temp = (X[:, N_VIB] - self.temp_ref) / (self.temp_limit - self.temp_ref)
        return s_vib, s_temp

    def score(self, X):
        """Indice d'anomalie global : plus il est grand, plus c'est anormal (seuil = 1)."""
        s_vib, s_temp = self.components(X)
        return np.maximum(s_vib, s_temp)

    def predict(self, X):
        return self.score(X) > self.threshold

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)


class ThresholdBaseline:
    """Référence classique : seuils fixes sur le RMS et sur la température."""

    def __init__(self, quantile=0.995):
        self.quantile = quantile

    def fit(self, X_val):
        self.rms_limit = float(np.quantile(X_val[:, 0], self.quantile))
        self.temp_limit = float(np.quantile(X_val[:, N_VIB], self.quantile))
        return self

    def predict(self, X):
        X = np.atleast_2d(X)
        return (X[:, 0] > self.rms_limit) | (X[:, N_VIB] > self.temp_limit)
