"""Évalue le détecteur sur des données simulées et produit figures + métriques.

Usage : python -m pmaint.experiment
"""
import json

import numpy as np

from pmaint import plots
from pmaint.alert import AlertManager
from pmaint.features import extract_many
from pmaint.model import ThresholdBaseline
from pmaint.pipeline import first_alert, run_stream
from pmaint.simulate import make_dataset, make_scenario
from pmaint.train import train_detector

FAULTS = ("imbalance", "bearing", "overheat")
SEVERITIES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def _baseline_alert(baseline, windows, n_confirm=3):
    manager = AlertManager(n_confirm)
    flags = baseline.predict(extract_many(windows))
    return first_alert([manager.update(f) for f in flags])


def main(seed=0, n_test=300, n_sweep=100):
    detector, X_val = train_detector(seed)
    baseline = ThresholdBaseline().fit(X_val)

    rng = np.random.default_rng(1000 + seed)          # données de test indépendantes
    X_h = extract_many(make_dataset(rng, n_test))
    metrics = {"n_fenetres_test_sain": n_test,
               "fausses_alarmes_IA": float(detector.predict(X_h).mean()),
               "fausses_alarmes_seuils": float(baseline.predict(X_h).mean()),
               "detection_par_severite": {}, "scenarios": {}}

    sweep = {}
    for fault in FAULTS:
        ia, th = [], []
        for s in SEVERITIES:
            X_f = extract_many(make_dataset(rng, n_sweep, fault, s))
            ia.append(float(detector.predict(X_f).mean()))
            th.append(float(baseline.predict(X_f).mean()))
        sweep[fault] = {"severites": SEVERITIES, "IA": ia, "seuils": th}
    metrics["detection_par_severite"] = sweep

    plot_data = {}
    for k, fault in enumerate(FAULTS):
        windows, sev, onset = make_scenario(np.random.default_rng(7 + k), fault)
        scores, states = run_stream(detector, windows)
        alert = first_alert(states)
        alert_base = _baseline_alert(baseline, windows)
        plot_data[fault] = {"scores": scores, "onset": onset, "alert": alert}
        metrics["scenarios"][fault] = {
            "debut_defaut": onset,
            "IA": {"alerte_a_la_fenetre": alert,
                   "severite_a_l_alerte": None if alert is None else round(float(sev[alert]), 3)},
            "seuils": {"alerte_a_la_fenetre": alert_base,
                       "severite_a_l_alerte": None if alert_base is None else round(float(sev[alert_base]), 3)},
            "alerte_avant_defaut": alert is not None and alert < onset}

    plots.architecture("docs/architecture.png")
    plots.signals("docs/signaux.png")
    plots.scenarios(plot_data, detector.threshold, "docs/scenarios.png")
    plots.detection_curves(sweep, "docs/courbes_detection.png")
    with open("results/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(json.dumps(metrics, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
