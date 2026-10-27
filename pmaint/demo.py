"""Démonstration en flux : un moteur sain qui se dégrade, avec émission d'alertes.

Usage : python -m pmaint.demo --fault bearing
"""
import argparse

import numpy as np

from pmaint.pipeline import run_stream
from pmaint.simulate import make_scenario
from pmaint.train import train_detector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fault", choices=["imbalance", "bearing", "overheat"], default="bearing")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    detector, _ = train_detector()
    windows, sev, onset = make_scenario(np.random.default_rng(args.seed), args.fault)
    scores, states = run_stream(detector, windows)

    print(f"Défaut simulé : {args.fault} (début à la fenêtre {onset})")
    print(f"Seuil de l'indice d'anomalie : {detector.threshold:.1f}\n")
    previous = None
    for i, (sc, st) in enumerate(zip(scores, states)):
        if st != previous:
            print(f"fenêtre {i:3d} | indice {sc:.2f} | sévérité du défaut {sev[i]:.2f} | état : {st}")
            if st == "ALERTE":
                print(">>> ALERTE : maintenance à prévoir sur le moteur")
            previous = st


if __name__ == "__main__":
    main()
