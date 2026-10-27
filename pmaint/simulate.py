"""Génération de mesures simulées (vibrations + température) d'un moteur.

Les données sont SIMULÉES : elles servent à valider la chaîne de traitement,
pas à prouver des performances sur un vrai moteur.
"""
import numpy as np

FS = 5000        # fréquence d'échantillonnage de l'accéléromètre (Hz)
WIN = 5000       # taille d'une fenêtre : 1 s de signal
F_ROT = 50.0     # fréquence de rotation du moteur (Hz) = 3000 tr/min
T_NOM = 60.0     # température nominale du bobinage (°C)

CONDITIONS = ("healthy", "imbalance", "bearing", "overheat")


def _impacts(rng, n, severity):
    """Chocs périodiques d'un roulement abîmé : résonance à 1,5 kHz amortie."""
    out = np.zeros(n)
    period = int(FS / (3.6 * F_ROT))            # fréquence de défaut ~ 180 Hz
    tt = np.arange(int(0.012 * FS)) / FS
    burst = np.exp(-tt / 0.002) * np.sin(2 * np.pi * 1500 * tt)
    for start in range(int(rng.integers(0, period)), n - burst.size, period):
        out[start:start + burst.size] += burst * 1.2 * severity * rng.uniform(0.7, 1.3)
    return out


def make_window(rng, condition="healthy", severity=1.0):
    """Retourne (vibration, température) pour une fenêtre de 1 s."""
    if condition not in CONDITIONS:
        raise ValueError(f"condition inconnue : {condition}")
    t = np.arange(WIN) / FS
    load = rng.uniform(0.85, 1.15)              # la charge varie d'une fenêtre à l'autre
    a1 = 0.50 * load
    if condition == "imbalance":
        a1 *= 1.0 + 2.0 * severity              # balourd : la composante à 1x augmente
    a2 = 0.15 * load
    vib = (a1 * np.sin(2 * np.pi * F_ROT * t + rng.uniform(0, 2 * np.pi))
           + a2 * np.sin(2 * np.pi * 2 * F_ROT * t + rng.uniform(0, 2 * np.pi))
           + rng.normal(0, 0.20, WIN))
    if condition == "bearing":
        vib = vib + _impacts(rng, WIN, severity)
    temp = T_NOM + 10.0 * (load - 1.0) + rng.normal(0, 0.7)
    if condition == "overheat":
        temp += 18.0 * severity
    return vib, temp


def make_dataset(rng, n, condition="healthy", severity=1.0):
    return [make_window(rng, condition, severity) for _ in range(n)]


def make_scenario(rng, fault, n_healthy=60, n_degrad=80):
    """Moteur sain, puis un défaut qui s'aggrave progressivement (0 -> 1).

    Retourne (fenêtres, sévérités, indice de début du défaut).
    """
    windows, sev = [], []
    for _ in range(n_healthy):
        windows.append(make_window(rng, "healthy"))
        sev.append(0.0)
    for k in range(1, n_degrad + 1):
        s = k / n_degrad
        windows.append(make_window(rng, fault, s))
        sev.append(s)
    return windows, np.array(sev), n_healthy
