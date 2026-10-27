"""Figures du README (PNG)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

from pmaint.simulate import FS, make_window

BLUE, GREEN, ORANGE, RED, GREY = "#2b6cb0", "#2f855a", "#dd6b20", "#c53030", "#4a5568"


def _flow(ax, y, labels, color, title):
    n = len(labels)
    w, gap = 0.16, (1 - 0.16 * n) / (n - 1)
    ax.text(0.0, y + 0.19, title, fontsize=12, fontweight="bold", color=color, va="bottom")
    for i, lab in enumerate(labels):
        x = i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, y - 0.12), w, 0.24, boxstyle="round,pad=0.008",
                                    fc="white", ec=color, lw=2))
        ax.text(x + w / 2, y, lab, ha="center", va="center", fontsize=9.5)
        if i < n - 1:
            ax.annotate("", xy=(x + w + gap - 0.004, y), xytext=(x + w + 0.004, y),
                        arrowprops=dict(arrowstyle="->", color=GREY, lw=1.8))


def architecture(path):
    fig, ax = plt.subplots(figsize=(13, 5.6))
    ax.set_xlim(-0.01, 1.01), ax.set_ylim(0, 1), ax.axis("off")
    _flow(ax, 0.74, ["Mesures du moteur\nsain (simulées)", "Indicateurs\n(7 par fenêtre)",
                     "Isolation Forest\n(vibrations) + limite\nde température", "Seuils calibrés\n(quantile 99,5 %)",
                     "Modèle prêt"],
          BLUE, "1. Apprentissage (hors ligne) : le modèle ne voit que du moteur sain")
    _flow(ax, 0.26, ["Capteurs :\nvibrations +\ntempérature", "Fenêtre\nde 1 s", "Indicateurs\n(7 par fenêtre)",
                     "Indice\nd'anomalie", "Alerte si 3 fenêtres\nconsécutives > seuil"],
          GREEN, "2. Surveillance (en continu)")
    ax.annotate("", xy=(0.72, 0.385), xytext=(0.90, 0.615),
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.8, ls="--"))
    ax.text(0.83, 0.50, "modèle + seuils", color=ORANGE, fontsize=10, ha="left")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def signals(path):
    rng = np.random.default_rng(3)
    cases = [("Moteur sain", "healthy"), ("Balourd (1x amplifié)", "imbalance"),
             ("Roulement abîmé (chocs)", "bearing")]
    fig, axes = plt.subplots(2, 3, figsize=(13, 5.8))
    for j, (title, cond) in enumerate(cases):
        vib, _ = make_window(rng, cond, 1.0)
        n = int(0.1 * FS)
        axes[0, j].plot(np.arange(n) / FS * 1000, vib[:n], lw=0.8, color=BLUE)
        axes[0, j].set_title(title, fontweight="bold")
        axes[0, j].set_xlabel("temps (ms)"), axes[0, j].set_ylim(-4, 4)
        spec = np.abs(np.fft.rfft(vib * np.hanning(vib.size))) / vib.size * 2
        f = np.fft.rfftfreq(vib.size, 1 / FS)
        axes[1, j].plot(f, spec, lw=0.8, color=GREEN)
        axes[1, j].set_xlim(0, 2200), axes[1, j].set_ylim(0, 1.0)
        axes[1, j].set_xlabel("fréquence (Hz)")
    axes[0, 0].set_ylabel("accélération (g)"), axes[1, 0].set_ylabel("amplitude spectrale")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def scenarios(results, threshold, path):
    titles = {"imbalance": "Balourd", "bearing": "Roulement abîmé", "overheat": "Échauffement"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (fault, r) in zip(axes, results.items()):
        ax.plot(r["scores"], color=BLUE, lw=1.2, label="indice d'anomalie")
        ax.axhline(threshold, color=ORANGE, ls="--", label="seuil (= 1)")
        ax.axvline(r["onset"], color=GREY, ls=":", label="début du défaut")
        if r["alert"] is not None:
            ax.axvline(r["alert"], color=RED, lw=2, label="alerte émise")
        ax.set_title(titles[fault], fontweight="bold")
        ax.set_xlabel("fenêtre (1 s)")
        ax.set_ylabel("indice d'anomalie")
    axes[0].legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def detection_curves(sweep, path):
    titles = {"imbalance": "Balourd", "bearing": "Roulement abîmé", "overheat": "Échauffement"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.0), sharey=True)
    for ax, fault in zip(axes, titles):
        s = sweep[fault]
        ax.plot(s["severites"], s["IA"], "o-", color=BLUE, label="IA (Isolation Forest + température)")
        ax.plot(s["severites"], s["seuils"], "s--", color=ORANGE, label="seuils fixes (RMS + température)")
        ax.set_title(titles[fault], fontweight="bold")
        ax.set_xlabel("sévérité du défaut (0 = sain, 1 = franc)")
        ax.set_ylim(-0.03, 1.05)
    axes[0].set_ylabel("part de fenêtres détectées")
    axes[0].legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
