"""Logique d'alerte : une anomalie isolée ne suffit pas, il faut qu'elle persiste."""

NORMAL, SURVEILLANCE, ALERTE = "NORMAL", "SURVEILLANCE", "ALERTE"


class AlertManager:
    def __init__(self, n_confirm=3):
        if n_confirm < 1:
            raise ValueError("n_confirm doit être >= 1")
        self.n_confirm = n_confirm
        self.count = 0

    def update(self, is_anomaly):
        """À appeler à chaque fenêtre ; retourne l'état courant."""
        self.count = self.count + 1 if is_anomaly else 0
        if self.count == 0:
            return NORMAL
        return ALERTE if self.count >= self.n_confirm else SURVEILLANCE
