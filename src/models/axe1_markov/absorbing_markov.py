import numpy as np
from typing import Dict, Any, Tuple

class AbsorbingMarkovDecision:
    """
    Modèle de chaîne de Markov absorbante pour l'évaluation du risque
    d'atteinte de l'état critique dans un horizon temporel fini h.
    Conforme à la section 5.4 du Cahier des Charges.
    """

    def __init__(self, transition_matrix: np.ndarray = None):
        if transition_matrix is None:
            # Matrice 4x4 par défaut : Normal (0), Suspect (1), Confirmé (2), Critique (3 - Absorbant)
            self.P = np.array([
                [0.85, 0.12, 0.02, 0.01],
                [0.10, 0.72, 0.16, 0.02],
                [0.02, 0.08, 0.78, 0.12],
                [0.00, 0.00, 0.00, 1.00]   # État absorbant
            ], dtype=np.float64)
        else:
            self.P = transition_matrix.copy()
            # Forcer le dernier état à être absorbant
            self.P[-1, :] = 0.0
            self.P[-1, -1] = 1.0
            # Renormalisation par ligne pour cohérence stochastique
            self.P = self.P / self.P.sum(axis=1, keepdims=True)

        self._compute_absorbing_properties()

    def _compute_absorbing_properties(self):
        """
        Décomposition canonique :
        P = [ Q   R ]
            [ 0   I ]
        Matrice fondamentale : N = (I - Q)^(-1)
        """
        n_transient = self.P.shape[0] - 1
        self.Q = self.P[:n_transient, :n_transient]
        self.R = self.P[:n_transient, n_transient:]
        
        # Matrice fondamentale N
        I = np.eye(n_transient, dtype=np.float64)
        self.N = np.linalg.inv(I - self.Q)

        # Temps moyen jusqu'à absorption depuis chaque état transitoire : t = N * 1
        self.mean_time_to_absorb = self.N @ np.ones((n_transient, 1), dtype=np.float64)

    def update_transition_matrix(self, new_P: np.ndarray):
        """Met à jour la matrice de transition à partir des observations réelles."""
        self.P = new_P.copy()
        self.P[-1, :] = 0.0
        self.P[-1, -1] = 1.0
        row_sums = self.P.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        self.P = self.P / row_sums
        self._compute_absorbing_properties()

    def probability_critical_at_horizon(self, state_probs: np.ndarray, horizon: int = 5) -> float:
        """
        Calcule la probabilité d'atteindre l'état critique dans un horizon donné h :
        P(tau <= h | distribution p)
        """
        if horizon <= 0:
            return float(state_probs[3])

        n_transient = self.Q.shape[0]
        # Q^h
        Q_h = np.linalg.matrix_power(self.Q, horizon)

        # Probabilité d'absorption pour les états transitoires : 1 - sum(Q_h, axis=1)
        prob_abs_transient = 1.0 - np.sum(Q_h, axis=1)

        # Probabilité globale pondérée par la distribution de probabilité courante
        risk = state_probs[3]  # Si déjà en état critique
        for i in range(n_transient):
            risk += state_probs[i] * prob_abs_transient[i]

        return float(np.clip(risk, 0.0, 1.0))

    def compute_risk_curve(self, state_probs: np.ndarray, max_horizon: int = 20) -> np.ndarray:
        """Calcule la courbe d'évolution du risque pour h = 1 .. max_horizon."""
        horizons = np.arange(1, max_horizon + 1)
        risks = np.array([self.probability_critical_at_horizon(state_probs, h) for h in horizons])
        return risks

    def get_graduated_alert(self, state_idx: int, state_probs: np.ndarray,
                            horizon: int = 5) -> Dict[str, Any]:
        """
        Fournit une alerte graduée accompagnée d'explications chiffrées :
        - Vert : Fonctionnement normal
        - Jaune : Surveillance renforcée (anomalie mineure / suspecte)
        - Orange : Pré-alerte (anomalie confirmée en développement)
        - Rouge : Alerte critique (intervention immédiate requise)
        """
        risk = self.probability_critical_at_horizon(state_probs, horizon=horizon)

        if state_idx == 3 or risk >= 0.75:
            level = 'Critique'
            color = 'Rouge'
            urgency = 'Haute'
            recommendation = "Inspection immédiate requise : fort risque d'avarie majeure ou arrêt machine."
        elif state_idx == 2 or risk >= 0.40:
            level = 'Confirmé'
            color = 'Orange'
            urgency = 'Moyenne'
            recommendation = "Planifier une maintenance préventive : échauffement avéré et persistant."
        elif state_idx == 1 or risk >= 0.15:
            level = 'Suspect'
            color = 'Jaune'
            urgency = 'Basse'
            recommendation = "Surveillance renforcée : dérive thermique suspecte détectée, suivre l'évolution."
        else:
            level = 'Normal'
            color = 'Vert'
            urgency = 'Aucune'
            recommendation = "Fonctionnement thermique nominal conforme."

        return {
            'level': level,
            'color': color,
            'urgency': urgency,
            'risk_horizon_h': round(risk, 4),
            'horizon': horizon,
            'recommendation': recommendation,
            'state_probs': [round(float(p), 4) for p in state_probs]
        }
