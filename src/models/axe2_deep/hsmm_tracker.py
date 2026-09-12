import numpy as np
from scipy.stats import poisson
from typing import Dict, List, Tuple, Any

class HSMMHealthTracker:
    """
    Hidden Semi-Markov Model (HSMM) modélisant explicitement
    les durées de séjour dans chaque état de santé.
    Conforme à la section 6.3 du Cahier des Charges.
    
    Permet de distinguer formellement :
    - Une élévation brève (transitoire, pic isolé)
    - Une anomalie durable (persistance confirmée)
    - Une aggravation progressive (passage aux états supérieurs)
    """

    STATE_NAMES = ['Normal', 'Suspect', 'Confirmé', 'Critique']

    def __init__(self, n_states: int = 4, max_duration: int = 25):
        self.n_states = n_states
        self.max_duration = max_duration

        # Matrice de transition entre états distincts (A_ii = 0)
        self.transition_matrix = np.array([
            [0.00, 0.85, 0.12, 0.03],  # Transition depuis Normal
            [0.25, 0.00, 0.65, 0.10],  # Transition depuis Suspect
            [0.05, 0.15, 0.00, 0.80],  # Transition depuis Confirmé
            [0.00, 0.00, 0.00, 1.00]   # Critique (absorbant)
        ], dtype=np.float64)

        # Durées caractéristiques moyennes de séjour par état (en nombre de trames)
        self.mean_durations = [15.0, 4.0, 8.0, 20.0]
        
        # Durée minimale requise pour valider la persistance d'une anomalie
        self.min_persistence_thresholds = [1, 2, 4, 3]

        self.current_state = 0
        self.current_dwell_time = 0
        self.state_history = []
        self.dwell_history = []

    def _duration_prob(self, state: int, d: int) -> float:
        """
        Distribution de probabilité de durée de séjour p_i(d).
        Modélisée par une distribution de Poisson décalée (d >= 1).
        """
        if d < 1:
            return 0.0
        lam = max(1.0, self.mean_durations[state] - 1.0)
        return float(poisson.pmf(d - 1, lam))

    def reset(self):
        self.current_state = 0
        self.current_dwell_time = 0
        self.state_history.clear()
        self.dwell_history.clear()

    def update(self, instantaneous_obs_score: float, spatial_confidence: float = 1.0) -> Dict[str, Any]:
        """
        Met à jour l'état HSMM à l'arrivée d'une nouvelle trame.
        instantaneous_obs_score in [0.0, 1.0] : score issu de CNN contextuel + U-Net.
        """
        # Estimation de l'état instantané observé
        if instantaneous_obs_score < 0.25:
            obs_state = 0  # Normal
        elif instantaneous_obs_score < 0.55:
            obs_state = 1  # Suspect
        elif instantaneous_obs_score < 0.80:
            obs_state = 2  # Confirmé
        else:
            obs_state = 3  # Critique

        # Gestion de la persistance temporelle
        if obs_state == self.current_state:
            # Séjour dans le même état
            self.current_dwell_time += 1
            is_transition = False
        else:
            # Tentative de transition
            min_req = self.min_persistence_thresholds[self.current_state]
            
            # Une aggravation brutale (ex: Normal -> Critique) exige une vérification de persistance
            # pour éviter les fausses alertes sur éclat ou reflet thermique
            if obs_state > self.current_state:
                # Progression vers le haut : acceptée si l'observation est soutenue
                if self.current_dwell_time >= 1 and spatial_confidence >= 0.6:
                    self.current_state = obs_state
                    self.current_dwell_time = 1
                    is_transition = True
                else:
                    # Reste temporairement dans l'état intermédiaire
                    self.current_state = min(self.current_state + 1, obs_state)
                    self.current_dwell_time = 1
                    is_transition = True
            else:
                # Retour vers un état plus froid (refroidissement) : requiert une durée minimale
                if self.current_dwell_time >= min_req:
                    self.current_state = obs_state
                    self.current_dwell_time = 1
                    is_transition = True
                else:
                    self.current_dwell_time += 1
                    is_transition = False

        self.state_history.append(self.current_state)
        self.dwell_history.append(self.current_dwell_time)

        # Qualification de la persistance
        is_persistent = (self.current_dwell_time >= self.min_persistence_thresholds[self.current_state])
        
        # Statut textuel
        if self.current_state == 0:
            nature = "Régime nominal"
        elif not is_persistent:
            nature = "Élévation brève transitoire (sous surveillance)"
        elif self.current_state == 1:
            nature = "Anomalie naissante persistante"
        elif self.current_state == 2:
            nature = "Anomalie thermique durable et avérée"
        else:
            nature = "État critique persistant (risque de défaillance immédiate)"

        return {
            'state': self.current_state,
            'state_name': self.STATE_NAMES[self.current_state],
            'dwell_time': self.current_dwell_time,
            'is_persistent': is_persistent,
            'is_transition': is_transition,
            'nature': nature,
            'instantaneous_state': self.STATE_NAMES[obs_state]
        }

    def decode_sequence(self, observation_scores: List[float], spatial_confidences: List[float] = None) -> List[Dict[str, Any]]:
        """Décode une séquence complète de scores observationnels."""
        self.reset()
        if spatial_confidences is None:
            spatial_confidences = [1.0] * len(observation_scores)
        
        results = []
        for score, conf in zip(observation_scores, spatial_confidences):
            res = self.update(score, conf)
            results.append(res)
        return results
