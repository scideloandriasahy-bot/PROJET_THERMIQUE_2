import numpy as np
from hmmlearn import hmm
from typing import List, Dict, Tuple, Any

class HMMHealthTracker:
    """
    Modèle de Markov Caché (HMM) pour le suivi temporel de l'état de santé.
    Filtre les perturbations transitoires et décode la séquence d'états :
    - État 0 : Normal
    - État 1 : Suspect (défaut naissant ou perturbation légère)
    - État 2 : Anomalie confirmée (échauffement avéré persistant)
    - État 3 : Critique (surchauffe sévère, risque immédiat)
    """

    STATE_NAMES = ['Normal', 'Suspect', 'Confirmé', 'Critique']

    def __init__(self, n_states: int = 4, n_features: int = 3, random_state: int = 42):
        self.n_states = n_states
        self.n_features = n_features
        self.model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type='diag',
            n_iter=100,
            random_state=random_state,
            init_params=''
        )
        self.model.n_features = n_features
        self._set_prior_parameters()

    def _set_prior_parameters(self):
        """
        Initialise la matrice de transition et les émissions selon
        la physique de dégradation thermique industrielle.
        """
        # Distribution initiale
        self.startprob = np.array([0.70, 0.20, 0.08, 0.02], dtype=np.float64)

        # Matrice de transition
        self.transmat = np.array([
            [0.85, 0.12, 0.02, 0.01],  # Depuis Normal
            [0.10, 0.72, 0.16, 0.02],  # Depuis Suspect
            [0.02, 0.08, 0.78, 0.12],  # Depuis Confirmé
            [0.01, 0.02, 0.05, 0.92]   # Depuis Critique
        ], dtype=np.float64)

        # Moyennes des caractéristiques : [health_indicator, delta_t, area_ratio]
        self.means = np.array([
            [0.08,  3.0, 0.005],  # Normal
            [0.30, 12.0, 0.040],  # Suspect
            [0.60, 25.0, 0.090],  # Confirmé
            [0.88, 48.0, 0.130]   # Critique
        ], dtype=np.float64)

        # Variances diagonales
        self.vars = np.array([
            [0.010,  4.0, 0.0005],
            [0.015,  9.0, 0.0010],
            [0.020, 16.0, 0.0015],
            [0.025, 25.0, 0.0020]
        ], dtype=np.float64)

        # Assignation au modèle hmmlearn
        self.model.startprob_ = self.startprob
        self.model.transmat_ = self.transmat
        self.model.means_ = self.means
        self.model.covars_ = self.vars

    def extract_feature_vector(self, features_dict: Dict[str, float]) -> np.ndarray:
        """Convertit le dictionnaire de descripteurs en vecteur pour le HMM."""
        return np.array([
            float(features_dict.get('health_indicator', 0.0)),
            float(features_dict.get('delta_t', 0.0)),
            float(features_dict.get('area_ratio', 0.0))
        ], dtype=np.float64)

    def fit(self, feature_sequences: List[np.ndarray], lengths: List[int] = None):
        """
        Entraîne / affine le modèle HMM sur des séquences d'observations réelles.
        """
        if len(feature_sequences) == 0:
            return
        X = np.vstack(feature_sequences)
        if lengths is None:
            lengths = [len(seq) for seq in feature_sequences]
        self.model.fit(X, lengths)
        self.startprob = self.model.startprob_
        self.transmat = self.model.transmat_
        self.means = self.model.means_
        self.vars = self.model.covars_

    def decode_sequence(self, feature_sequence: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Décode la séquence d'états la plus probable (Viterbi)
        et calcule les probabilités a posteriori.
        """
        if len(feature_sequence) == 0:
            return np.array([], dtype=int), np.array([])
        
        _, state_seq = self.model.decode(feature_sequence, algorithm='viterbi')
        posteriors = self.model.predict_proba(feature_sequence)
        return state_seq, posteriors

    def filter_step(self, current_features: np.ndarray, prev_state_probs: np.ndarray = None) -> np.ndarray:
        """
        Inférence récursive en temps réel (Forward step) pour une nouvelle frame.
        Retourne la distribution de probabilité sur les 4 états de santé.
        """
        if prev_state_probs is None:
            prev_state_probs = self.startprob.copy()

        # Vraisemblance d'émission p(y_t | S_t = k)
        emiss_probs = np.zeros(self.n_states, dtype=np.float64)
        for k in range(self.n_states):
            diff = current_features - self.means[k]
            v = self.vars[k]
            # Gaussienne multidimensionnelle diagonale
            exp_term = -0.5 * np.sum((diff ** 2) / v)
            norm_term = 1.0 / np.sqrt((2.0 * np.pi) ** self.n_features * np.prod(v))
            emiss_probs[k] = max(1e-12, norm_term * np.exp(exp_term))

        # Prédiction d'état basée sur la matrice de transition
        pred_probs = prev_state_probs @ self.transmat

        # Mise à jour bayésienne
        updated_probs = pred_probs * emiss_probs
        norm_const = np.sum(updated_probs)
        if norm_const > 1e-12:
            updated_probs /= norm_const
        else:
            updated_probs = pred_probs

        return updated_probs
