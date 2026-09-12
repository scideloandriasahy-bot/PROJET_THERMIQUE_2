import numpy as np
import cv2
from typing import Tuple, Dict, Any

class MRFThermalSegmenter:
    """
    Segmentation spatiale par Champ Aléatoire de Markov (MRF).
    Exploite le modèle de Potts spatial et l'optimisation par Iterated Conditional Modes (ICM)
    pour séparer l'image thermique en 3 régions spatialement cohérentes :
    - Classe 0 : Région normale / fond froid
    - Classe 1 : Région tiède / suspecte
    - Classe 2 : Point chaud / échauffement sévère
    """

    def __init__(self, n_classes: int = 3, beta: float = 1.2, max_iter: int = 15, tol: float = 1e-3):
        self.n_classes = n_classes
        self.beta = beta  # Poids du terme de régularisation spatiale (Potts)
        self.max_iter = max_iter
        self.tol = tol
        self.means = None
        self.variances = None

    def _initialize_parameters(self, intensity: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
        """
        Initialise les étiquettes et les paramètres gaussiens (moyenne, variance)
        par quantification des percentiles thermiques dans la région d'intérêt.
        """
        roi_pixels = intensity[roi_mask > 0] if roi_mask is not None and roi_mask.sum() > 0 else intensity.ravel()
        p_min, p_max = float(np.min(roi_pixels)), float(np.max(roi_pixels))
        
        # Initialisation par seuils thermiques relatifs et robustes
        if p_max - p_min > 1e-3:
            p33 = float(p_min + 0.30 * (p_max - p_min))
            p75 = float(p_min + 0.70 * (p_max - p_min))
        else:
            p33, p75 = 0.33, 0.66

        labels = np.zeros_like(intensity, dtype=np.int32)
        labels[(intensity >= p33) & (intensity < p75)] = 1
        labels[intensity >= p75] = 2

        self._update_gaussian_params(intensity, labels, roi_mask)
        return labels

    def _update_gaussian_params(self, intensity: np.ndarray, labels: np.ndarray, roi_mask: np.ndarray):
        """Met à jour les moyennes et variances de chaque classe."""
        self.means = np.zeros(self.n_classes, dtype=np.float32)
        self.variances = np.zeros(self.n_classes, dtype=np.float32)

        valid_mask = (roi_mask > 0) if roi_mask is not None else np.ones_like(intensity, dtype=bool)

        for k in range(self.n_classes):
            mask_k = (labels == k) & valid_mask
            if np.sum(mask_k) > 10:
                self.means[k] = float(np.mean(intensity[mask_k]))
                self.variances[k] = float(np.var(intensity[mask_k])) + 1e-4
            else:
                # Valeurs par défaut si classe quasi-vide
                self.means[k] = 0.15 if k == 0 else (0.50 if k == 1 else 0.85)
                self.variances[k] = 0.02

        # Assure l'ordre croissant des moyennes pour conserver la cohérence sémantique
        idx_order = np.argsort(self.means)
        self.means = self.means[idx_order]
        self.variances = self.variances[idx_order]

    def _compute_data_energy(self, intensity: np.ndarray) -> np.ndarray:
        """
        Calcule l'énergie d'attache aux données :
        U_data(y_p | x_p = k) = 0.5 * ln(2*pi*sigma_k^2) + (y_p - mu_k)^2 / (2*sigma_k^2)
        Shape: (H, W, n_classes)
        """
        H, W = intensity.shape
        data_energy = np.zeros((H, W, self.n_classes), dtype=np.float32)

        for k in range(self.n_classes):
            mu = self.means[k]
            var = self.variances[k]
            diff = intensity - mu
            data_energy[:, :, k] = 0.5 * np.log(2.0 * np.pi * var) + (diff ** 2) / (2.0 * var)

        return data_energy

    def segment(self, intensity: np.ndarray, roi_mask: np.ndarray = None) -> Dict[str, Any]:
        """
        Exécute l'algorithme ICM pour trouver la configuration spatiale optimale.
        Retourne :
        - labels : carte d'étiquettes {0, 1, 2}
        - hotspot_mask : masque binaire de surchauffe (classe 2)
        - suspect_mask : masque binaire des zones tièdes/suspectes (classe 1)
        - energy_history : évolution de l'énergie au fil des itérations
        """
        H, W = intensity.shape
        if roi_mask is None:
            roi_mask = np.ones((H, W), dtype=np.uint8)

        labels = self._initialize_parameters(intensity, roi_mask)
        energy_history = []

        # 8-voisinage spatial
        shifts = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

        for it in range(self.max_iter):
            # Énergie d'attache aux données
            data_energy = self._compute_data_energy(intensity)

            # Énergie de régularisation spatiale (Modèle de Potts)
            prior_energy = np.zeros((H, W, self.n_classes), dtype=np.float32)

            for dy, dx in shifts:
                shifted_labels = np.roll(np.roll(labels, dy, axis=0), dx, axis=1)
                for k in range(self.n_classes):
                    # Pénalise les désaccords entre le pixel p et son voisin q
                    prior_energy[:, :, k] += self.beta * (shifted_labels != k).astype(np.float32)

            # Énergie totale a posteriori
            total_energy = data_energy + prior_energy
            
            # Mise à jour ICM : assignation au label minimisant l'énergie locale
            new_labels = np.argmin(total_energy, axis=2).astype(np.int32)
            new_labels[roi_mask == 0] = 0

            # Calcul de la fraction de pixels modifiés
            changes = np.mean(new_labels != labels)
            energy_val = float(np.mean(np.min(total_energy, axis=2)))
            energy_history.append(energy_val)

            labels = new_labels
            self._update_gaussian_params(intensity, labels, roi_mask)

            if changes < self.tol:
                break

        # Post-filtrage morphologique léger pour éliminer de rares résidus
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        hotspot_mask = cv2.morphologyEx((labels == 2).astype(np.uint8), cv2.MORPH_OPEN, kernel)
        suspect_mask = (labels == 1).astype(np.uint8)

        return {
            'labels': labels,
            'hotspot_mask': hotspot_mask,
            'suspect_mask': suspect_mask,
            'means': self.means,
            'variances': self.variances,
            'iterations': len(energy_history),
            'energy_history': energy_history
        }
