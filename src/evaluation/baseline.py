import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Any, List

class ThermalBaselineModel:
    """
    Modèle de référence simple (Baseline) conforme à la section 7 du Cahier des Charges :
    1. Seuil thermique fixe ou adaptatif (Otsu)
    2. Classifieur classique (Random Forest sur statistiques globales de l'image)
    """

    def __init__(self, fixed_temp_thresh: float = 45.0, ambient_temp: float = 23.0, max_ref_temp: float = 120.0):
        self.fixed_temp_thresh = fixed_temp_thresh
        self.ambient_temp = ambient_temp
        self.max_ref_temp = max_ref_temp
        self.classifier = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        self.is_fitted = False

    def segment_threshold(self, img_bgr: np.ndarray, method: str = 'otsu') -> np.ndarray:
        """Segmentation naïve par seuillage thermique fixe ou Otsu."""
        if len(img_bgr.shape) == 3:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_bgr

        if method == 'otsu':
            _, mask = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return mask.astype(np.uint8)
        else:
            # Seuil de température fixe
            norm_thresh = (self.fixed_temp_thresh - self.ambient_temp) / (self.max_ref_temp - self.ambient_temp)
            pixel_thresh = int(np.clip(norm_thresh * 255.0, 0, 255))
            return (gray >= pixel_thresh).astype(np.uint8)

    def extract_global_stats(self, img_bgr: np.ndarray) -> np.ndarray:
        """Extrait des statistiques globales basiques sans modélisation spatiale avancée."""
        if len(img_bgr.shape) == 3:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_bgr

        mean_val = float(np.mean(gray))
        max_val = float(np.max(gray))
        std_val = float(np.std(gray))
        p90_val = float(np.percentile(gray, 90))

        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mean = float(np.mean(np.sqrt(sobel_x ** 2 + sobel_y ** 2)))

        return np.array([mean_val, max_val, std_val, p90_val, grad_mean], dtype=np.float32)

    def fit(self, images_bgr: List[np.ndarray], labels: List[int]):
        """Entraîne le classifieur Random Forest de référence."""
        X = np.array([self.extract_global_stats(im) for im in images_bgr])
        y = np.array(labels)
        self.classifier.fit(X, y)
        self.is_fitted = True

    def predict(self, img_bgr: np.ndarray) -> Dict[str, Any]:
        """Prédit la classe et la probabilité via la baseline."""
        stats = self.extract_global_stats(img_bgr)
        mask_otsu = self.segment_threshold(img_bgr, method='otsu')
        mask_fixed = self.segment_threshold(img_bgr, method='fixed')

        if self.is_fitted:
            prob = float(self.classifier.predict_proba([stats])[0][1]) if len(self.classifier.classes_) > 1 else 0.5
            pred_class = int(self.classifier.predict([stats])[0])
        else:
            # Règle heuristique simple si non entraîné
            prob = float(np.clip((stats[1] - 40.0) / 100.0, 0.0, 1.0))
            pred_class = 1 if prob >= 0.5 else 0

        return {
            'predicted_class': pred_class,
            'probability': round(prob, 4),
            'mask_otsu': mask_otsu,
            'mask_fixed': mask_fixed,
            'otsu_area': int(mask_otsu.sum())
        }
