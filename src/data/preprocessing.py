import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional

class ThermalPreprocessor:
    """
    Module de prétraitement des thermogrammes infrarouges.
    Assure la normalisation, l'estimation de température radiative,
    le débruitage préservant les contours et l'isolation du fond/ROI.
    """
    def __init__(self, target_size: Tuple[int, int] = (240, 320), bilateral_d: int = 5,
                 bilateral_sigma_color: float = 25.0, bilateral_sigma_space: float = 25.0):
        self.target_size = target_size  # (height, width)
        self.bilateral_d = bilateral_d
        self.bilateral_sigma_color = bilateral_sigma_color
        self.bilateral_sigma_space = bilateral_sigma_space

    def resize_if_needed(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        if (h, w) != self.target_size:
            return cv2.resize(img, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_AREA)
        return img

    def rgb_to_thermal_intensity(self, rgb_img: np.ndarray) -> np.ndarray:
        """
        Convertit un thermogramme RGB (palette pseudo-couleurs Ironbow/Jet)
        en intensité thermique normalisée [0.0, 1.0].
        """
        if len(rgb_img.shape) == 2:
            return rgb_img.astype(np.float32) / 255.0
        
        # Pour les palettes thermiques standard, les canaux R et G capturent les températures élevées
        b, g, r = rgb_img[:, :, 0], rgb_img[:, :, 1], rgb_img[:, :, 2]
        thermal_proxy = 0.50 * r.astype(np.float32) + 0.35 * g.astype(np.float32) + 0.15 * b.astype(np.float32)
        norm_intensity = np.clip(thermal_proxy / 255.0, 0.0, 1.0)
        return norm_intensity

    def denoise(self, img_float: np.ndarray) -> np.ndarray:
        """
        Filtre bilatéral pour éliminer le bruit capteur sans flouter
        les fronts thermiques raides (gradients d'échauffement).
        """
        img_uint8 = (img_float * 255.0).astype(np.uint8)
        denoised = cv2.bilateralFilter(
            img_uint8,
            d=self.bilateral_d,
            sigmaColor=self.bilateral_sigma_color,
            sigmaSpace=self.bilateral_sigma_space
        )
        return denoised.astype(np.float32) / 255.0

    def extract_thermal_roi(self, img_float: np.ndarray, threshold_ratio: float = 0.15) -> Tuple[np.ndarray, np.ndarray]:
        """
        Isole la machine du fond ambiant froid.
        Retourne (roi_mask, img_masked).
        """
        roi_mask = (img_float > threshold_ratio).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, kernel)
        roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN, kernel)
        img_masked = img_float * roi_mask
        return roi_mask, img_masked

    def process_pipeline(self, img_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Pipeline complet de prétraitement pour une image thermique brute.
        """
        img_resized = self.resize_if_needed(img_bgr)
        intensity = self.rgb_to_thermal_intensity(img_resized)
        denoised = self.denoise(intensity)
        roi_mask, masked_intensity = self.extract_thermal_roi(denoised)

        return {
            'original_bgr': img_resized,
            'raw_intensity': intensity,
            'denoised_intensity': denoised,
            'roi_mask': roi_mask,
            'masked_intensity': masked_intensity
        }
