import numpy as np
import cv2
from typing import Dict, Any

class ThermalFeatureExtractor:
    """
    Extraction de descripteurs physiques et morphologiques interprétables
    à partir du thermogramme et du masque de segmentation MRF / U-Net.
    Conforme à la section 5.2 du Cahier des Charges.
    """

    def __init__(self, ambient_temp: float = 23.0, max_ref_temp: float = 120.0):
        self.ambient_temp = ambient_temp
        self.max_ref_temp = max_ref_temp

    def extract_features(self, intensity: np.ndarray, hotspot_mask: np.ndarray,
                         suspect_mask: np.ndarray = None, ambient_temp: float = None) -> Dict[str, float]:
        """
        Extrait les variables explicables :
        - t_max, t_mean : températures radiatives estimées (°C)
        - delta_t : écart thermique avec le fond / ambiance (°C)
        - area_pixels, area_ratio : étendue spatiale
        - centroid_x, centroid_y : position
        - dissymmetry : écart au centre machine (dissymétrie thermique)
        - gradient_mean : raideur du front thermique
        - texture_contrast, texture_homogeneity : texture locale
        - health_indicator : indicateur scalaire de sévérité [0.0, 1.0]
        """
        if ambient_temp is None:
            ambient_temp = self.ambient_temp

        H, W = intensity.shape
        total_pixels = H * W

        # Température apparente estimée en °C : T = T_amb + intensity * (T_max_ref - T_amb)
        temp_map = ambient_temp + intensity * (self.max_ref_temp - ambient_temp)

        has_hotspot = hotspot_mask is not None and np.sum(hotspot_mask) > 5

        if has_hotspot:
            hot_pixels = temp_map[hotspot_mask > 0]
            t_max = float(np.max(hot_pixels))
            t_mean = float(np.mean(hot_pixels))
            area_pixels = int(np.sum(hotspot_mask))
            area_ratio = float(area_pixels / total_pixels)

            # Calcul du barycentre (position)
            y_indices, x_indices = np.where(hotspot_mask > 0)
            centroid_y = float(np.mean(y_indices))
            centroid_x = float(np.mean(x_indices))

            # Dissymétrie thermique : distance normalisée au centre géométrique
            center_x, center_y = W / 2.0, H / 2.0
            dist_center = np.sqrt((centroid_x - center_x) ** 2 + (centroid_y - center_y) ** 2)
            dissymmetry = float(dist_center / (np.sqrt(center_x ** 2 + center_y ** 2)))

            # Fond thermique (pixels hors hotspot)
            bg_pixels = temp_map[hotspot_mask == 0]
            t_bg_mean = float(np.mean(bg_pixels)) if len(bg_pixels) > 0 else ambient_temp
            delta_t = max(0.0, t_mean - t_bg_mean)

            # Gradient thermique sur le contour du point chaud
            contours, _ = cv2.findContours(hotspot_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour_mask = np.zeros_like(hotspot_mask, dtype=np.uint8)
            cv2.drawContours(contour_mask, contours, -1, 1, thickness=2)

            sobel_x = cv2.Sobel(intensity, cv2.CV_32F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(intensity, cv2.CV_32F, 0, 1, ksize=3)
            grad_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

            if np.sum(contour_mask) > 0:
                gradient_mean = float(np.mean(grad_mag[contour_mask > 0]))
            else:
                gradient_mean = float(np.mean(grad_mag))

            # Descripteurs de texture locale dans la zone chaude
            local_var = float(np.var(intensity[hotspot_mask > 0]))
            texture_contrast = float(local_var * 10.0)
            texture_homogeneity = float(1.0 / (1.0 + local_var * 20.0))

        else:
            # Cas sans point chaud significatif (machine saine)
            t_max = float(np.max(temp_map))
            t_mean = float(np.mean(temp_map))
            delta_t = max(0.0, t_mean - ambient_temp)
            area_pixels = 0
            area_ratio = 0.0
            centroid_x = float(W / 2.0)
            centroid_y = float(H / 2.0)
            dissymmetry = 0.0

            sobel_x = cv2.Sobel(intensity, cv2.CV_32F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(intensity, cv2.CV_32F, 0, 1, ksize=3)
            grad_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
            gradient_mean = float(np.mean(grad_mag))

            texture_contrast = float(np.var(intensity) * 5.0)
            texture_homogeneity = 0.95

        # Indicateur de santé synthétique H_I in [0, 1]
        # Équilibré entre élévation thermique, surface de surchauffe et gradient
        norm_delta_t = np.clip(delta_t / 40.0, 0.0, 1.0)
        norm_area = np.clip(area_ratio / 0.15, 0.0, 1.0)
        norm_grad = np.clip(gradient_mean / 0.35, 0.0, 1.0)

        health_indicator = float(0.45 * norm_delta_t + 0.35 * norm_area + 0.20 * norm_grad)
        health_indicator = float(np.clip(health_indicator, 0.0, 1.0))

        return {
            't_max': round(t_max, 2),
            't_mean': round(t_mean, 2),
            'delta_t': round(delta_t, 2),
            'area_pixels': area_pixels,
            'area_ratio': round(area_ratio, 4),
            'centroid_x': round(centroid_x, 2),
            'centroid_y': round(centroid_y, 2),
            'dissymmetry': round(dissymmetry, 4),
            'gradient_mean': round(gradient_mean, 4),
            'texture_contrast': round(texture_contrast, 4),
            'texture_homogeneity': round(texture_homogeneity, 4),
            'health_indicator': round(health_indicator, 4)
        }
