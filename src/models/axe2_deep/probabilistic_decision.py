import numpy as np
from typing import Dict, Any

class ProbabilisticDecisionEngine:
    """
    Moteur décisionnel probabiliste fusionnant :
    1. La localisation spatiale U-Net (confiance et surface du point chaud)
    2. Le score contextuel CNN (intégrant la charge et la température ambiante)
    3. La persistance temporelle HSMM (durée de séjour dans l'état)
    4. L'incertitude épistémique (Monte Carlo Dropout)
    
    Conforme à la section 6.4 du Cahier des Charges.
    """

    def __init__(self, uncertainty_threshold: float = 0.08):
        self.uncertainty_threshold = uncertainty_threshold

    def evaluate_decision(self, unet_result: Dict[str, Any],
                          cnn_result: Dict[str, Any],
                          hsmm_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Génère une alerte probabiliste avec justification explicable.
        """
        prob_anomaly = cnn_result.get('calibrated_prob', 0.5)
        uncertainty = cnn_result.get('epistemic_uncertainty', 0.05)
        spatial_conf = unet_result.get('mean_hotspot_confidence', 0.0)
        hotspot_area = unet_result.get('hotspot_area', 0)
        
        state_idx = hsmm_result.get('state', 0)
        state_name = hsmm_result.get('state_name', 'Normal')
        dwell_time = hsmm_result.get('dwell_time', 1)
        is_persistent = hsmm_result.get('is_persistent', False)
        nature_text = hsmm_result.get('nature', '')

        # Confiance globale dans la décision
        # Pénalisée par l'incertitude du modèle et le manque de cohérence spatiale
        base_confidence = 1.0 - 2.0 * uncertainty
        if hotspot_area > 0 and spatial_conf < 0.4:
            base_confidence *= 0.85
        global_confidence = float(np.clip(base_confidence, 0.05, 0.99))

        # Vérification du besoin d'expertise humaine
        needs_human_review = bool(
            uncertainty > self.uncertainty_threshold or
            cnn_result.get('needs_human_review', False) or
            (state_idx >= 1 and not is_persistent and dwell_time <= 1)
        )

        # Détermination du niveau d'alerte gradué
        if state_idx == 3 and is_persistent:
            alert_level = 'Critique'
            alert_color = 'Rouge'
            urgency = 'Haute'
            action = "Arrêt de sécurité et inspection immédiate du moteur."
        elif state_idx == 2 and is_persistent:
            alert_level = 'Confirmé'
            alert_color = 'Orange'
            urgency = 'Moyenne'
            action = "Programmer une vérification thermique approfondie."
        elif state_idx == 1 or (state_idx >= 2 and not is_persistent):
            alert_level = 'Suspect'
            alert_color = 'Jaune'
            urgency = 'Basse'
            action = "Maintenir sous surveillance automatique renforcée."
        else:
            alert_level = 'Normal'
            alert_color = 'Vert'
            urgency = 'Aucune'
            action = "Poursuite de la surveillance en régime normal."

        # Explication textuelle complète
        explanation = (
            f"État: {state_name} ({nature_text}). "
            f"Probabilité d'anomalie contextuelle: {prob_anomaly*100:.1f}% "
            f"(Charge: {cnn_result.get('load_level', 0.0)*100:.0f}%, Ambiance: {cnn_result.get('ambient_temp', 23):.1f}°C). "
            f"Point chaud: {hotspot_area} px (confiance spatiale U-Net: {spatial_conf*100:.1f}%). "
            f"Persistance: {dwell_time} trame(s). "
            f"Incertitude modèle: {uncertainty*100:.1f}%."
        )

        if needs_human_review:
            explanation += " [ATTENTION: Validation humaine conseillée en raison de l'incertitude ou du caractère transitoire]."

        return {
            'alert_level': alert_level,
            'alert_color': alert_color,
            'urgency': urgency,
            'action_recommended': action,
            'anomaly_probability': round(prob_anomaly, 4),
            'global_confidence': round(global_confidence, 4),
            'epistemic_uncertainty': round(uncertainty, 4),
            'needs_human_review': needs_human_review,
            'state_name': state_name,
            'dwell_time': dwell_time,
            'is_persistent': is_persistent,
            'hotspot_area': hotspot_area,
            'explanation': explanation
        }
