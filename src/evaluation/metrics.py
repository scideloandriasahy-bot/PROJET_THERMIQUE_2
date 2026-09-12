import numpy as np
from sklearn.metrics import (
    f1_score, balanced_accuracy_score, roc_auc_score,
    precision_recall_curve, auc, precision_score, recall_score
)
from typing import Dict, List, Tuple, Any

class ThermalEvaluationMetrics:
    """
    Calculateur unifié des métriques d'évaluation requises par la section 8 du Cahier des Charges :
    - Localisation : Dice, IoU (Jaccard), Précision, Rappel
    - Détection : Macro-F1, Balanced Accuracy, AUROC, AUPRC
    - Temporel : Délai de détection, Stabilité des états, Taux de fausses alertes
    - Calibration : Expected Calibration Error (ECE)
    """

    @staticmethod
    def compute_spatial_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, float]:
        """
        Calcule les métriques de segmentation spatiale au niveau pixel.
        """
        p = (pred_mask > 0).astype(np.uint8).ravel()
        g = (gt_mask > 0).astype(np.uint8).ravel()

        intersection = np.logical_and(p, g).sum()
        union = np.logical_or(p, g).sum()
        pred_sum = p.sum()
        gt_sum = g.sum()

        iou = float(intersection / (union + 1e-7))
        dice = float((2.0 * intersection) / (pred_sum + gt_sum + 1e-7))
        precision = float(intersection / (pred_sum + 1e-7)) if pred_sum > 0 else (1.0 if gt_sum == 0 else 0.0)
        recall = float(intersection / (gt_sum + 1e-7)) if gt_sum > 0 else 1.0

        return {
            'dice': round(dice, 4),
            'iou': round(iou, 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4)
        }

    @staticmethod
    def compute_detection_metrics(y_true: List[int], y_prob: List[float], threshold: float = 0.5) -> Dict[str, float]:
        """
        Calcule les métriques globales de détection / classification.
        """
        y_true_arr = np.array(y_true)
        y_prob_arr = np.array(y_prob)

        if threshold == 'optimal' or threshold is None:
            from sklearn.metrics import roc_curve
            try:
                fpr, tpr, thresholds = roc_curve(y_true_arr, y_prob_arr)
                j_scores = tpr - fpr
                best_idx = int(np.argmax(j_scores))
                best_thresh = float(thresholds[best_idx])
                y_pred = (y_prob_arr >= best_thresh).astype(int)
            except Exception:
                y_pred = (y_prob_arr >= 0.5).astype(int)
        else:
            y_pred = (y_prob_arr >= threshold).astype(int)

        macro_f1 = float(f1_score(y_true_arr, y_pred, average='macro', zero_division=0))
        balanced_acc = float(balanced_accuracy_score(y_true_arr, y_pred))

        # AUROC et AUPRC
        try:
            auroc = float(roc_auc_score(y_true_arr, y_prob_arr))
        except ValueError:
            auroc = 0.5  # Si une seule classe est présente

        try:
            precision_curve, recall_curve, _ = precision_recall_curve(y_true_arr, y_prob_arr)
            auprc = float(auc(recall_curve, precision_curve))
        except ValueError:
            auprc = 0.0

        return {
            'macro_f1': round(macro_f1, 4),
            'balanced_acc': round(balanced_acc, 4),
            'auroc': round(auroc, 4),
            'auprc': round(auprc, 4)
        }

    @staticmethod
    def compute_temporal_metrics(state_sequence: List[int], true_fault_onset: int = None,
                                 is_healthy_seq: bool = False) -> Dict[str, float]:
        """
        Calcule les métriques temporelles :
        - detection_delay : nombre de trames entre l'apparition du défaut et la première alerte
        - state_stability : proportion de trames où l'état ne fluctue pas brusquement
        - false_alarm_count : nombre d'alertes intempestives sur séquence saine
        """
        T = len(state_sequence)
        if T == 0:
            return {'detection_delay': 0, 'state_stability': 1.0, 'false_alarm_rate': 0.0}

        # Stabilité : ratio de non-changements d'états
        transitions = sum(1 for i in range(1, T) if state_sequence[i] != state_sequence[i-1])
        state_stability = 1.0 - (transitions / max(1, T - 1))

        # Taux de fausses alertes sur séquence saine
        if is_healthy_seq:
            false_alarms = sum(1 for s in state_sequence if s >= 2)
            false_alarm_rate = float(false_alarms / T)
            detection_delay = 0.0
        else:
            false_alarm_rate = 0.0
            # Délai de détection
            first_alert_idx = None
            for idx, s in enumerate(state_sequence):
                if s >= 2:  # État Confirmé ou Critique
                    first_alert_idx = idx
                    break
            
            if first_alert_idx is not None and true_fault_onset is not None:
                detection_delay = max(0, first_alert_idx - true_fault_onset)
            elif first_alert_idx is not None:
                detection_delay = float(first_alert_idx)
            else:
                detection_delay = float(T)  # Non détecté

        return {
            'detection_delay': round(float(detection_delay), 2),
            'state_stability': round(float(state_stability), 4),
            'false_alarm_rate': round(float(false_alarm_rate), 4)
        }

    @staticmethod
    def compute_calibration_error(y_true: List[int], y_prob: List[float], n_bins: int = 10) -> Dict[str, float]:
        """
        Calcule l'Expected Calibration Error (ECE).
        """
        y_true = np.array(y_true)
        y_prob = np.array(y_prob)

        bins = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        bin_accuracies = []
        bin_confidences = []

        for i in range(n_bins):
            in_bin = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
            prop_in_bin = np.mean(in_bin)

            if prop_in_bin > 0:
                acc_in_bin = np.mean(y_true[in_bin])
                conf_in_bin = np.mean(y_prob[in_bin])
                ece += np.abs(acc_in_bin - conf_in_bin) * prop_in_bin
                bin_accuracies.append(float(acc_in_bin))
                bin_confidences.append(float(conf_in_bin))
            else:
                bin_accuracies.append(0.0)
                bin_confidences.append(float((bins[i] + bins[i+1]) / 2.0))

        return {
            'ece': round(float(ece), 4),
            'bin_accuracies': [round(v, 4) for v in bin_accuracies],
            'bin_confidences': [round(v, 4) for v in bin_confidences]
        }
