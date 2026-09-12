import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.models.axe2_deep.unet_segmenter import UNetSegmenter
from src.evaluation.metrics import ThermalEvaluationMetrics

def main():
    unet = UNetSegmenter(model_path='experiments_results/unet_weights.pt')
    gt_dir = 'Thermal image of equipment (Induction Motor) + 40 Ground Truths added/40_GT_with_Refrences/Subset_40_Thermal_GT'

    sample_ids = [
        ('017', 'Sain (Noload)'),
        ('160', 'Court-circuit Stator 30% (A30)'),
        ('281', 'Court-circuit Stator 50% (A50)'),
        ('f086', 'Défaut de Ventilation (Fan)')
    ]
    
    fig, axes = plt.subplots(4, 4, figsize=(18, 14))

    for row_idx, (sid, cond_name) in enumerate(sample_ids):
        bmp_path = os.path.join(gt_dir, f'{sid}.bmp')
        tiff_path = os.path.join(gt_dir, f'{sid}.tiff')
        
        img_bgr = cv2.imread(bmp_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gt_raw = cv2.imread(tiff_path)
        gt_mask = (gt_raw.mean(axis=2) > 128).astype(np.uint8)
        
        pred_res = unet.predict_mask(img_bgr, threshold=0.45)
        pred_mask = pred_res['binary_mask']
        
        # Calcul des métriques réelles
        m = ThermalEvaluationMetrics.compute_spatial_metrics(pred_mask, gt_mask)
        
        # Carte de Concordance RVB :
        # Vert = Vrais Positifs (Accord parfait IA et Expert)
        # Rouge = Faux Positifs (Prédit par IA mais absent de la vérité terrain)
        # Bleu = Faux Négatifs (Annoté par l'expert mais manqué par l'IA)
        h, w = gt_mask.shape
        concordance = np.zeros((h, w, 3), dtype=np.uint8)
        tp = np.logical_and(pred_mask == 1, gt_mask == 1)
        fp = np.logical_and(pred_mask == 1, gt_mask == 0)
        fn = np.logical_and(pred_mask == 0, gt_mask == 1)
        
        # Fond gris adouci
        gray_bg = (cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) * 0.35).astype(np.uint8)
        concordance[:, :, 0] = gray_bg
        concordance[:, :, 1] = gray_bg
        concordance[:, :, 2] = gray_bg

        concordance[tp] = [0, 230, 20]    # Vert éclatant
        concordance[fp] = [240, 20, 20]   # Rouge
        concordance[fn] = [20, 120, 255]  # Bleu

        # Affichage
        axes[row_idx, 0].imshow(img_rgb)
        axes[row_idx, 0].set_title(f"1. Thermogramme {sid}.bmp\\n({cond_name})", fontsize=11, fontweight='bold')

        axes[row_idx, 1].imshow(gt_mask, cmap='gray', vmin=0, vmax=1)
        axes[row_idx, 1].set_title("2. Masque Expert (Vérité Terrain)", fontsize=11, fontweight='bold')

        axes[row_idx, 2].imshow(pred_mask, cmap='hot', vmin=0, vmax=1)
        axes[row_idx, 2].set_title("3. Masque Prédit par IA (U-Net)", fontsize=11, fontweight='bold')

        title_metrics = f"4. Concordance Pixels\\nDice: {m['dice']*100:.1f}% | IoU: {m['iou']*100:.1f}% | Préc: {m['precision']*100:.1f}%"
        axes[row_idx, 3].imshow(concordance)
        axes[row_idx, 3].set_title(title_metrics, fontsize=10, color='darkgreen', fontweight='bold')

        for c in range(4):
            axes[row_idx, c].axis('off')

    plt.suptitle("ÉVALUATION VISUELLE DE LA SEGMENTATION : VÉRITÉ TERRAIN D'EXPERTS vs PRÉDICTION IA (U-NET)\\nLégende Concordance : Vert = Vrai Positif (Accord parfait), Rouge = Faux Positif, Bleu = Faux Négatif", fontsize=13, fontweight='bold', y=0.99)
    plt.tight_layout()

    out_dir = 'experiments_results'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'comparaison_visuelle_gt_vs_ia.png')
    plt.savefig(out_path, dpi=160, bbox_inches='tight')
    print(f"[SUCCESS] Image de comparaison visuelle générée avec succès dans : {out_path}")

if __name__ == '__main__':
    main()
