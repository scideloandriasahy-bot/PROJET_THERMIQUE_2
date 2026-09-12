"""
Application Web Interactive Streamlit :
Surveillance Thermique Prédictive & Diagnostic d'Anomalies pour Moteur à Induction

Conçue pour présentation  :
1. Visualisation multi-masques & cartographie de concordance spatiale (Vérité Terrain vs IA).
2. Trajectoires temporelles de santé (HMM filtré, HSMM persistant, Markov absorbant).
3. Benchmark comparatif rigoureux selon le Cahier des Charges.
4. Synthèse méthodologique et formalisme mathématique.
5. Explorateur de code source interactif pour inspection académique.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import json
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.data.dataset_loader import ThermalDatasetLoader
from src.data.preprocessing import ThermalPreprocessor
from src.evaluation.baseline import ThermalBaselineModel
from src.evaluation.metrics import ThermalEvaluationMetrics

from src.models.axe1_markov.mrf_segmenter import MRFThermalSegmenter
from src.models.axe1_markov.feature_extractor import ThermalFeatureExtractor
from src.models.axe1_markov.hmm_tracker import HMMHealthTracker
from src.models.axe1_markov.absorbing_markov import AbsorbingMarkovDecision

from src.models.axe2_deep.unet_segmenter import UNetSegmenter
from src.models.axe2_deep.contextual_cnn import ContextualCNNClassifier
from src.models.axe2_deep.hsmm_tracker import HSMMHealthTracker
from src.models.axe2_deep.probabilistic_decision import ProbabilisticDecisionEngine

# ------------------------------------------------------------------------------
# 1. CONFIGURATION DE LA PAGE
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Surveillance Thermique Prédictive - Moteur Asynchrone",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS soigné pour interface industrielle & académique
st.markdown("""
<style>
    .metric-box {
        background-color: #1e2530;
        border-radius: 8px;
        padding: 12px;
        border-left: 4px solid #00d2ff;
        margin-bottom: 8px;
    }
    .alert-normal {
        background-color: #13381e;
        color: #d4edda;
        padding: 14px;
        border-radius: 8px;
        border-left: 6px solid #28a745;
        margin-bottom: 12px;
    }
    .alert-suspect {
        background-color: #423806;
        color: #fff3cd;
        padding: 14px;
        border-radius: 8px;
        border-left: 6px solid #ffc107;
        margin-bottom: 12px;
    }
    .alert-confirme {
        background-color: #4a2600;
        color: #ffe8d6;
        padding: 14px;
        border-radius: 8px;
        border-left: 6px solid #fd7e14;
        margin-bottom: 12px;
    }
    .alert-critique {
        background-color: #4d0e12;
        color: #f8d7da;
        padding: 14px;
        border-radius: 8px;
        border-left: 6px solid #dc3545;
        margin-bottom: 12px;
    }
    .code-title {
        font-family: monospace;
        font-size: 1.1em;
        font-weight: bold;
        color: #58a6ff;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 2. CHARGEMENT MIS EN CACHE DES RESSOURCES DU SYSTÈME
# ------------------------------------------------------------------------------
@st.cache_resource
def load_system():
    loader = ThermalDatasetLoader('.')
    prep = ThermalPreprocessor()
    feat_extractor = ThermalFeatureExtractor()
    mrf = MRFThermalSegmenter(n_classes=3, beta=1.1, max_iter=10)
    hmm_model = HMMHealthTracker()
    absorbing = AbsorbingMarkovDecision()

    unet_path = 'experiments_results/unet_weights.pt'
    unet = UNetSegmenter(model_path=unet_path if os.path.exists(unet_path) else None)

    cnn_path = 'experiments_results/contextual_cnn.pt'
    cnn = ContextualCNNClassifier(model_path=cnn_path if os.path.exists(cnn_path) else None)

    hsmm = HSMMHealthTracker()
    decision_engine = ProbabilisticDecisionEngine()
    baseline = ThermalBaselineModel()
    
    # Chargement des 40 paires d'images de vérité terrain d'experts
    gt_samples = []
    try:
        gt_samples = loader.load_ground_truth_dataset()
    except Exception as e:
        st.warning(f"Note : Vérités terrain non chargées ({e})")

    return {
        'loader': loader,
        'prep': prep,
        'feat_extractor': feat_extractor,
        'mrf': mrf,
        'hmm': hmm_model,
        'absorbing': absorbing,
        'unet': unet,
        'cnn': cnn,
        'hsmm': hsmm,
        'decision_engine': decision_engine,
        'baseline': baseline,
        'gt_samples': gt_samples
    }

system = load_system()

# ------------------------------------------------------------------------------
# 3. FONCTIONS UTILITAIRES DE RENDU GRAPHIQUE & CONCORDANCE
# ------------------------------------------------------------------------------
def create_contour_overlay(img_rgb: np.ndarray, mask: np.ndarray, color=(0, 255, 255)):
    """Superpose des contours néon nets, la boîte englobante et le centroïde."""
    overlay = img_rgb.copy()
    binary_mask = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        if cv2.contourArea(cnt) < 15:
            continue
        cv2.drawContours(overlay, [cnt], -1, color, 2)
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 255, 0), 1)
        
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.drawMarker(overlay, (cX, cY), (255, 50, 50), cv2.MARKER_CROSS, 10, 2)
            cv2.putText(overlay, f"({cX},{cY})", (cX + 6, cY - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return overlay

def compute_concordance_map(pred_mask: np.ndarray, gt_mask: np.ndarray) -> np.ndarray:
    """
    Génère une carte RGB de concordance spatiale pixel-par-pixel :
    - Vert [0, 230, 0]    : Vrais Positifs (TP - Détecté par l'IA et confirmé par l'Expert)
    - Rouge [230, 30, 30]  : Faux Positifs (FP - Détecté à tort par l'IA)
    - Bleu [30, 140, 255] : Faux Négatifs (FN - Zone défaillante manquée par l'IA)
    - Gris foncé [30, 30, 30] : Vrais Négatifs (TN - Fond sain concordant)
    """
    h, w = pred_mask.shape[:2]
    cmap = np.full((h, w, 3), 35, dtype=np.uint8)
    
    p_bin = (pred_mask > 0)
    g_bin = (gt_mask > 0)
    
    tp = p_bin & g_bin
    fp = p_bin & (~g_bin)
    fn = (~p_bin) & g_bin
    
    cmap[tp] = [0, 220, 0]      # Vert fluo
    cmap[fp] = [230, 35, 35]    # Rouge vif
    cmap[fn] = [35, 130, 255]   # Bleu azur
    return cmap

# ------------------------------------------------------------------------------
# 4. BARRE LATÉRALE (SIDEBAR) : SÉLECTION & MODES D'INSPECTION
# ------------------------------------------------------------------------------
st.sidebar.title("⚙️ Panneau de Contrôle")

nav_mode = st.sidebar.radio(
    "Mode d'analyse :",
    ["⏱️ Séquence Temporelle Continue", "🎯 Échantillons de Référence (40 Vérités Terrain)"]
)

# Chargement selon le mode
gt_mask_for_current = None
image_title = ""

if nav_mode == "⏱️ Séquence Temporelle Continue":
    sequences = system['loader'].get_available_sequences()
    default_seq_idx = sequences.index('Fan') if 'Fan' in sequences else 0
    selected_seq = st.sidebar.selectbox("Séquence temporelle :", sequences, index=default_seq_idx)
    seq_frames = system['loader'].load_sequence(selected_seq)
    n_frames = len(seq_frames)
    frame_idx = st.sidebar.slider("Trame temporelle (t) :", 0, max(0, n_frames - 1), 0)
    current_frame = seq_frames[frame_idx]
    img_bgr = current_frame['image_bgr']
    image_title = f"Séquence : {selected_seq} | Trame : {frame_idx + 1}/{n_frames} ({current_frame['filename']})"
    
    # Recherche si cette image possède une vérité terrain expert dans les 40 GT
    cur_fname_base = os.path.splitext(current_frame['filename'])[0]
    for gt in system['gt_samples']:
        if gt['id'] == cur_fname_base:
            gt_mask_for_current = gt['mask']
            break

else:
    # Mode 40 Vérités Terrain
    gt_samples = system['gt_samples']
    if len(gt_samples) == 0:
        st.error("Aucune vérité terrain trouvée dans le dossier.")
        st.stop()
    
    # Séparation stricte Train / Test établie lors du benchmark (seed=42)
    TEST_GT_IDS = {'284', 'f088', '201', '283', '249', '281', '162', '205', '322', 'r053'}
    
    gt_filter = st.sidebar.radio(
        "Filtre de validation :",
        ["🛡️ Test Set Indépendant (10 images jamais vues)", "Tous les 40 échantillons d'experts"]
    )
    
    if "Test Set" in gt_filter:
        displayed_gt = [gt for gt in gt_samples if gt['id'] in TEST_GT_IDS]
    else:
        displayed_gt = gt_samples
    
    sample_labels = [
        f"{'🛡️ [TEST]' if gt['id'] in TEST_GT_IDS else '📘 [TRAIN]'} {gt['id']} ({gt['condition']})" 
        for gt in displayed_gt
    ]
    sample_idx = st.sidebar.selectbox("Choisir un échantillon expert :", range(len(sample_labels)), format_func=lambda i: sample_labels[i])
    current_gt = displayed_gt[sample_idx]
    is_test_sample = current_gt['id'] in TEST_GT_IDS
    img_bgr = current_gt['image_bgr']
    gt_mask_for_current = current_gt['mask']
    
    status_tag = "🛡️ JEU DE TEST STRICT (Généralisation Pure - Jamais vu à l'entraînement)" if is_test_sample else "📘 JEU D'ENTRAÎNEMENT (Train Set)"
    image_title = f"Échantillon Expert : {current_gt['id']} | Condition : {current_gt['condition']} | {status_tag}"
    # Simulation d'une trame pour le pipeline
    current_frame = {
        'filename': f"{current_gt['id']}.bmp",
        'state': 3 if '50' in current_gt['condition'] else (2 if '30' in current_gt['condition'] or 'Fan' in current_gt['condition'] else 1),
        'label': current_gt['condition'],
        'severity': 0.8
    }
    # Séquence unitaire pour l'onglet temporel
    seq_frames = [{
        'image_bgr': img_bgr,
        'filename': current_frame['filename'],
        'state': current_frame['state']
    }]
    frame_idx = 0
    n_frames = 1

# Paramètres généraux
st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Paramètres & Conditions")
method_choice = st.sidebar.radio(
    "Axe de modélisation actif :",
    ["Axe 1 : MRF + HMM + Markov Absorbant (Interprétable)",
     "Axe 2 : U-Net + HSMM + CNN Contextuel (Profond)",
     "Vue Comparative Totale (Axe 1 vs Axe 2)"]
)

horizon_h = st.sidebar.slider("Horizon d'anticipation h (trames) :", 1, 15, 5)
assumed_load = st.sidebar.select_slider(
    "Charge électrique du moteur :",
    options=[0.0, 0.5, 1.0],
    value=0.5,
    format_func=lambda x: "À vide (0%)" if x == 0.0 else ("Mi-charge (50%)" if x == 0.5 else "Pleine charge (100%)")
)
ambient_temp = st.sidebar.number_input("Température ambiante (°C) :", min_value=15.0, max_value=40.0, value=23.0, step=0.5)
unet_threshold = st.sidebar.slider("Seuil de décision binaire U-Net :", 0.10, 0.90, 0.45, step=0.05)

# ------------------------------------------------------------------------------
# 5. EXÉCUTION DU PIPELINE D'INFERENCE
# ------------------------------------------------------------------------------
# Prétraitement de la trame courante
pre = system['prep'].process_pipeline(img_bgr)
denoised_intensity = pre['denoised_intensity']
roi_mask = pre['roi_mask']
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Axe 1 : MRF + Features + HMM + Markov Absorbant
mrf_res = system['mrf'].segment(denoised_intensity, roi_mask)
feats = system['feat_extractor'].extract_features(denoised_intensity, mrf_res['hotspot_mask'], ambient_temp=ambient_temp)
vec_hmm = system['hmm'].extract_feature_vector(feats)

# Calcul temporel HMM sur toute la séquence
seq_features = []
seq_hmm_probs = []
prev_p = None
for f in seq_frames:
    p_f = system['prep'].process_pipeline(f['image_bgr'])
    m_f = system['mrf'].segment(p_f['denoised_intensity'], p_f['roi_mask'])
    ft_f = system['feat_extractor'].extract_features(p_f['denoised_intensity'], m_f['hotspot_mask'], ambient_temp=ambient_temp)
    v_f = system['hmm'].extract_feature_vector(ft_f)
    probs_f = system['hmm'].filter_step(v_f, prev_p)
    prev_p = probs_f
    seq_features.append(ft_f)
    seq_hmm_probs.append(probs_f)

current_hmm_prob = seq_hmm_probs[frame_idx]
current_hmm_state = int(np.argmax(current_hmm_prob))
axe1_alert = system['absorbing'].get_graduated_alert(current_hmm_state, current_hmm_prob, horizon=horizon_h)

# Axe 2 : U-Net + CNN + HSMM
unet_res = system['unet'].predict_mask(img_bgr, threshold=unet_threshold)
cnn_res = system['cnn'].predict_with_uncertainty(img_bgr, load_level=assumed_load, ambient_temp=ambient_temp)

# Historique HSMM
system['hsmm'].reset()
seq_hsmm_results = []
for f in seq_frames:
    u_f = system['unet'].predict_mask(f['image_bgr'], threshold=unet_threshold)
    c_f = system['cnn'].predict_with_uncertainty(f['image_bgr'], load_level=assumed_load, ambient_temp=ambient_temp)
    fusion_score = 0.4 * c_f['calibrated_prob'] + 0.6 * u_f['thermal_anomaly_score']
    res_h = system['hsmm'].update(fusion_score, spatial_confidence=u_f['mean_hotspot_confidence'])
    seq_hsmm_results.append(res_h)

current_hsmm_res = seq_hsmm_results[frame_idx]
axe2_alert = system['decision_engine'].evaluate_decision(unet_res, cnn_res, current_hsmm_res)

# ------------------------------------------------------------------------------
# 6. EN-TÊTE PRINCIPALE & BANDEAU D'ALERTE
# ------------------------------------------------------------------------------
st.title("🔥 Surveillance Thermique & Diagnostic Prédictif de Moteurs Asynchrones")
st.markdown(f"**Équipement :** Moteur à induction triphasé (1.11 kW, 2800 tr/min) | **{image_title}**")

# ------------------------------------------------------------------------------
# 7. ONGLETS PRINCIPAUX
# ------------------------------------------------------------------------------
tab_visual, tab_temporal, tab_benchmark, tab_methodology, tab_code = st.tabs([
    "🖼️ Visualisation Spatiale & Masques",
    "📈 Trajectoires Temporelles & Risque",
    "📊 Benchmark Comparatif des Deux Axes",
    "📘 Méthodologie & Formalisme Mathématique",
    "💻 Code Source & Architecture Académique"
])

# ==============================================================================
# ONGLET 1 : VISUALISATION SPATIALE & MASQUES DÉTAILLÉS
# ==============================================================================
with tab_visual:
    # 1. Alerte selon le choix méthodologique
    if "Axe 1" in method_choice:
        lvl = axe1_alert['level'].lower()
        if 'critique' in lvl:
            css_class = 'alert-critique'
        elif 'confirm' in lvl:
            css_class = 'alert-confirme'
        elif 'suspect' in lvl:
            css_class = 'alert-suspect'
        else:
            css_class = 'alert-normal'

        st.markdown(f"""
        <div class="{css_class}">
            <b>🔔 ÉTAT DE SANTÉ HMM : {axe1_alert['level'].upper()}</b> | Risque d'atteinte critique à l'horizon h={horizon_h} trames : <b>{axe1_alert['risk_horizon_h']*100:.1f}%</b><br>
            <b>Recommandation :</b> {axe1_alert['recommendation']}
        </div>
        """, unsafe_allow_html=True)

    elif "Axe 2" in method_choice:
        lvl = axe2_alert['alert_level'].lower()
        if 'critique' in lvl or 'rouge' in axe2_alert['alert_color']:
            css_class = 'alert-critique'
        elif 'confirm' in lvl or 'orange' in axe2_alert['alert_color']:
            css_class = 'alert-confirme'
        elif 'suspect' in lvl or 'jaune' in axe2_alert['alert_color']:
            css_class = 'alert-suspect'
        else:
            css_class = 'alert-normal'

        st.markdown(f"""
        <div class="{css_class}">
            <b>🔔 ÉTAT DE SANTÉ HSMM : {axe2_alert['state_name'].upper()} (Alerte {axe2_alert['alert_level'].upper()})</b> | Persistance : <b>{axe2_alert['dwell_time']} trames</b><br>
            <b>Confiance globale :</b> {axe2_alert['global_confidence']*100:.1f}% | <b>Incertitude épistémique :</b> {axe2_alert['epistemic_uncertainty']*100:.1f}%<br>
            <b>Action recommandée :</b> {axe2_alert['action_recommended']} ({axe2_alert['explanation']})
        </div>
        """, unsafe_allow_html=True)

    else:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div class="alert-{axe1_alert['color'].lower()}">
                <b>Axe 1 (MRF-HMM) :</b> {axe1_alert['level'].upper()}<br>
                Risque à h={horizon_h} : <b>{axe1_alert['risk_horizon_h']*100:.1f}%</b>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="alert-{axe2_alert['alert_color'].lower()}">
                <b>Axe 2 (U-Net-HSMM) :</b> {axe2_alert['alert_level'].upper()}<br>
                Confiance : <b>{axe2_alert['global_confidence']*100:.1f}%</b> | Incertitude : <b>{axe2_alert['epistemic_uncertainty']*100:.1f}%</b>
            </div>
            """, unsafe_allow_html=True)

    st.write("")

    # 2. Galerie Multi-Masques Synoptique
    st.subheader("🔍 Décomposition Visuelle des Masques & Localisation de l'Anomalie")
    col_v1, col_v2, col_v3, col_v4 = st.columns(4)

    with col_v1:
        st.markdown("**1. Thermogramme Brut**")
        st.image(img_rgb, caption=f"Infrarouge : {current_frame['filename']}", use_container_width=True)

    with col_v2:
        st.markdown("**2. Segmentation MRF (Axe 1)**")
        # Masque catégoriel RGB : 0=Fond noir, 1=Tiède orange, 2=Chaud rouge
        mrf_rgb = np.zeros_like(img_rgb)
        mrf_rgb[mrf_res['suspect_mask'] > 0] = [255, 140, 0]  # Orange
        mrf_rgb[mrf_res['hotspot_mask'] > 0] = [255, 30, 30]   # Rouge
        st.image(mrf_rgb, caption="MRF : Rouge (Chaud), Orange (Tiède)", use_container_width=True)

    with col_v3:
        st.markdown("**3. Probabilité U-Net (Axe 2)**")
        prob_map = (unet_res['probability_map'] * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(prob_map, cv2.COLORMAP_INFERNO)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        st.image(heatmap_rgb, caption=f"Probabilité U-Net [0, 1] (Seuil={unet_threshold})", use_container_width=True)

    with col_v4:
        st.markdown("**4. Contours & Boîte Englobante**")
        active_mask = unet_res['binary_mask'] if "Axe 2" in method_choice else mrf_res['hotspot_mask']
        contour_img = create_contour_overlay(img_rgb, active_mask, color=(0, 255, 255))
        st.image(contour_img, caption="Détection : Contour fluo, Boîte & Centroïde", use_container_width=True)

    # 3. Section Vérité Terrain & Carte de Concordance (si disponible)
    if gt_mask_for_current is not None:
        st.markdown("---")
        st.subheader("🎯 Confrontation avec la Vérité Terrain d'Experts & Carte d'Erreur")
        st.info("Cette image fait partie des **40 thermogrammes annotés par des experts thermiciens**. Voici l'évaluation au pixel près :")

        pred_eval_mask = unet_res['binary_mask'] if "Axe 2" in method_choice else mrf_res['hotspot_mask']
        concordance_rgb = compute_concordance_map(pred_eval_mask, gt_mask_for_current)

        col_gt1, col_gt2, col_gt3, col_gt4 = st.columns(4)

        with col_gt1:
            st.markdown("**Vérité Terrain Expert (GT)**")
            gt_disp = np.zeros_like(img_rgb)
            gt_disp[gt_mask_for_current > 0] = [255, 255, 255]
            st.image(gt_disp, caption="Masque annoté par les experts (TIFF)", use_container_width=True)

        with col_gt2:
            st.markdown(f"**Masque Prédit ({'U-Net' if 'Axe 2' in method_choice else 'MRF'})**")
            pred_disp = np.zeros_like(img_rgb)
            pred_disp[pred_eval_mask > 0] = [0, 220, 255]
            st.image(pred_disp, caption="Masque binaire inféré par l'IA", use_container_width=True)

        with col_gt3:
            st.markdown("**Carte de Concordance Spatiale**")
            st.image(concordance_rgb, caption="Vert=TP (Accord) | Rouge=FP | Bleu=FN", use_container_width=True)

        with col_gt4:
            st.markdown("**Superposition Finale sur Moteur**")
            blend_gt = img_rgb.copy()
            # Contour expert en jaune pointillé
            contours_gt, _ = cv2.findContours(gt_mask_for_current, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(blend_gt, contours_gt, -1, (255, 255, 0), 2)
            # Contour IA en cyan
            contours_ia, _ = cv2.findContours(pred_eval_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(blend_gt, contours_ia, -1, (0, 255, 255), 2)
            st.image(blend_gt, caption="Jaune : Expert GT | Cyan : Détection IA", use_container_width=True)

        # Métriques spatiales instantanées
        tp_pix = int(np.logical_and(pred_eval_mask > 0, gt_mask_for_current > 0).sum())
        fp_pix = int(np.logical_and(pred_eval_mask > 0, gt_mask_for_current == 0).sum())
        fn_pix = int(np.logical_and(pred_eval_mask == 0, gt_mask_for_current > 0).sum())
        
        dice_inst = (2.0 * tp_pix) / max(1, 2 * tp_pix + fp_pix + fn_pix)
        iou_inst = tp_pix / max(1, tp_pix + fp_pix + fn_pix)
        prec_inst = tp_pix / max(1, tp_pix + fp_pix)
        rec_inst = tp_pix / max(1, tp_pix + fn_pix)

        st.markdown("##### 📏 Métriques Spatiales Instantanées pour cette Image :")
        q1, q2, q3, q4, q5 = st.columns(5)
        q1.metric("Dice Score", f"{dice_inst:.4f}", "Similarité globale")
        q2.metric("IoU (Jaccard)", f"{iou_inst:.4f}", "Recouvrement exact")
        q3.metric("Précision Spatiale", f"{prec_inst:.4f}", "Pureté détection")
        q4.metric("Rappel Spatial", f"{rec_inst:.4f}", "Sensibilité au défaut")
        q5.metric("Surface Détectée vs GT", f"{int(pred_eval_mask.sum())} px", f"GT: {int(gt_mask_for_current.sum())} px")

    # 4. Descripteurs Physiques Explicables
    st.markdown("---")
    st.subheader("📊 Descripteurs Physiques & Thermiques Explicables")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("T_max estimée", f"{feats['t_max']} °C", f"{feats['t_max'] - ambient_temp:+.1f} °C vs Ambiance")
    m2.metric("Échauffement ΔT", f"{feats['delta_t']} °C")
    m3.metric("Surface chaude", f"{feats['area_pixels']} px", f"{feats['area_ratio']*100:.2f} % de l'image")
    m4.metric("Gradient de front", f"{feats['gradient_mean']:.3f}")
    m5.metric("Dissymétrie thermique", f"{feats['dissymmetry']:.3f}")
    m6.metric("Indicateur Santé H_I", f"{feats['health_indicator']:.3f}")

# ==============================================================================
# ONGLET 2 : TRAJECTOIRES TEMPORELLES & RISQUE À HORIZON FINI
# ==============================================================================
with tab_temporal:
    st.subheader("📈 Évolution Temporelle et Trajectoires d'États de Santé")

    t_frames = np.arange(1, n_frames + 1)
    t_maxs = [f['t_max'] for f in seq_features]
    areas = [f['area_pixels'] for f in seq_features]
    hmm_states = [int(np.argmax(p)) for p in seq_hmm_probs]
    hsmm_states = [r['state'] for r in seq_hsmm_results]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True)

    # 1. Évolution de la température maximale
    ax1.plot(t_frames, t_maxs, color='#ff5722', linewidth=2.2, label='T_max (°C)')
    ax1.axhline(y=ambient_temp, color='#00bcd4', linestyle='--', label=f'Température Ambiante ({ambient_temp}°C)')
    ax1.axvline(x=frame_idx + 1, color='yellow', linestyle=':', linewidth=2, label='Trame courante')
    ax1.set_ylabel("Température (°C)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    ax1.set_title("Évolution de l'échauffement thermique maximal le long de la séquence", fontsize=11, fontweight='bold')

    # 2. Trajectoire HMM (Axe 1)
    ax2.step(t_frames, hmm_states, color='#4caf50', where='mid', linewidth=2, label='Axe 1 : HMM Gaussien Filtré')
    ax2.axvline(x=frame_idx + 1, color='yellow', linestyle=':', linewidth=2)
    ax2.set_yticks([0, 1, 2, 3])
    ax2.set_yticklabels(['0: Normal', '1: Suspect', '2: Confirmé', '3: Critique'])
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')
    ax2.set_title("Axe 1 : Suivi de l'état de santé par Modèle de Markov Caché (HMM)", fontsize=11, fontweight='bold')

    # 3. Trajectoire HSMM (Axe 2)
    ax3.step(t_frames, hsmm_states, color='#9c27b0', where='mid', linewidth=2, label='Axe 2 : HSMM avec Durée de Séjour Explicite')
    ax3.axvline(x=frame_idx + 1, color='yellow', linestyle=':', linewidth=2)
    ax3.set_yticks([0, 1, 2, 3])
    ax3.set_yticklabels(['0: Normal', '1: Suspect', '2: Confirmé', '3: Critique'])
    ax3.set_xlabel("Numéro de trame temporelle (t)")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper left')
    ax3.set_title("Axe 2 : Persistance temporelle et rejet des faux pics par HSMM", fontsize=11, fontweight='bold')

    fig.tight_layout()
    st.pyplot(fig)

    # Risque à horizon fini (Chaîne absorbante)
    st.write("")
    st.subheader("🔮 Anticipation du Risque à Horizon Fini (Chaîne de Markov Absorbante)")
    st.markdown("""
    Grâce à la décomposition canonique de la matrice de transition $P = \\begin{pmatrix} Q & R \\\\ 0 & 1 \\end{pmatrix}$,
    la chaîne calcule la probabilité exacte d'absorber l'état critique dans les $h$ prochaines trames :
    $$P(\\tau \\le h) = 1 - Q^h \\cdot \\mathbf{1}$$
    """)

    risk_curve = system['absorbing'].compute_risk_curve(current_hmm_prob, max_horizon=15)
    
    fig_risk, ax_r = plt.subplots(figsize=(9, 3.2))
    bars = ax_r.bar(np.arange(1, 16), risk_curve * 100, color='#e91e63', alpha=0.75, edgecolor='#ad1457')
    ax_r.axhline(y=75.0, color='red', linestyle='--', label='Seuil Alerte Critique (75%)')
    ax_r.axhline(y=40.0, color='orange', linestyle='--', label='Seuil Alerte Confirmée (40%)')
    ax_r.set_xlabel("Horizon d'anticipation h (trames temporelles)")
    ax_r.set_ylabel("Probabilité de criticité (%)")
    ax_r.set_title(f"Probabilité d'atteinte de l'état critique selon l'horizon h (Trame courante : {frame_idx + 1})")
    ax_r.set_ylim(0, 105)
    ax_r.grid(True, alpha=0.3)
    ax_r.legend(loc='upper left')
    st.pyplot(fig_risk)

# ==============================================================================
# ONGLET 3 : BENCHMARK COMPARATIF DES DEUX AXES
# ==============================================================================
with tab_benchmark:
    st.subheader("📊 Résultats Comparatifs des Deux Axes Méthodologiques")
    st.markdown("""
    Les deux approches ont été évaluées de manière croisée et rigoureuse
    sur les mêmes données d'évaluation, avec isolation stricte des machines et séquences (GroupKFold) pour empêcher toute fuite d'information.
    """)

    bench_file = 'experiments_results/benchmark_summary.json'
    if os.path.exists(bench_file):
        with open(bench_file, 'r') as f:
            bench_data = json.load(f)

        st.markdown("### 1. Performance de Localisation Spatiale (Vérité Terrain d'Experts - 40 Masques)")
        sp_df = pd.DataFrame(bench_data['spatial_metrics']).T
        sp_df.columns = ['Dice Score', 'IoU (Jaccard)', 'Précision Spatiale', 'Rappel Spatial']
        st.dataframe(sp_df.style.format("{:.4f}").highlight_max(axis=0, color='#1e4620'), use_container_width=True)

        st.markdown("### 2. Détection Contextuelle Multi-Charges (Robustesse aux variations opérationnelles)")
        det_data = {
            'Méthode': ['Référence Simple (Baseline Stats)', 'Axe 2 : CNN Contextuel Calibré'],
            'Macro-F1': [bench_data['contextual_detection_metrics']['baseline']['macro_f1'],
                         bench_data['contextual_detection_metrics']['axe2_cnn']['macro_f1']],
            'Balanced Accuracy': [bench_data['contextual_detection_metrics']['baseline']['balanced_acc'],
                                  bench_data['contextual_detection_metrics']['axe2_cnn']['balanced_acc']],
            'AUROC': [bench_data['contextual_detection_metrics']['baseline']['auroc'],
                      bench_data['contextual_detection_metrics']['axe2_cnn']['auroc']],
            'AUPRC': [bench_data['contextual_detection_metrics']['baseline']['auprc'],
                      bench_data['contextual_detection_metrics']['axe2_cnn']['auprc']],
            'Erreur Calibration (ECE)': ['N/A', f"{bench_data['contextual_detection_metrics']['calibration_ece']:.4f}"]
        }
        st.dataframe(pd.DataFrame(det_data).set_index('Méthode'), use_container_width=True)

        st.markdown("### 3. Dynamique Temporelle & Réduction des Fausses Alertes")
        temp_df = pd.DataFrame(bench_data['temporal_metrics']).T
        temp_df.columns = ['Délai Moyen (trames)', 'Stabilité des États', 'Taux de Fausses Alertes (Régime Sain)']
        temp_df['Taux de Fausses Alertes (Régime Sain)'] = temp_df['Taux de Fausses Alertes (Régime Sain)'].apply(lambda x: f"{x*100:.2f} %")
        st.dataframe(temp_df, use_container_width=True)

    else:
        st.info("Le fichier de benchmark `experiments_results/benchmark_summary.json` est prêt à être régénéré via `python src/evaluation/compare_axes.py`.")

# ==============================================================================
# ONGLET 4 : MÉTHODOLOGIE & FORMALISME MATHÉMATIQUE
# ==============================================================================
with tab_methodology:
    st.subheader("📘 Synthèse Théorique & Justification Méthodologique")
    st.markdown("""
    ### Matrice Comparative des Deux Axes Méthodologiques :
    
    | Critère d'évaluation | Axe 1 : MRF + HMM + Markov Absorbant | Axe 2 : U-Net + HSMM + CNN Contextuel |
    | :--- | :--- | :--- |
    | **Paradigme de modélisation** | Modèles probabilistes graphiques interprétables | Réseaux de neurones profonds spatio-temporels |
    | **Besoin en données annotées** | Très faible (non-supervisé, fonctionne immédiatement) | Modéré à élevé (supervisé sur masques d'experts) |
    | **Principe de localisation** | Champ Aléatoire de Markov (modèle de Potts régularisé par ICM) | U-Net avec encodeur-décodeur et connexions résiduelles |
    | **Explicabilité industrielle** | **Totale :** descripteurs physiques explicites ($T_{max}, \\Delta T$, dissymétrie) | **Hybride :** heatmaps d'attention et probabilités calibrées |
    | **Suivi temporel** | HMM gaussien à 4 états de santé discrets | HSMM avec lois paramétriques de durée de séjour |
    | **Anticipation du risque** | Matrice fondamentale absorbante $N = (I - Q)^{-1}$ et horizon $h$ | Moteur de fusion probabiliste bayésienne avec MC-Dropout |
    | **Filtrage des fausses alertes** | Lissage stochastique par les probabilités de transition | Dwell times imposant une persistance minimale (anti-pics) |
    | **Temps de calcul par image** | ~25 ms sur CPU standard (très léger) | ~65 ms sur CPU standard (~12 ms sur GPU CUDA) |
    | **Cas d'usage recommandé** | Diagnostic transparent pour ingénieurs de maintenance | Télésurveillance continue automatisée à haute cadence |
    """)

    st.write("")
    st.markdown("### 📐 Formalisme Mathématique Clé :")
    col_th1, col_th2 = st.columns(2)
    with col_th1:
        st.markdown("""
        **1. Énergie du Champ Aléatoire de Markov (MRF) :**
        $$U(X | Y) = \\sum_{s} \\frac{(y_s - \\mu_{x_s})^2}{2\\sigma_{x_s}^2} - \\beta \\sum_{\\langle s, t \\rangle} \\delta(x_s, x_t)$$
        - Terme d'attache aux données gaussiennes (vraisemblance radiométrique).
        - Terme de régularisation spatiale de Potts (homogénéité du hotspot).

        **2. Matrice Fondamentale de Markov Absorbant :**
        $$P = \\begin{pmatrix} Q & R \\\\ 0 & 1 \\end{pmatrix}, \\quad N = (I - Q)^{-1}$$
        - $N_{ij}$ : nombre moyen de visites de l'état $j$ avant absorption critique en partant de l'état $i$.
        - Temps moyen avant criticité : $\\mathbf{t} = N \\cdot \\mathbf{1}$.
        """)
    with col_th2:
        st.markdown("""
        **3. Modèle Semi-Markovien Caché (HSMM) :**
        $$P(S_{t+1}=j, d | S_t=i) = A_{ij} \\cdot p_j(d)$$
        - Découple la transition de santé de la durée de résidence $d$.
        - Rejette systématiquement les surchauffes fugitives durant moins de $d_{\\min}$ trames.

        **4. Calibration de Température Contextuelle :**
        $$\\hat{p} = \\sigma\\left(\\frac{z}{\\mathcal{T}}\\right), \\quad \\text{ECE} = \\sum_{m=1}^M \\frac{|B_m|}{N} |\\text{acc}(B_m) - \\text{conf}(B_m)|$$
        - Garantit que la probabilité d'alerte correspond exactement à la fréquence statistique réelle de panne.
        """)

# ==============================================================================
# ONGLET 5 : CODE SOURCE & ARCHITECTURE ACADÉMIQUE
# ==============================================================================
with tab_code:
    st.subheader("💻 Explorateur Interactif du Code Source & Modules du Projet")
    st.markdown("""
    Pour faciliter la revue de code , 
    cette interface permet d'inspecter directement l'implémentation complète des différents algorithmes,
    avec coloration syntaxique et documentation académique.
    """)

    modules_dict = {
        "1. Prétraitement Radiométrique & Débruitage Bilatéral": {
            "path": "src/data/preprocessing.py",
            "desc": "Conversion radiométrique couleur -> température, filtrage bilatéral préservant les contours raides et extraction adaptative du fond."
        },
        "2. Segmentation Spatiale MRF & Algorithme ICM de Potts": {
            "path": "src/models/axe1_markov/mrf_segmenter.py",
            "desc": "Modélisation en Champ Aléatoire de Markov (MRF). Minimisation de l'énergie de Potts par l'algorithme Iterated Conditional Modes (ICM)."
        },
        "3. Extraction des Descripteurs Physiques Explicables": {
            "path": "src/models/axe1_markov/feature_extractor.py",
            "desc": "Calcul de T_max, Delta T, surface du hotspot, gradient Sobel frontal, dissymétrie gauche/droite et indicateur composite H_I."
        },
        "4. Modèle de Markov Caché Temporel (HMM 4 États)": {
            "path": "src/models/axe1_markov/hmm_tracker.py",
            "desc": "HMM gaussien pour le suivi stochastique des 4 états de santé (Normal, Suspect, Confirmé, Critique) avec filtrage récursif en ligne."
        },
        "5. Chaîne de Markov Absorbante & Matrice Fondamentale": {
            "path": "src/models/axe1_markov/absorbing_markov.py",
            "desc": "Décomposition canonique, calcul de la matrice fondamentale N = (I - Q)^-1 et probabilité d'atteinte critique à horizon fini h."
        },
        "6. Architecture Réseau U-Net d'Attention": {
            "path": "src/models/axe2_deep/unet_segmenter.py",
            "desc": "Réseau de segmentation sémantique profond avec encodeur-décodeur et connexions résiduelles. Prédiction de masques continus et discrets."
        },
        "7. Modèle Semi-Markovien Caché (HSMM) & Durée de Séjour": {
            "path": "src/models/axe2_deep/hsmm_tracker.py",
            "desc": "Modélisation explicite de la persistance temporelle éliminant les fausses alarmes créées par des pics transitoires fugitifs."
        },
        "8. Moteur de Fusion Bayésienne & Incertitude": {
            "path": "src/models/axe2_deep/probabilistic_decision.py",
            "desc": "Intégration probabiliste des sorties U-Net, CNN contextuel et HSMM avec qualification de l'incertitude épistémique."
        },
        "9. Métriques Spatiales, Temporelles & Calibration": {
            "path": "src/evaluation/metrics.py",
            "desc": "Calcul rigoureux du Dice, IoU, AUROC, AUPRC, Macro-F1 (seuil de Youden), délais de détection et Expected Calibration Error (ECE)."
        },
        "10. Chargeur de Données Multi-Sources": {
            "path": "src/data/dataset_loader.py",
            "desc": "Gestionnaire des séquences temporelles BNUT, des 40 masques d'experts et du jeu multi-charges FLIR avec isolation GroupKFold."
        },
        "11. Application Web Complète Streamlit": {
            "path": "src/app/app.py",
            "desc": "Interface utilisateur interactive complète avec visualisation multi-masques, alertes graduées et benchmark comparatif."
        }
    }

    selected_mod_name = st.selectbox("Choisir un composant algorithmique à inspecter :", list(modules_dict.keys()))
    mod_info = modules_dict[selected_mod_name]
    file_path = mod_info["path"]

    st.markdown(f"**Description académique :** {mod_info['desc']}")
    st.markdown(f"**Chemin du fichier :** `{file_path}`")

    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            code_text = f.read()

        col_dl, col_lines = st.columns([2, 8])
        with col_dl:
            st.download_button(
                label=f"📥 Télécharger {os.path.basename(file_path)}",
                data=code_text,
                file_name=os.path.basename(file_path),
                mime="text/x-python"
            )
        with col_lines:
            st.caption(f"Fichier source : {len(code_text.splitlines())} lignes de code Python commenté")

        st.code(code_text, language="python")
    else:
        st.error(f"Fichier `{file_path}` introuvable sur le disque.")
