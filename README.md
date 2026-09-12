# 🔥 Surveillance Thermique Prédictive & Diagnostic d'Anomalies sur Moteurs Asynchrones

Projet académique, Stage et industriel d'analyse d'images thermiques infrarouges pour la maintenance prédictive et la détection précoce des défaillances de moteurs asynchrones triphasés (1.11 kW, 2800 tr/min).

Ce projet met en œuvre, compare et valide deux approches méthodologiques complémentaires conformément au cahier des charges :
1. **Axe 1 (Modèles Graphiques Probabilistes Interprétables) :** Champ Aléatoire de Markov (**MRF**) avec modèle de Potts et optimisation **ICM**, extraction de descripteurs physiques explicables, suivi stochastique par Modèle de Markov Caché (**HMM**) et anticipation du risque d'avarie par **Chaîne de Markov Absorbante** (matrice fondamentale $N = (I - Q)^{-1}$).
2. **Axe 2 (Vision Profonde & Processus Spatio-Temporels) :** Réseau de segmentation sémantique profond **Attention U-Net** (Dice 98.5%), classification contextuelle multi-charges par **CNN Calibré** avec estimation d'incertitude épistémique, et filtrage temporel par Modèle Semi-Markovien Caché (**HSMM**) avec durées de séjour explicites pour éliminer les fausses alertes dues aux pics transitoires fugitifs.

---

## 🚀 Accès Rapide & Démonstration 

On dispose de trois canaux prêts à l'emploi :

| Canal d'évaluation | Description | Comment l'utiliser |
| :--- | :--- | :--- |
| **🌐 Application Web Interactive (Streamlit)** | Interface graphique complète : visualisation multi-masques, vérité terrain expert, cartes de concordance pixel-par-pixel, trajectoires temporelles et explorateur de code source. | `streamlit run src/app/app.py` *(ou via le lien Streamlit Cloud hébergé : https://projetthermique2-4alljlvrbmd6npmnzjjksq.streamlit.app)* |
| **📓 Google Colab Pédagogique** | Notebook pas-à-pas documenté avec théorie, formules mathématiques et visualisations graphiques. | Ouvrir `Projet_Thermique_Induction_Motor_Colab.ipynb` dans Google Colab et cliquer sur **"Tout exécuter"** |
| **🧪 Suite de Tests & Benchmark** | Validation rigoureuse des propriétés mathématiques et calcul des métriques croisées (Dice, IoU, AUROC, ECE, Délais). | `python run_all_tests.py`<br>`python src/evaluation/compare_axes.py` |

Lien du depot github : https://github.com/scideloandriasahy-bot/PROJET_THERMIQUE_2
Lien de la démo streamlit : https://projetthermique2-4alljlvrbmd6npmnzjjksq.streamlit.app
---

## 🗂️ Structure et Organisation du Projet

Le projet suit une arborescence modulaire et propre :

```text
PROJET_THERMIQUE_2/
├── README.md                                  # Présentation générale et guide de prise en main
├── requirements.txt                           # Dépendances Python nécessaires
├── run_all_tests.py                           # Lanceur automatique des 6 tests unitaires
├── build_notebook.py                          # Script de génération du notebook Colab pédagogique
├── Projet_Thermique_Induction_Motor_Colab.ipynb # Notebook interactif autonome pour Google Colab
│
├── src/                                       # Code source modulaire
│   ├── app/
│   │   └── app.py                             # Application web interactive Streamlit complète
│   │
│   ├── data/
│   │   ├── dataset_loader.py                  # Chargeur de séquences BNUT, 40 vérités terrain et FLIR
│   │   └── preprocessing.py                   # Débruitage bilatéral et normalisation radiométrique
│   │
│   ├── models/
│   │   ├── axe1_markov/
│   │   │   ├── mrf_segmenter.py               # Segmentation MRF (Potts + ICM)
│   │   │   ├── feature_extractor.py           # Extraction de Tmax, Delta T, surface, gradient, dissymétrie
│   │   │   ├── hmm_tracker.py                 # HMM gaussien à 4 états de santé
│   │   │   └── absorbing_markov.py            # Chaîne absorbante (matrice fondamentale N, risque à horizon h)
│   │   │
│   │   └── axe2_deep/
│   │       ├── unet_segmenter.py              # Réseau Attention U-Net PyTorch (inférence et métriques)
│   │       ├── contextual_cnn.py              # CNN multi-charges calibré avec MC-Dropout
│   │       ├── hsmm_tracker.py                # Modèle semi-markovien avec durées de séjour minimales
│   │       └── probabilistic_decision.py      # Moteur de fusion bayésienne des alertes
│   │
│   └── evaluation/
│       ├── baseline.py                        # Modèle de référence simple (Otsu / seuil statique)
│       ├── metrics.py                         # Métriques spatiales, de détection, temporelles et ECE
│       └── compare_axes.py                    # Benchmark automatisé comparant Axe 1 vs Axe 2 vs Baseline
│
├── experiments_results/                       # Poids entraînés et résultats synthétiques
│   ├── unet_weights.pt                        # Poids PyTorch du réseau U-Net
│   ├── contextual_cnn.pt                      # Poids PyTorch du CNN multi-charges
│   └── benchmark_summary.json                 # Rapport JSON consolidé du benchmark croisé
│
├── docs/                                      # Documentation académique détaillée
│   ├── GUIDE_EVALUATION_IMPLEMENTATION.md     # Guide pour l'evaluation et la demonstration
│   └── REPRODUCTION_GUIDE.md                  # Guide de reproduction des expériences
│
└── tests/                                     # Tests unitaires de validation
    ├── test_preprocessing.py                  # Vérifie le débruitage et l'isolation du fond
    ├── test_mrf.py                            # Vérifie la convergence ICM de l'énergie de Gibbs
    ├── test_axe1.py                           # Valide le pipeline complet Axe 1 sur séquence réelle
    ├── test_hmm_absorbing.py                  # Valide la matrice fondamentale N et le calcul du risque
    ├── test_unet.py                           # Valide la segmentation spatiale U-Net
    └── test_hsmm.py                           # Valide le rejet des faux pics transitoires par HSMM
```

---

## 🛠️ Installation et Exécution Locale

### 1. Prérequis
- Python 3.10, 3.11 ou 3.12
- Un gestionnaire d'environnement recommandé (`conda` ou `venv`)

### 2. Installation des dépendances
```bash
# Cloner le dépôt ou naviguer dans le dossier
cd PROJET_THERMIQUE_2

# Installer les dépendances
pip install -r requirements.txt
```

### 3. Lancer l'application web Streamlit
```bash
streamlit run src/app/app.py
```
L'interface s'ouvre automatiquement dans votre navigateur à l'adresse `http://localhost:8501`.

### 4. Valider l'ensemble des tests
```bash
python run_all_tests.py
```


## 📊 Synthèse des Résultats Expérimentaux (Benchmark)

Les deux approches ont été évaluées de façon croisée :

| Métrique d'Évaluation | Baseline (Otsu / RF) | Axe 1 : MRF + HMM + Markov Absorbant | Axe 2 : U-Net + HSMM + CNN Contextuel |
| :--- | :---: | :---: | :---: |
| **Dice Score (Spatial)** | 0.7288 | 0.7494 | **0.9851** |
| **IoU - Jaccard (Spatial)** | 0.5742 | 0.6070 | **0.9707** |
| **Précision Spatiale** | 0.5820 | **0.9603** | **0.9824** |
| **Rappel Spatial** | 0.9750 | 0.6145 | **0.9878** |
| **AUROC Détection** | 0.9696 | *Explicite* | **0.9316** |
| **AUPRC Détection** | 0.9904 | *Explicite* | **0.9806** |
| **Délai Détection Moyen** | 20.2 trames | **14.2 trames (Détection Précoce)** | 20.2 trames |
| **Stabilité Temporelle** | 0.9412 | **1.0000 (Lissage Parfait)** | **0.9893** |
| **Taux Fausses Alertes** | **0.00 %** | **0.00 %** | **0.00 %** |

### Conclusions Clés :
- **Axe 1 (Interprétable) :** Détecte les anomalies plus tôt (dès 14.2 trames en moyenne), ne nécessite aucun entraînement lourd, et fournit des explications physiques auditables ($T_{max}$, $\Delta T$, dissymétrie, probabilité d'absorption $(I-Q)^{-1}$) idéales pour le dialogue avec l'équipe de maintenance.
- **Axe 2 (Profond) :** Offre une délimitation géométrique quasi-parfaite de la zone défaillante (Dice 98.5%, IoU 97.1%) et son filtre HSMM élimine radicalement les faux positifs sur pics fugitifs.

---

## 👥 Auteurs
- **Auteur :** ANDRIASAHY Tsikomia Scidelo
- **Projet :** Système Intelligent de Surveillance Thermique et Diagnostic Prédictif de Moteurs Électriques
