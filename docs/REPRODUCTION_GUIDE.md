# Guide de Reproduction - Projet Détection et Suivi d'Anomalies Thermiques

Ce guide permet de reproduire l'intégralité des expérimentations, d'exécuter la suite de tests unitaires, de relancer le benchmark comparatif et de démarrer le démonstrateur interactif sous Streamlit.

---

## 1. Prérequis & Environnement

Le projet fonctionne sous Python 3.10, 3.11 ou 3.12 (environnement Windows / Linux / macOS).

### Installation des dépendances
```bash
pip install -r requirements.txt
```

Principales bibliothèques utilisées :
- `torch` & `torchvision` : Réseaux profonds (Attention U-Net, CNN contextuel)
- `hmmlearn` & `scipy` : Modélisation markovienne et semi-markovienne (HMM, HSMM, processus stochastiques absorbants)
- `opencv-python` & `Pillow` : Prétraitement d'images infrarouges, morphologie mathématique, gradients
- `scikit-learn` : Métriques d'évaluation (Dice, IoU, AUROC, AUPRC, Macro-F1, Balanced Accuracy)
- `streamlit` : Démonstrateur interactif web
- `pandas` & `matplotlib` : Analyse de données et visualisations temporelles

---

## 2. Structure du Code Source

```
PROJET_THERMIQUE_2/
├── src/
│   ├── data/
│   │   ├── dataset_loader.py         # Chargeur des séquences BNUT, masques experts et dataset multi-charges
│   │   └── preprocessing.py          # Prétraitement, filtrage bilatéral, isolation ROI et normalisation
│   ├── models/
│   │   ├── axe1_markov/
│   │   │   ├── mrf_segmenter.py      # MRF spatial (Potts) optimisé par ICM (Iterated Conditional Modes)
│   │   │   ├── feature_extractor.py  # Descripteurs physiques explicables (Tmax, Delta-T, surface, gradient)
│   │   │   ├── hmm_tracker.py        # HMM gaussien à 4 états de santé (Normal, Suspect, Confirmé, Critique)
│   │   │   └── absorbing_markov.py   # Chaîne absorbante : matrice fondamentale (I-Q)^(-1) et risque à horizon h
│   │   └── axe2_deep/
│   │       ├── unet_segmenter.py     # Attention U-Net PyTorch pour la segmentation fine des anomalies
│   │       ├── contextual_cnn.py     # CNN contextuel (Charge, Ambiance) + Monte Carlo Dropout + Calibration
│   │       ├── hsmm_tracker.py       # HSMM avec durées explicites de séjour (rejet des pics transitoires)
│   │       └── probabilistic_decision.py # Fusion décisionnelle avec niveau d'alerte et confiance
│   ├── evaluation/
│   │   ├── baseline.py               # Seuillage thermique d'Otsu + Random Forest classique
│   │   ├── metrics.py                # Calcul unifié Dice, IoU, AUROC, Macro-F1, Délai, Stabilité, ECE
│   │   └── compare_axes.py           # Benchmark comparatif rigoureux et sauvegarde des résultats
│   └── app/
│       └── app.py                    # Démonstrateur interactif complet sous Streamlit
├── tests/                            # Suite complète de tests unitaires
├── docs/
│   ├── RAPPORT_FINAL.md              # Rapport technique et scientifique complet
│   └── REPRODUCTION_GUIDE.md         # Le présent guide
├── run_all_tests.py                  # Exécuteur unifié des tests unitaires
└── requirements.txt
```

---

## 3. Exécution des Tests Unitaires

Pour vérifier la validité mathématique et l'intégrité de tous les composants :
```bash
python run_all_tests.py
```

Le script exécute :
- `test_preprocessing.py` : Contrôle des formats et du débruitage bilatéral
- `test_axe1.py` : Intégration de bout en bout de l'Axe 1 sur séquence réelle
- `test_mrf.py` : Convergence ICM et élimination du bruit isolé par régularisation markovienne
- `test_hmm_absorbing.py` : Propriétés stochastiques, matrice fondamentale et monotonie du risque
- `test_unet.py` : Inférence U-Net et dimension des masques de sortie
- `test_hsmm.py` : Rejet des pics fugitifs d'une trame et validation de la persistance durable

---

## 4. Exécution du Benchmark Comparatif

Pour ré-entraîner les modèles et recalculer l'ensemble des métriques comparatives :
```bash
python src/evaluation/compare_axes.py
```

Les résultats synthétiques sont automatiquement enregistrés au format JSON dans :
`experiments_results/benchmark_summary.json`

---

## 5. Démarrage du Démonstrateur Interactif Streamlit

Pour lancer l'interface utilisateur interactive :
```bash
streamlit run src/app/app.py
```

L'application s'ouvre automatiquement dans le navigateur (par défaut sur `http://localhost:8501`).

### Fonctionnalités du démonstrateur :
1. **Sélecteur de séquence** : Navigation parmi les 11 séquences industrielles (`Noload`, `Fan`, `A10`, `A30`, `A50`, `Rotor-0`, etc.).
2. **Curseur temporel** : Parcours trame par trame avec mise à jour en temps réel des affichages.
3. **Visualisation spatiale** : Comparaison côte à côte du thermogramme brut, du masque MRF (Axe 1) et du masque U-Net (Axe 2).
4. **Descripteurs explicables** : Affichage dynamique de $T_{max}$, $\Delta T$, surface de la zone chaude en pixels, gradient de front et dissymétrie thermique.
5. **Alerte graduée & Risque à horizon $h$** :
   - Code couleur d'alerte (Vert, Jaune, Orange, Rouge).
   - Jauge de risque d'atteinte critique à horizon paramétrable $h$ via la chaîne absorbante.
   - Recommandation d'intervention de maintenance explicable.
6. **Suivi temporel** : Trajectoires comparées des états cachés HMM et HSMM au fil de la séquence.
7. **Onglet Benchmark** : Tableaux de performance complets (Dice, IoU, AUROC, AUPRC, Macro-F1, Stabilité, Délais).
