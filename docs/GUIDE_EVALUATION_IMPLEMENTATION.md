# 🎓 Guide d'Évaluation Académique & Technique


Bienvenue dans l'espace d'évaluation du projet de **Surveillance Thermique Prédictive et Diagnostic d'Anomalies de Moteurs Asynchrones**.

Ce document a été rédigé pour vous permettre d'évaluer rapidement le travail accompli, de naviguer aisément dans le code source et de tester les algorithmes développés.

---

## 🧭 1. Où Trouver Quoi ? (Cartographie du Travail)

Lien du depot github : https://github.com/scideloandriasahy-bot/PROJET_THERMIQUE_2
Lien de la démo streamlit : https://projetthermique2-4alljlvrbmd6npmnzjjksq.streamlit.app

| Ce que vous souhaitez examiner | Emplacement dans le projet | Description |
| :--- | :--- | :--- |
| **Démonstration Visuelle & Décision** | `src/app/app.py` *(ou lien Streamlit en ligne: https://projetthermique2-4alljlvrbmd6npmnzjjksq.streamlit.app)* | Visualisation des masques, cartes de concordance avec vos vérités terrain, courbes de santé et alertes. |
| **Exploration Pédagogique Pas-à-Pas** | `Projet_Thermique_Induction_Motor_Colab.ipynb` | Notebook Colab complet et autonome (exécutable en 1 clic sur Google Colab). |
| **Axe 1 : MRF & Potts (ICM)** | `src/models/axe1_markov/mrf_segmenter.py` | Modélisation markovienne spatiale et régularisation d'énergie. |
| **Axe 1 : HMM & Markov Absorbant** | `src/models/axe1_markov/hmm_tracker.py`<br>`src/models/axe1_markov/absorbing_markov.py` | Filtrage stochastique à 4 états et calcul du risque à horizon $h$ par matrice fondamentale $N=(I-Q)^{-1}$. |
| **Axe 2 : U-Net Spatio-Temporel** | `src/models/axe2_deep/unet_segmenter.py` | Architecture avec encodeur-décodeur et connexions résiduelles (poids dans `experiments_results/unet_weights.pt`). |
| **Axe 2 : Modèle HSMM Persistant** | `src/models/axe2_deep/hsmm_tracker.py` | Loi de durée de séjour explicite éliminant les fausses alertes sur pics fugitifs. |
| **Benchmark Croisé & Métriques** | `src/evaluation/metrics.py`<br>`src/evaluation/compare_axes.py` | Protocole d'évaluation strict sans fuite de données (Dice, IoU, AUROC, ECE, Délais). |
| **Rapport Scientifique Complet** | `docs/RAPPORT_FINAL_PROJET_THERMIQUE.pdf` | Rapport de synthèse avec justification des choix et état de l'art. |

---

## ⚡ 2. Comment lancer la démonstration ?

### Option A : Depuis votre navigateur sans rien installer (Recommandé)
1. **Application Web :** Ouvrez le lien d'accès partagé de l'application Streamlit: `https://projetthermique2-4alljlvrbmd6npmnzjjksq.streamlit.app`
   - Dans l'onglet **"🖼️ Visualisation Spatiale & Masques"**, vous pouvez basculer en mode **"Échantillons de Référence (40 Vérités Terrain)"** pour voir la confrontation directe entre les masques annotés par les experts et les prédictions de l'IA (carte Vert/Rouge/Bleu et métriques Dice/IoU calculées en temps réel).
   - Dans l'onglet **"💻 Code Source & Architecture Académique"**, vous pouvez lire et inspecter chaque fichier source Python directement depuis votre navigateur avec explications.
2. **Google Colab :** Ouvrez `Projet_Thermique_Induction_Motor_Colab.ipynb` dans Google Colab et cliquez sur **Environnement d'exécution > Tout exécuter**. Le notebook détecte l'environnement de façon autonome et affiche tous les graphiques.

### Option B : En local sur votre machine
```bash
# 1. Cloner et installer les dépendances
git clone <URL_DU_DEPOT>
cd PROJET_THERMIQUE_2
pip install -r requirements.txt

# 2. Lancer les 6 tests unitaires de validation
python run_all_tests.py

# 3. Lancer l'application web
streamlit run src/app/app.py
```

---


