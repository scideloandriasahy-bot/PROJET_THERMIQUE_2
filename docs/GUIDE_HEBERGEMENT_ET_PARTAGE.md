# 🌐 Guide Complet d'Hébergement et de Partage de l'Application Web

Ce guide détaille les méthodes recommandées pour rendre l'application Streamlit accessible en ligne pour votre **encadrant de stage** et votre **professeur**, sans qu'ils aient besoin d'installer Python sur leur ordinateur.

---

## Option 1 : Déploiement Gratuit sur Streamlit Community Cloud (Méthode Recommandée)

**Streamlit Community Cloud** (`share.streamlit.io`) est la solution officielle, 100% gratuite et permanente. Votre application aura sa propre adresse web (ex: `https://surveillance-thermique-moteur.streamlit.app`) accessible 24h/24 et 7j/7.

### Étape 1 : Publier le projet sur GitHub
1. Ouvrez un terminal dans le dossier `PROJET_THERMIQUE_2` :
   ```bash
   git init
   git add .
   git commit -m "Version finale pour encadrant et jury : Surveillance Thermique Moteurs"
   ```
2. Créez un nouveau dépôt sur votre compte GitHub (ex: `PROJET_THERMIQUE_2`).
3. Liez votre dépôt local et poussez le code :
   ```bash
   git remote add origin https://github.com/VOTRE_PSEUDO/PROJET_THERMIQUE_2.git
   git branch -M main
   git push -u origin main
   ```

### Étape 2 : Connecter Streamlit Community Cloud
1. Rendez-vous sur [share.streamlit.io](https://share.streamlit.io) et connectez-vous avec votre compte GitHub.
2. Cliquez sur le bouton bleu **"New app"**.
3. Renseignez les 3 champs :
   - **Repository :** `VOTRE_PSEUDO/PROJET_THERMIQUE_2`
   - **Branch :** `main`
   - **Main file path :** `src/app/app.py`
4. Cliquez sur **"Deploy!"**.

> 💡 **Remarque sur les dépendances :** Le fichier `requirements.txt` a été spécialement configuré avec `opencv-python-headless>=4.7.0` afin de garantir que Streamlit Cloud installe le projet sans aucune erreur de bibliothèque système Linux (`libGL.so`).

---

## Option 2 : Partage Instantané par Tunnel Local (Cloudflared ou LocalTunnel)

Si vous voulez montrer l'application **immédiatement** à votre encadrant pendant une réunion ou par email sans créer de compte cloud, vous pouvez faire tourner l'application sur votre PC et ouvrir un tunnel web sécurisé en 30 secondes.

### Méthode avec Cloudflared (Tunnel Cloudflare Gratuit) :
1. Téléchargez `cloudflared` (ou installez-le via winget sous Windows : `winget install --id Cloudflare.cloudflared`).
2. Lancez l'application Streamlit :
   ```bash
   streamlit run src/app/app.py
   ```
3. Dans un deuxième terminal, lancez le tunnel :
   ```bash
   cloudflared tunnel --url http://localhost:8501
   ```
4. Cloudflare vous affiche une URL publique temporaire en `https://...trycloudflare.com`. Envoyez ce lien à votre professeur : il verra votre application en direct !

### Méthode avec LocalTunnel (Node.js) :
```bash
# Dans un terminal :
streamlit run src/app/app.py

# Dans un autre terminal :
npx localtunnel --port 8501
```

---

## Option 3 : Lancer l'Application Directement dans Google Colab avec LocalTunnel

Si votre encadrant préfère tout exécuter dans Google Colab sans rien avoir en local, vous pouvez exécuter Streamlit à l'intérieur d'une cellule Colab :

```python
# Cellule Colab pour lancer l'application Streamlit dans le cloud
!pip install -q streamlit opencv-python-headless hmmlearn
!npm install -g localtunnel

# Lancement de Streamlit en arrière-plan
!streamlit run src/app/app.py &>/content/logs.txt &

# Récupération de l'adresse IP publique pour déverrouiller localtunnel
!curl https://loca.lt/mytunnelpassword

# Création du tunnel d'accès public
!npx localtunnel --port 8501
```
En cliquant sur le lien fourni et en entrant le mot de passe IP généré, votre encadrant accède directement à l'application web servie par les serveurs de Google !
