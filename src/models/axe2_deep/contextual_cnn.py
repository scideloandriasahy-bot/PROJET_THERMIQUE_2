import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, Any, Tuple, List

class ContextualAnomalyNet(nn.Module):
    """
    Réseau neuronal combinant la représentation visuelle du thermogramme
    et le contexte opérationnel (charge machine, température ambiante).
    """
    def __init__(self, context_dim: int = 2, dropout_rate: float = 0.3):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # 128 -> 64
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                        # 64 -> 32

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 32 -> 16
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                        # 16 -> 8

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # 8 -> 8
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))                            # 64 x 1 x 1
        )

        self.dropout = nn.Dropout(p=dropout_rate)
        
        # Fusion des caractéristiques visuelles (64) + statistiques thermiques physiques (2: mean, max) + contexte (context_dim)
        self.fc = nn.Sequential(
            nn.Linear(64 + 2 + context_dim, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(32, 1)  # Logit d'anomalie
        )

    def forward(self, img_tensor: torch.Tensor, context_tensor: torch.Tensor) -> torch.Tensor:
        vis_feat = self.conv_layers(img_tensor)
        vis_feat = vis_feat.view(vis_feat.size(0), -1)  # (B, 64)
        
        # Caractéristiques physiques globales directes (mean, max)
        mean_stat = img_tensor.mean(dim=(2, 3)) # (B, 1)
        max_stat = img_tensor.amax(dim=(2, 3))  # (B, 1)

        fused = torch.cat([vis_feat, mean_stat, max_stat, context_tensor], dim=1)  # (B, 68)
        logit = self.fc(fused)
        return logit

class ContextualCNNClassifier:
    """
    Classifieur contextuel avec calibration de probabilités (Temperature Scaling)
    et quantification de l'incertitude par Monte Carlo Dropout.
    """
    def __init__(self, device: str = None, model_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() and device != 'cpu' else 'cpu')
        self.model = ContextualAnomalyNet(context_dim=2, dropout_rate=0.3).to(self.device)
        self.temperature = 1.0  # Paramètre de calibration

        if model_path and os.path.exists(model_path):
            self.load_weights(model_path)

    def load_weights(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.temperature = checkpoint.get('temperature', 1.0)
        self.model.eval()

    def save_weights(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'state_dict': self.model.state_dict(),
            'temperature': self.temperature
        }, path)

    def _prepare_inputs(self, img_bgr: np.ndarray, load_level: float = 0.5,
                        ambient_temp: float = 23.0) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(img_bgr.shape) == 3:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_bgr

        gray_resized = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
        img_norm = (gray_resized.astype(np.float32) / 255.0)[None, None, :, :]

        # Contexte normalisé : [charge entre 0.0 et 1.0, température ambiante normalisée]
        norm_amb = (ambient_temp - 20.0) / 25.0
        context_arr = np.array([[load_level, norm_amb]], dtype=np.float32)

        return (torch.tensor(img_norm, dtype=torch.float32).to(self.device),
                torch.tensor(context_arr, dtype=torch.float32).to(self.device))

    def train_classifier(self, train_df, val_df = None, epochs: int = 25, lr: float = 1e-3, batch_size: int = 16):
        """
        Entraîne le classifieur contextuel sur le dataset multi-charges (HNL, HML, HFL vs CNL, CML, CFL).
        """
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-3)
        criterion = nn.BCEWithLogitsLoss()

        # Construction du batch d'entraînement
        X_imgs, X_ctx, Y_labels = [], [], []
        for _, row in train_df.iterrows():
            img = cv2.imread(row['path'])
            if img is None:
                continue
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            gray_res = cv2.resize(gray, (128, 128))[None, :, :] / 255.0
            ctx = [row['load_level'], (23.0 - 20.0) / 25.0]
            
            X_imgs.append(gray_res)
            X_ctx.append(ctx)
            Y_labels.append([float(row['health_label'])])

        X_img_tensor = torch.tensor(np.array(X_imgs), dtype=torch.float32).to(self.device)
        X_ctx_tensor = torch.tensor(np.array(X_ctx), dtype=torch.float32).to(self.device)
        Y_tensor = torch.tensor(np.array(Y_labels), dtype=torch.float32).to(self.device)

        n_samples = len(X_img_tensor)
        self.model.train()

        for epoch in range(epochs):
            perm = torch.randperm(n_samples)
            for i in range(0, n_samples, batch_size):
                idx = perm[i:i + batch_size]
                b_img, b_ctx, b_y = X_img_tensor[idx], X_ctx_tensor[idx], Y_tensor[idx]

                optimizer.zero_grad()
                logits = self.model(b_img, b_ctx)
                loss = criterion(logits, b_y)
                loss.backward()
                optimizer.step()

        # Calibration post-hoc par Temperature Scaling sur validation
        self.calibrate_temperature(val_df if val_df is not None else train_df)
        self.model.eval()

    def calibrate_temperature(self, calib_df):
        """Ajuste la température T pour minimiser l'Expected Calibration Error."""
        logits_list, labels_list = [], []
        self.model.eval()

        with torch.no_grad():
            for _, row in calib_df.iterrows():
                img = cv2.imread(row['path'])
                if img is None:
                    continue
                t_img, t_ctx = self._prepare_inputs(img, row['load_level'], 23.0)
                logit = self.model(t_img, t_ctx).item()
                logits_list.append(logit)
                labels_list.append(row['health_label'])

        logits_arr = np.array(logits_list)
        labels_arr = np.array(labels_list)

        # Recherche de T optimal par descente 1D
        best_t, best_loss = 1.0, float('inf')
        for t_candidate in np.linspace(0.5, 3.0, 50):
            probs = 1.0 / (1.0 + np.exp(-logits_arr / t_candidate))
            probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
            bce_loss = -np.mean(labels_arr * np.log(probs) + (1.0 - labels_arr) * np.log(1.0 - probs))
            if bce_loss < best_loss:
                best_loss = bce_loss
                best_t = t_candidate

        self.temperature = float(best_t)

    def predict_with_uncertainty(self, img_bgr: np.ndarray, load_level: float = 0.5,
                                 ambient_temp: float = 23.0, n_mc: int = 10) -> Dict[str, Any]:
        """
        Inférence avec Monte Carlo Dropout pour quantifier l'incertitude épistémique.
        Conforme à l'exigence 6.4 : 'Les cas incertains doivent être signalés'.
        """
        t_img, t_ctx = self._prepare_inputs(img_bgr, load_level, ambient_temp)

        # Active le dropout en mode inférence stochastique
        self.model.train()
        mc_predictions = []

        with torch.no_grad():
            for _ in range(n_mc):
                logit = self.model(t_img, t_ctx).item()
                calibrated_prob = 1.0 / (1.0 + np.exp(-logit / self.temperature))
                mc_predictions.append(calibrated_prob)

        self.model.eval()

        mc_predictions = np.array(mc_predictions)
        mean_prob = float(np.mean(mc_predictions))
        uncertainty = float(np.std(mc_predictions))

        # Alerte si incertitude élevée ou probabilité indécise
        needs_human_review = bool(uncertainty > 0.08 or (0.38 <= mean_prob <= 0.62))

        return {
            'calibrated_prob': round(mean_prob, 4),
            'epistemic_uncertainty': round(uncertainty, 4),
            'confidence': round(1.0 - min(1.0, 2.0 * uncertainty), 4),
            'needs_human_review': needs_human_review,
            'load_level': load_level,
            'ambient_temp': ambient_temp
        }
