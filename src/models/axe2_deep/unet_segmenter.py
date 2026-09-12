import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, Dict, Any, List

class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class ThermalUNet(nn.Module):
    """
    Architecture U-Net légère optimisée pour la segmentation précise
    des anomalies et points chauds sur thermogrammes infrarouges.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 1, features: List[int] = [16, 32, 64, 128]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encodeur (Downsampling)
        curr_in = in_channels
        for feature in features:
            self.downs.append(DoubleConv(curr_in, feature))
            curr_in = feature

        # Goulot d'étranglement (Bottleneck)
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Décodeur (Upsampling + Skip connections)
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        # Couche finale de projection
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=True)

            concat_x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_x)

        return torch.sigmoid(self.final_conv(x))

class UNetSegmenter:
    """
    Gestionnaire d'entraînement et d'inférence pour la segmentation thermique U-Net.
    """
    def __init__(self, device: str = None, model_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() and device != 'cpu' else 'cpu')
        self.model = ThermalUNet(in_channels=1, out_channels=1).to(self.device)
        self.model_path = model_path

        if model_path and os.path.exists(model_path):
            self.load_weights(model_path)

    def load_weights(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()

    def save_weights(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def train_on_ground_truth(self, gt_samples: List[Dict[str, Any]],
                              val_samples: List[Dict[str, Any]] = None,
                              epochs: int = 35, lr: float = 1e-3, batch_size: int = 4) -> Dict[str, List[float]]:
        """
        Entraîne le réseau U-Net sur les masques de vérité terrain annotés par les experts.
        Utilise une combinaison de BCE et Dice Loss.
        """
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        history = {'train_loss': [], 'val_dice': []}

        # Préparation des données d'entraînement
        X_train, Y_train = [], []
        for sample in gt_samples:
            img = sample['image_bgr']
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            gray_norm = (gray.astype(np.float32) / 255.0)[np.newaxis, :, :]  # (1, H, W)
            mask_norm = (sample['mask'].astype(np.float32))[np.newaxis, :, :] # (1, H, W)

            X_train.append(gray_norm)
            Y_train.append(mask_norm)

            # Augmentation de données (flip horizontal pour symétrie moteur)
            X_train.append(np.flip(gray_norm, axis=2).copy())
            Y_train.append(np.flip(mask_norm, axis=2).copy())

        X_tensor = torch.tensor(np.array(X_train), dtype=torch.float32).to(self.device)
        Y_tensor = torch.tensor(np.array(Y_train), dtype=torch.float32).to(self.device)

        n_samples = len(X_tensor)
        self.model.train()

        for epoch in range(epochs):
            permutation = torch.randperm(n_samples)
            epoch_loss = 0.0

            for i in range(0, n_samples, batch_size):
                indices = permutation[i:i + batch_size]
                batch_x, batch_y = X_tensor[indices], Y_tensor[indices]

                optimizer.zero_grad()
                pred = self.model(batch_x)

                # Loss composite BCE + Dice Loss
                bce = F.binary_cross_entropy(pred, batch_y)
                intersection = (pred * batch_y).sum(dim=(2, 3))
                dice = (2.0 * intersection + 1e-5) / (pred.sum(dim=(2, 3)) + batch_y.sum(dim=(2, 3)) + 1e-5)
                dice_loss = 1.0 - dice.mean()

                loss = 0.5 * bce + 0.5 * dice_loss
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * len(indices)

            epoch_loss /= n_samples
            history['train_loss'].append(epoch_loss)

            # Évaluation validation si disponible
            if val_samples:
                val_dices = []
                self.model.eval()
                with torch.no_grad():
                    for v in val_samples:
                        pred_m = self.predict_mask(v['image_bgr'])['probability_map']
                        gt_m = v['mask']
                        inter = np.sum((pred_m > 0.5) & (gt_m > 0))
                        dice_v = (2.0 * inter + 1e-5) / (np.sum(pred_m > 0.5) + np.sum(gt_m > 0) + 1e-5)
                        val_dices.append(dice_v)
                self.model.train()
                history['val_dice'].append(float(np.mean(val_dices)))

        self.model.eval()
        return history

    def predict_mask(self, img_bgr: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Infère la carte continue de probabilité et le masque binaire.
        """
        self.model.eval()
        H_orig, W_orig = img_bgr.shape[:2]

        if len(img_bgr.shape) == 3:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_bgr

        # Redimensionnement vers dimensions multiples de 16 pour U-Net
        target_h, target_w = (H_orig // 16) * 16, (W_orig // 16) * 16
        gray_resized = cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_AREA)

        tensor_in = torch.tensor(gray_resized.astype(np.float32) / 255.0, dtype=torch.float32)[None, None, :, :].to(self.device)

        with torch.no_grad():
            prob_tensor = self.model(tensor_in)
            prob_map = prob_tensor.squeeze().cpu().numpy()

        # Remise à l'échelle originale
        prob_map_orig = cv2.resize(prob_map, (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)
        binary_mask = (prob_map_orig >= threshold).astype(np.uint8)
        
        # Intensité thermique dans la zone segmentée
        gray_norm = gray.astype(np.float32) / 255.0
        mean_conf = float(np.mean(prob_map_orig[binary_mask > 0])) if binary_mask.sum() > 0 else 0.0
        
        if binary_mask.sum() > 0:
            hotspot_intensity = float(np.mean(gray_norm[binary_mask > 0]))
            hotspot_max = float(np.max(gray_norm[binary_mask > 0]))
        else:
            hotspot_intensity = float(np.mean(gray_norm))
            hotspot_max = float(np.max(gray_norm))

        # Score d'anomalie thermique : pondération par l'échauffement effectif par rapport au régime nominal froid (~0.16)
        thermal_excess = np.clip((hotspot_max - 0.16) / 0.50, 0.0, 1.0)
        thermal_anomaly_score = float(np.clip(mean_conf * thermal_excess, 0.0, 1.0))

        return {
            'probability_map': prob_map_orig,
            'binary_mask': binary_mask,
            'mean_hotspot_confidence': mean_conf,
            'hotspot_area': int(binary_mask.sum()),
            'hotspot_intensity': round(hotspot_intensity, 4),
            'hotspot_max': round(hotspot_max, 4),
            'thermal_anomaly_score': round(thermal_anomaly_score, 4)
        }
