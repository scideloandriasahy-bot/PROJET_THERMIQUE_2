import os
import glob
import cv2
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Generator
from sklearn.model_selection import GroupKFold

class ThermalDatasetLoader:
    """
    Gestionnaire et chargeur des jeux de données d'images thermiques :
    1. Séquences temporelles BNUT (Dali-tech T8, défauts stator, rotor, ventilation)
    2. Masques de vérité terrain d'experts (Subset_40_Thermal_GT)
    3. Dataset multi-charges (FLIR C5, charges NL, ML, FL et défauts de roulement)
    """

    SEQUENCE_HEALTH_MAP = {
        'Noload': {'state': 0, 'label': 'Normal', 'severity': 0.0},
        'A10': {'state': 1, 'label': 'Suspect', 'severity': 0.25},
        'A&C10': {'state': 1, 'label': 'Suspect', 'severity': 0.30},
        'A&C&B10': {'state': 1, 'label': 'Suspect', 'severity': 0.35},
        'A30': {'state': 2, 'label': 'Confirmed', 'severity': 0.55},
        'A&C30': {'state': 2, 'label': 'Confirmed', 'severity': 0.65},
        'A&C&B30': {'state': 2, 'label': 'Confirmed', 'severity': 0.70},
        'Fan': {'state': 2, 'label': 'Confirmed', 'severity': 0.60},
        'A50': {'state': 3, 'label': 'Critical', 'severity': 0.90},
        'A&B50': {'state': 3, 'label': 'Critical', 'severity': 0.95},
        'Rotor-0': {'state': 3, 'label': 'Critical', 'severity': 1.00}
    }

    MULTI_LOAD_MAP = {
        'HNL': {'health': 0, 'load': 0.0, 'name': 'Healthy No Load'},
        'HML': {'health': 0, 'load': 0.5, 'name': 'Healthy Medium Load'},
        'HFL': {'health': 0, 'load': 1.0, 'name': 'Healthy Full Load'},
        'CNL': {'health': 1, 'load': 0.0, 'name': 'Corroded No Load'},
        'CML': {'health': 1, 'load': 0.5, 'name': 'Corroded Medium Load'},
        'CFL': {'health': 1, 'load': 1.0, 'name': 'Corroded Full Load'},
    }

    def __init__(self, root_dir: str = '.'):
        self.root_dir = root_dir
        self.seq_dir = os.path.join(root_dir, 'Thermal image of equipment (Induction Motor) + 40 Ground Truths added', 'IR-Motor-bmp')
        self.gt_dir = os.path.join(root_dir, 'Thermal image of equipment (Induction Motor) + 40 Ground Truths added', '40_GT_with_Refrences', 'Subset_40_Thermal_GT')
        self.multi_load_dir = os.path.join(root_dir, 'Induction Motor-Thermal Images')

    def get_available_sequences(self) -> List[str]:
        """Retourne la liste ordonnée des séquences temporelles disponibles."""
        if not os.path.exists(self.seq_dir):
            return []
        return sorted([d for d in os.listdir(self.seq_dir) if os.path.isdir(os.path.join(self.seq_dir, d))])

    def load_sequence(self, seq_name: str) -> List[Dict[str, any]]:
        """
        Charge une séquence temporelle ordonnée d'images thermiques.
        Chaque élément contient :
        - frame_idx : numéro d'ordre dans la séquence
        - filename : nom du fichier
        - path : chemin complet
        - image_bgr : image brute
        - state : état de santé discret (0=Normal, 1=Suspect, 2=Confirmé, 3=Critique)
        - label : libellé textuel
        - severity : score continu théorique
        """
        seq_path = os.path.join(self.seq_dir, seq_name)
        if not os.path.exists(seq_path):
            raise FileNotFoundError(f"Séquence introuvable: {seq_path}")

        files = sorted(os.listdir(seq_path))
        seq_info = self.SEQUENCE_HEALTH_MAP.get(seq_name, {'state': 1, 'label': 'Unknown', 'severity': 0.5})
        
        sequence_data = []
        for idx, f in enumerate(files):
            if not f.lower().endswith(('.bmp', '.jpg', '.png', '.jpeg')):
                continue
            full_path = os.path.join(seq_path, f)
            img = cv2.imread(full_path)
            if img is None:
                continue
            
            # Pour la séquence Fan, la température augmente progressivement
            if seq_name == 'Fan':
                progression = idx / max(1, len(files) - 1)
                if progression < 0.25:
                    cur_state = 0
                    cur_label = 'Normal'
                elif progression < 0.60:
                    cur_state = 1
                    cur_label = 'Suspect'
                elif progression < 0.85:
                    cur_state = 2
                    cur_label = 'Confirmed'
                else:
                    cur_state = 3
                    cur_label = 'Critical'
                cur_severity = float(progression)
            else:
                cur_state = seq_info['state']
                cur_label = seq_info['label']
                cur_severity = seq_info['severity']

            sequence_data.append({
                'seq_name': seq_name,
                'frame_idx': idx,
                'filename': f,
                'path': full_path,
                'image_bgr': img,
                'state': cur_state,
                'label': cur_label,
                'severity': cur_severity,
                'ambient_temp': 23.0  # Dali-tech température ambiante déclarée
            })
        return sequence_data

    def load_ground_truth_dataset(self) -> List[Dict[str, any]]:
        """
        Charge les 40 paires d'images thermiques et masques de vérité terrain experts.
        """
        if not os.path.exists(self.gt_dir):
            raise FileNotFoundError(f"Dossier vérité terrain introuvable: {self.gt_dir}")

        bmps = sorted(glob.glob(os.path.join(self.gt_dir, '*.bmp')))
        gt_samples = []

        for bmp_path in bmps:
            base_name = os.path.splitext(os.path.basename(bmp_path))[0]
            tiff_path = os.path.join(self.gt_dir, f"{base_name}.tiff")
            if not os.path.exists(tiff_path):
                continue
            
            img = cv2.imread(bmp_path)
            gt_img = cv2.imread(tiff_path)
            if img is None or gt_img is None:
                continue
            
            # Masque binaire : pixels blancs (>128) correspondent à la zone chaude annotée
            mask = (gt_img.mean(axis=2) > 128).astype(np.uint8)

            # Identification de la condition d'origine
            cond_found = 'Unknown'
            for cond_name in self.SEQUENCE_HEALTH_MAP.keys():
                seq_folder = os.path.join(self.seq_dir, cond_name)
                if os.path.exists(os.path.join(seq_folder, f"{base_name}.bmp")):
                    cond_found = cond_name
                    break

            gt_samples.append({
                'id': base_name,
                'image_bgr': img,
                'mask': mask,
                'condition': cond_found,
                'hotspot_area': int(mask.sum()),
                'hotspot_ratio': float(mask.mean())
            })

        return gt_samples

    def load_multi_load_dataset(self) -> pd.DataFrame:
        """
        Charge les métadonnées et chemins d'accès du jeu de données FLIR multi-charges.
        """
        if not os.path.exists(self.multi_load_dir):
            return pd.DataFrame()

        records = []
        for cond_folder, meta in self.MULTI_LOAD_MAP.items():
            folder_path = os.path.join(self.multi_load_dir, cond_folder)
            if not os.path.exists(folder_path):
                continue
            
            files = sorted(os.listdir(folder_path))
            for f in files:
                if not f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                records.append({
                    'filename': f,
                    'path': os.path.join(folder_path, f),
                    'condition_code': cond_folder,
                    'health_label': meta['health'],  # 0=Healthy, 1=Corroded/Faulty
                    'load_level': meta['load'],      # 0.0=NL, 0.5=ML, 1.0=FL
                    'condition_name': meta['name']
                })

        return pd.DataFrame(records)

    def get_group_splits(self, n_splits: int = 4) -> Generator[Tuple[List[str], List[str]], None, None]:
        """
        Générateur de découpage croisé groupé par séquence.
        Garantit qu'aucune séquence n'est à la fois dans le train et le test (exigence stricte anti-fuite).
        """
        sequences = self.get_available_sequences()
        groups = np.arange(len(sequences))
        gkf = GroupKFold(n_splits=n_splits)

        for train_idx, test_idx in gkf.split(sequences, sequences, groups):
            train_seqs = [sequences[i] for i in train_idx]
            test_seqs = [sequences[i] for i in test_idx]
            yield train_seqs, test_seqs
