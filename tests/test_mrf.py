import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.models.axe1_markov.mrf_segmenter import MRFThermalSegmenter

def test_mrf_segmenter():
    mrf = MRFThermalSegmenter(n_classes=3, beta=1.0, max_iter=8)
    
    # Création d'une image avec fond froid (0.1) et point chaud régulier (0.9)
    intensity = np.full((64, 64), 0.1, dtype=np.float32)
    intensity[20:44, 20:44] = 0.9
    # Ajout d'un pixel isolé de bruit
    intensity[10, 10] = 0.85

    res = mrf.segment(intensity)
    hotspot = res['hotspot_mask']

    assert hotspot.shape == (64, 64)
    # Le pixel isolé de bruit doit être éliminé par la régularisation de Potts
    assert hotspot[10, 10] == 0
    # Le cœur du point chaud doit être correctement détecté
    assert hotspot[30, 30] == 1
    print("test_mrf_segmenter: SUCCESS (ICM convergence & spatial regularization verified)")

if __name__ == '__main__':
    test_mrf_segmenter()
