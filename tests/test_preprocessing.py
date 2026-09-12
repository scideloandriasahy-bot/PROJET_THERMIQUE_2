import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import cv2
from src.data.preprocessing import ThermalPreprocessor

def test_preprocessing():
    prep = ThermalPreprocessor(target_size=(240, 320))
    # Synthèse d'une image thermique de test
    synthetic_img = np.zeros((240, 320, 3), dtype=np.uint8)
    synthetic_img[50:150, 80:200] = [30, 40, 220]  # Point chaud rouge/orangé

    res = prep.process_pipeline(synthetic_img)

    assert res['denoised_intensity'].shape == (240, 320)
    assert 0.0 <= res['denoised_intensity'].min() <= res['denoised_intensity'].max() <= 1.0
    assert res['roi_mask'].shape == (240, 320)
    assert res['roi_mask'].max() == 1
    print("test_preprocessing: SUCCESS")

if __name__ == '__main__':
    test_preprocessing()
