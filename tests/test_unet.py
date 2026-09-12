import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import cv2
from src.models.axe2_deep.unet_segmenter import UNetSegmenter

def test_unet_inference():
    weights_path = 'experiments_results/unet_weights.pt'
    assert os.path.exists(weights_path), "Les poids unet_weights.pt doivent exister."
    
    unet = UNetSegmenter(model_path=weights_path)
    dummy_img = np.zeros((240, 320, 3), dtype=np.uint8)
    dummy_img[80:160, 100:220] = [30, 80, 240]

    res = unet.predict_mask(dummy_img)
    assert 'probability_map' in res
    assert 'binary_mask' in res
    assert res['probability_map'].shape == (240, 320)
    assert res['binary_mask'].shape == (240, 320)
    assert 0.0 <= res['thermal_anomaly_score'] <= 1.0

    print("test_unet_inference: SUCCESS")

if __name__ == '__main__':
    test_unet_inference()
