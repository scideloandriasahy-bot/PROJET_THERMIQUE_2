import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
from src.data.dataset_loader import ThermalDatasetLoader
from src.models.axe2_deep.unet_segmenter import UNetSegmenter
from src.data.preprocessing import ThermalPreprocessor

loader = ThermalDatasetLoader('.')
prep = ThermalPreprocessor()
unet = UNetSegmenter(model_path='experiments_results/unet_weights.pt')

seq_noload = loader.load_sequence('Noload')
print('Noload frames:', len(seq_noload))
for f in seq_noload[:5]:
    im = f['image_bgr']
    pre = prep.process_pipeline(im)
    res = unet.predict_mask(im)
    m_int = float(pre['denoised_intensity'].mean())
    max_int = float(pre['denoised_intensity'].max())
    print(f"Noload {f['filename']}: mean={m_int:.4f}, max={max_int:.4f}, unet_area={res['hotspot_area']}, unet_conf={res['mean_hotspot_confidence']:.4f}")

seq_a50 = loader.load_sequence('A50')
print('\nA50 frames:', len(seq_a50))
for f in seq_a50[:5]:
    im = f['image_bgr']
    pre = prep.process_pipeline(im)
    res = unet.predict_mask(im)
    m_int = float(pre['denoised_intensity'].mean())
    max_int = float(pre['denoised_intensity'].max())
    print(f"A50 {f['filename']}: mean={m_int:.4f}, max={max_int:.4f}, unet_area={res['hotspot_area']}, unet_conf={res['mean_hotspot_confidence']:.4f}")
