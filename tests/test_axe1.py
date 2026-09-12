import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.data.dataset_loader import ThermalDatasetLoader
from src.data.preprocessing import ThermalPreprocessor
from src.models.axe1_markov.mrf_segmenter import MRFThermalSegmenter
from src.models.axe1_markov.feature_extractor import ThermalFeatureExtractor
from src.models.axe1_markov.hmm_tracker import HMMHealthTracker
from src.models.axe1_markov.absorbing_markov import AbsorbingMarkovDecision

def test_axe1_fan_sequence():
    loader = ThermalDatasetLoader('.')
    prep = ThermalPreprocessor()
    mrf = MRFThermalSegmenter(n_classes=3, beta=1.0, max_iter=8)
    extractor = ThermalFeatureExtractor()
    hmm_model = HMMHealthTracker()
    absorbing = AbsorbingMarkovDecision()

    seq = loader.load_sequence('Fan')[:5]
    assert len(seq) == 5

    prev_probs = None
    for frame in seq:
        pre = prep.process_pipeline(frame['image_bgr'])
        seg = mrf.segment(pre['denoised_intensity'], pre['roi_mask'])
        feats = extractor.extract_features(pre['denoised_intensity'], seg['hotspot_mask'])
        vec = hmm_model.extract_feature_vector(feats)
        probs = hmm_model.filter_step(vec, prev_probs)
        prev_probs = probs
        alert = absorbing.get_graduated_alert(int(np.argmax(probs)), probs, horizon=5)
        
        print(f"Frame {frame['frame_idx']}: Tmax={feats['t_max']} C, Area={feats['area_pixels']}, Level={alert['level']}, Risk={alert['risk_horizon_h']}")
        assert 0.0 <= alert['risk_horizon_h'] <= 1.0
        assert alert['level'] in ['Normal', 'Suspect', 'Confirmé', 'Critique']

    print("Axe 1 test successfully passed!")

if __name__ == '__main__':
    test_axe1_fan_sequence()
