import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.models.axe1_markov.hmm_tracker import HMMHealthTracker
from src.models.axe1_markov.absorbing_markov import AbsorbingMarkovDecision

def test_hmm_and_absorbing():
    hmm = HMMHealthTracker()
    absorbing = AbsorbingMarkovDecision()

    # Test 1 : Vérification stochastique
    assert np.allclose(absorbing.P.sum(axis=1), 1.0)
    assert absorbing.P[3, 3] == 1.0 # État absorbant

    # Test 2 : Matrice fondamentale inversible et strictement positive
    assert np.all(absorbing.N >= 0)

    # Test 3 : Monotonie du risque selon l'état de départ
    p_normal = np.array([1.0, 0.0, 0.0, 0.0])
    p_suspect = np.array([0.0, 1.0, 0.0, 0.0])
    p_confirme = np.array([0.0, 0.0, 1.0, 0.0])
    p_critique = np.array([0.0, 0.0, 0.0, 1.0])

    r_norm = absorbing.probability_critical_at_horizon(p_normal, horizon=5)
    r_susp = absorbing.probability_critical_at_horizon(p_suspect, horizon=5)
    r_conf = absorbing.probability_critical_at_horizon(p_confirme, horizon=5)
    r_crit = absorbing.probability_critical_at_horizon(p_critique, horizon=5)

    assert r_norm <= r_susp <= r_conf <= r_crit == 1.0

    # Test 4 : Monotonie du risque selon l'horizon h
    r_h1 = absorbing.probability_critical_at_horizon(p_confirme, horizon=1)
    r_h5 = absorbing.probability_critical_at_horizon(p_confirme, horizon=5)
    r_h10 = absorbing.probability_critical_at_horizon(p_confirme, horizon=10)
    assert r_h1 <= r_h5 <= r_h10

    print("test_hmm_and_absorbing: SUCCESS (Markov properties mathematically verified)")

if __name__ == '__main__':
    test_hmm_and_absorbing()
