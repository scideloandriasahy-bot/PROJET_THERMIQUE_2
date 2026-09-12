import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.axe2_deep.hsmm_tracker import HSMMHealthTracker

def test_hsmm_transient_filtering():
    hsmm = HSMMHealthTracker()

    # Scénario : Régime nominal (0.10) puis pic transitoire isolé de 1 trame (0.85) puis retour au calme (0.10)
    scores = [0.10, 0.10, 0.85, 0.10, 0.10]
    results = hsmm.decode_sequence(scores)

    # La 3ème trame (pic transitoire) ne doit pas être validée comme persistante
    assert results[2]['is_persistent'] == False
    assert results[2]['dwell_time'] == 1

    # Scénario : Dégradation progressive soutenue
    sustained_scores = [0.10, 0.10, 0.75, 0.78, 0.80, 0.82, 0.85]
    results_sustained = hsmm.decode_sequence(sustained_scores)

    # Après plusieurs trames chaudes, la persistance doit être validée
    assert results_sustained[-1]['is_persistent'] == True
    assert results_sustained[-1]['state'] in [2, 3]

    print("test_hsmm_transient_filtering: SUCCESS (Transient spike rejection verified)")

if __name__ == '__main__':
    test_hsmm_transient_filtering()
