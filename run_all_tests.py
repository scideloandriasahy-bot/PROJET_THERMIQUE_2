import os
import sys
import unittest

def run_all():
    print("=========================================================")
    print("       EXÉCUTION DE TOUS LES TESTS DU PROJET            ")
    print("=========================================================")

    test_files = [
        'tests.test_preprocessing',
        'tests.test_axe1',
        'tests.test_mrf',
        'tests.test_hmm_absorbing',
        'tests.test_unet',
        'tests.test_hsmm'
    ]

    success = True
    for t_mod in test_files:
        try:
            mod = __import__(t_mod, fromlist=[''])
            # Exécute la fonction de test principale du fichier
            for attr in dir(mod):
                if attr.startswith('test_'):
                    func = getattr(mod, attr)
                    if callable(func):
                        print(f"-> Exécution de {t_mod}.{attr} ...")
                        func()
        except Exception as e:
            print(f"[ERREUR] Échec sur {t_mod}: {e}")
            success = False

    if success:
        print("\n[SUCCESS] TOUS LES TESTS DU PROJET ONT ETE VALIDES AVEC SUCCES !")
    else:
        print("\n[FAILURE] CERTAINS TESTS ONT ECHOUE.")
        sys.exit(1)

if __name__ == '__main__':
    run_all()
