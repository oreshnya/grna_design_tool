import sys
import numpy as np
import sklearn.ensemble._gb as new_gb

def safe_import_azimuth():
    # Патчим старый модуль, чтобы pickle-файл не падал
    sys.modules["sklearn.ensemble.gradient_boosting"] = new_gb

    # Только теперь импортируем predict
    from azimuth.model_comparison import predict as azimuth_predict

    def wrapped_predict(sgrna_windows: list[str]) -> list[float]:
        sgrna_array = np.array(sgrna_windows)
        n = len(sgrna_array)
        aa_cut = np.full(n, -1)
        percent_peptide = np.full(n, -1)
        return azimuth_predict(
            seq=sgrna_array,
            aa_cut=aa_cut,
            percent_peptide=percent_peptide,
            pam_audit=True,
            length_audit=False
        )
    
    return wrapped_predict