import numpy as np
from typing import Tuple
from pathlib import Path
import os

from scripts.digital_twins.predictions.trd_predictor import TRDPredictor

def load_data_set(patient_ids: set[str]) -> Tuple[np.array, np.array]:
    """Load all the patient vectors and find their labels

    Returns:
        Tuple[np.array, np.array]: vectors and labels
    """
    predictor = TRDPredictor()
    all_vector_paths = Path(os.environ['DETERMINISTIC_VECTORS_DIR']).glob("*.npy")
    patient_vector_paths = [p for p in all_vector_paths if p.stem in patient_ids]
    # Sort for the purposes of keeping X and y consistent with each other
    patient_vector_paths.sort(key=lambda x: x.stem)
    X = np.array([np.load(f, allow_pickle=True) for f in patient_vector_paths])
    y = np.array([predictor.get_trd_status(id) for id in sorted(list(patient_ids))])
    print(f"Shape of X: {X.shape}; Shape of y: {y.shape}", flush=True)
    return (X, y)