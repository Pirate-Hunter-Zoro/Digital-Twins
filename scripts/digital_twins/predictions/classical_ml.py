import numpy as np
from typing import Tuple
from pathlib import Path
import random
import os

from scripts.digital_twins.predictions.trd_predictor import TRDPredictor

def load_data_set() -> Tuple[np.array, np.array]:
    """Load all the patient vectors and find their labels

    Returns:
        Tuple[np.array, np.array]: vectors and labels
    """
    predictor = TRDPredictor()
    all_vector_paths = Path(os.environ['DETERMINISTIC_VECTORS_DIR']).glob("*.npy")
    all_patient_ids = [v_path.stem for v_path in all_vector_paths]
    X = np.array([np.load(f) for f in all_vector_paths])
    y = np.array([predictor.get_trd_status(id) for id in all_patient_ids])
    print(f"Shape of X: {X.shape}; Shape of y: {y.shape}", flush=True)
    return (X, y)