"""Cosine lookup utilities.
Maps patient_id → vector path and loads vectors to compute cosine safely."""

from pathlib import Path
import numpy as np
from typing import Dict
from scripts.digital_twins.shared.similarity import cosine

def create_cos_factory(vec_map: Dict[str, Path]):
    # Create functions based on the vector map for finding our four different cosine similarities (on the certain narrative sections)
    def cos(pid_a: str, pid_b: str) -> float:
        pa = vec_map.get(pid_a)
        pb = vec_map.get(pid_b)
        if not pa or not pb or (not pa.exists()) or (not pa.exists()):
            return float("nan")
        try:
            va = np.load(pa)
            vb = np.load(pb)
        except Exception:
            return float("nan")
        return cosine(va, vb)
    
    return cos