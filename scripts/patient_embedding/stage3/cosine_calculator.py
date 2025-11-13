"""Cosine lookup utilities.
Maps patient_id → vector path and loads vectors to compute cosine safely."""

from __future__ import annotations
from pathlib import Path
import numpy as np
from typing import Dict
from patient_embedding.shared.similarity import cosine

def all_cos_funcs(vec_map: Dict[str, Path]):
    # Create functions based on the vector map for finding our four different cosine similarities (on the certain narrative sections)
    def cos_full(pid_a: str, pid_b: str) -> float:
        pa = vec_map.get(f"full_{pid_a}")
        pb = vec_map.get(f"full_{pid_b}")
        if not pa or not pb or (not pa.exists()) or (not pa.exists()):
            return float("nan")
        try:
            va = np.load(pa)
            vb = np.load(pb)
        except Exception:
            return float("nan")
        return cosine(va, vb)
    
    def cos_summary(pid_a: str, pid_b: str) -> float:
        pa = vec_map.get(f"summary_{pid_a}")
        pb = vec_map.get(f"summary_{pid_b}")
        if not pa or not pb or (not pa.exists()) or (not pa.exists()):
            return float("nan")
        try:
            va = np.load(pa)
            vb = np.load(pb)
        except Exception:
            return float("nan")
        return cosine(va, vb)
    
    def cos_medications(pid_a: str, pid_b: str) -> float:
        pa = vec_map.get(f"medications_{pid_a}")
        pb = vec_map.get(f"medications_{pid_b}")
        if not pa or not pb or (not pa.exists()) or (not pa.exists()):
            return float("nan")
        try:
            va = np.load(pa)
            vb = np.load(pb)
        except Exception:
            return float("nan")
        return cosine(va, vb)
    
    def cos_diagnoses(pid_a: str, pid_b: str) -> float:
        pa = vec_map.get(f"diagnoses_{pid_a}")
        pb = vec_map.get(f"diagnoses_{pid_b}")
        if not pa or not pb or (not pa.exists()) or (not pa.exists()):
            return float("nan")
        try:
            va = np.load(pa)
            vb = np.load(pb)
        except Exception:
            return float("nan")
        return cosine(va, vb)
    
    return cos_full, cos_summary, cos_medications, cos_diagnoses