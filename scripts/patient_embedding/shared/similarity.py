"""Vector similarity math.
Cosine distance/utility functions that operate on numpy arrays; safe for NaNs/zeros."""

from __future__ import annotations
import numpy as np

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))