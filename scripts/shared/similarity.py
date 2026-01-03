"""Vector similarity math.
Cosine distance/utility functions that operate on numpy arrays; safe for NaNs/zeros."""

import os
from pathlib import Path
import numpy as np
from dotenv import load_dotenv
load_dotenv()

VECTORS_DIR = Path(os.environ['VECTORS_DIR'])

def _load_vector(vector_id: str) -> np.array:
    vector_file = VECTORS_DIR / f"vector_{id}.npy"
    if not vector_file.exists():
        raise FileNotFoundError(f"{str(vector_file)} not found...")
    else:
        return np.load(file=vector_file)

def cosine(id_a: str, id_b: str) -> float:
    a = _load_vector(vector_id=id_a).astype(np.float64).ravel()
    b = _load_vector(vector_id=id_b).astype(np.float64).ravel()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    prod = float(np.dot(a, b) / (na * nb))
    