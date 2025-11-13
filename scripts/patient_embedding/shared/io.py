"""Filesystem I/O helpers.
Thin wrappers for text/json/npy read/write, mkdir, size checks, and vector norms.
Used across all stages; no side effects beyond the filesystem."""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any

import numpy as np


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def read_text(p: Path) -> str:
    return Path(p).read_text(encoding="utf-8")

def write_text(p: Path, s: str) -> None:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8")

def write_npy(p: Path, arr: np.ndarray) -> None:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, arr)

def nonempty(p: Path) -> bool:
    return p.exists() and p.stat().st_size > 0