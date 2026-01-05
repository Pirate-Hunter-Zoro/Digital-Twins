"""Metrics helpers.
Computes and writes correlation stats (e.g., Spearman rho) for judge vs cosine series."""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import pandas as pd
from scipy.stats import spearmanr
from scripts.shared.io import write_text

def write_spearman(out_dir: Path, name: str, series: pd.Series, y: pd.Series) -> Optional[float]:
    mask = ~(series.isna() | y.isna())
    n = int(mask.sum())
    if n < 3:
        write_text(out_dir / f"spearman_rho_{name}.txt", f"{name}: insufficient data (n={n})\n")
        return None
    rho, pval = spearmanr(y[mask], series[mask])
    write_text(out_dir / f"spearman_rho_{name}.txt", f"{name}: n={n}, rho={rho:.4f} (p={pval:.3g})\n")
    return float(rho)