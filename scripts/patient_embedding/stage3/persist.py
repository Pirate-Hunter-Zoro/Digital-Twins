"""Pairs persistence.
Read prior pairs, merge with new results, write {pairs.parquet|csv}, and emit missing-vector counts."""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import pandas as pd
from patient_embedding.shared.io import write_text

def read_existing_pairs(out_dir: Path) -> Optional[pd.DataFrame]:
    pq = out_dir / "pairs.parquet"
    if pq.exists():
        try: 
            return pd.read_parquet(pq)
        except Exception as e: 
            print(f"[Stage3][err] : {e}", flush=True)
    return None

def write_pairs(out_dir: Path, df: pd.DataFrame) -> None:
    p = out_dir / "pairs.parquet"
    try:
        df.to_parquet(p, index=False)
    except Exception as e: 
        print(f"[Stage3][err] : {e}", flush=True)