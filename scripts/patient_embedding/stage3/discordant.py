"""Discordant example bundling.
Selects LLM-high/Cos-low and Cos-high/LLM-low pairs and writes markdown + parquet/csv bundles."""

from __future__ import annotations
import math
from pathlib import Path
from typing import Tuple
import pandas as pd
import numpy as np
import os
from patient_embedding.shared.io import read_text, write_text

def pair_md(dfsub: pd.DataFrame, title: str) -> str:
    lines = [f"# {title}", ""]
    for _, row in dfsub.iterrows():
        a, b = row["patient_a_id"], row["patient_b_id"]
        na_path = Path(os.environ['NARRATIVES_DIR']) / f'{a}.md'
        nb_path = Path(os.environ['NARRATIVES_DIR']) / f'{b}.md'

        if not na_path.exists() or not nb_path.exists():
            lines += [f"## {a}  vs  {b}", "- skipped: missing narrative file(s)", "---", ""]
            continue

        na = read_text(na_path); nb = read_text(nb_path)
        cos = row.get(f"cosine", float("nan"))
        cos_txt = "nan" if (cos is None or (isinstance(cos, float) and math.isnan(cos))) else f"{cos:.4f}"
        lines += [
            f"## {a}  vs  {b}",
            f"- judge_score: **{row['judge_score']}**",
            f"- cosine: **{cos_txt}**",
            "",
            "### Narrative A",
            "```markdown", na.strip(), "```",
            "",
            "### Narrative B",
            "```markdown", nb.strip(), "```",
            "", "---", ""
        ]
    return "\n".join(lines)

def write_discordant(out_dir: Path, df_pairs: pd.DataFrame, num_std: int) -> None:
    diff_col = "diff_norm"
    
    differences = df_pairs[diff_col]
    mean_diff = np.mean(differences)
    std_diff = np.std(differences)
    low_threshold = mean_diff - num_std*std_diff
    high_threshold = mean_diff + num_std*std_diff
    
    hl = df_pairs[df_pairs[diff_col] <= low_threshold]
    lh = df_pairs[df_pairs[diff_col] >= high_threshold]
    
    try:
        hl.to_parquet(out_dir / f"discordant_llmHigh_cosLow.parquet", index=False)
        lh.to_parquet(out_dir / f"discordant_cosHigh_llmLow.parquet", index=False)
    except Exception:
        hl.to_csv(out_dir / f"discordant_llmHigh_cosLow.csv", index=False)
        lh.to_csv(out_dir / f"discordant_cosHigh_llmLow.csv", index=False)

    write_text(out_dir / f"discordant_llmHigh_cosLow.md",
                pair_md(hl, "LLM-high / Cosine-low"))
    write_text(out_dir / f"discordant_cosHigh_llmLow.md",
                pair_md(lh, "Cosine-high / LLM-low"))