"""Discordant example bundling.
Selects LLM-high/Cos-low and Cos-high/LLM-low pairs and writes markdown + parquet/csv bundles."""

from __future__ import annotations
import math
from pathlib import Path
import pandas as pd
import os
import numpy as np
from scripts.digital_twins.shared.io import read_text, write_text

from dotenv import load_dotenv

load_dotenv()

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

def write_discordant(out_dir: Path, df_pairs: pd.DataFrame) -> None:
    low_percentile = int(os.environ['DISCORDANT_LOW_PERCENTILE'])
    low_percentile_cos = np.percentile(df_pairs['cosine'], low_percentile)
    low_percentile_judge = np.percentile(df_pairs['judge_score'], low_percentile)
    
    high_percentile = int(os.environ['DISCORDANT_HIGH_PERCENTILE'])
    high_percentile_cos = np.percentile(df_pairs['cosine'], high_percentile)
    high_percentile_judge = np.percentile(df_pairs['judge_score'], high_percentile)
    
    hl = df_pairs[(df_pairs['judge_score'] >= high_percentile_judge) & (df_pairs['cosine'] <= low_percentile_cos)]
    lh = df_pairs[(df_pairs['judge_score'] <= low_percentile_judge) & (df_pairs['cosine'] >= high_percentile_cos)]
    
    hl.to_parquet(out_dir / f"discordant_llmHigh_cosLow.parquet", index=False)
    lh.to_parquet(out_dir / f"discordant_cosHigh_llmLow.parquet", index=False)
    
    write_text(out_dir / f"discordant_llmHigh_cosLow.md",
                pair_md(hl, "LLM-high / Cosine-low"))
    write_text(out_dir / f"discordant_cosHigh_llmLow.md",
                pair_md(lh, "Cosine-high / LLM-low"))