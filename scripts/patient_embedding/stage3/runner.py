"""Stage-3 coordinator.
Loads data, builds/refines pairs, runs judging, computes correlations/plots, writes discordant bundles, and persists results."""

from __future__ import annotations
import os, random
from pathlib import Path
import numpy as np, pandas as pd
from typing import Callable

from patient_embedding.shared.io import ensure_dir
from patient_embedding.shared.plots import scatter, histogram
from patient_embedding.shared.metrics import write_spearman
from .cosine_calculator import all_cos_funcs
from .pairs import build_pairs, pair_id
from .persist import read_existing_pairs, write_pairs
from .judging import score_pairs
from .discordant import write_discordant

COUNTER_INTERVAL = 5

def handle_plotting(label: str, plots_dir: Path, judge_array: np.array, cos_array: np.array, diff_array: np.array, norm_judge_array: np.array, norm_cos_array: np.array, diff_norm_array: np.array):
    _ = write_spearman(plots_dir, f"spearman_rho_judge_vs_cos_{label}", cos_array, judge_array)
    _ = write_spearman(plots_dir, f"spearman_rho_norm_judge_vs_norm_cos_{label}", norm_cos_array, norm_judge_array)
         
    histogram(diff_array.tolist(), "Cosine minus Judge", plots_dir / f"histogram_cos_{label}_minus_judge.png")
    
    histogram(cos_array.tolist(), 'Cosine', plots_dir / f"histogram_cos_{label}.png")
    
    histogram(judge_array.tolist(), "Judge", plots_dir / f"histogram_judge_{label}")
    
    scatter(cos_array.tolist(), judge_array.tolist(),
        "Judge vs Cosine", plots_dir / f"scatter_judge_vs_cos_{label}.png",
        "cosine", "judge_score")
    
    histogram(diff_norm_array.tolist(), "Normalized Cosine minus Normalized Judge", plots_dir / f"histogram_normalzied_cos_{label}_minus_normalized_judge.png")
    
    histogram(norm_cos_array.tolist(), 'Normalized Cosine', plots_dir / f"histogram_normalized_cos_{label}.png")
    
    histogram(norm_judge_array.tolist(), "Normalized Judge", plots_dir / f"histogram_normalized_judge_{label}")
    
    scatter(norm_cos_array.tolist(), norm_judge_array.tolist(),
        "Normalized Judge vs Normalized Cosine", plots_dir / f"scatter_normalized_judge_vs_normalized_cos_{label}.png",
        "norm_cosine", "norm_judge_score")
    
def run_analysis(rnd: random.Random, label: str, cos_func: Callable[[str, str], float], out_dir: Path, scrub=False):
    # cos_func will find the cosine similarity of either the entire narrative, just the summary, just the medications, or just the diagnoses - depends on which callable gets passed into this function
    ensure_dir(out_dir)
    
    # Check for previously found cosine calculations
    print(f"[Stage3] checking for pre-existing cosine calculations in {out_dir}...")
    existing_df = read_existing_pairs(out_dir)
    # For pre-existing entries, we must have values for all of these
    columns_to_check = [
        "patient_a_id", 
        "patient_b_id", 
        "cosine", 
        "judge_score", 
        "judge_rationale", 
        "cosine_diff"
    ]
    existing_ids = set()
    if existing_df is not None and not existing_df.empty and not scrub:
        done = existing_df[columns_to_check].notna().all(axis=1)
        if not done.empty:
            existing_ids = set(existing_df.loc[done]
                            .apply(lambda r: pair_id(str(r["patient_a_id"]), str(r["patient_b_id"])), axis=1))
        print(f"[Stage3] resume: {len(existing_ids)} previously scored pairs detected in {out_dir}", flush=True)
    
    # Since we sample pairs based on cosine value distribution, build the pairs based on THIS specific cosine similarity value
    pairs_all = build_pairs(rnd, cos_func)

    pairs_new = [p for p in pairs_all if pair_id(p[0],p[1]) not in existing_ids]
    print(f"[Stage3] will judge {len(pairs_new)} new pairs; skipping {len(existing_ids)} already done in {out_dir}", flush=True)

    rows_new = []
    for i, record in enumerate(score_pairs(pairs_new, segment=label)):
        rows_new.append(record)
        if (i + 1) % COUNTER_INTERVAL == 0:
            print(f"[Stage3] {i+1}/{len(pairs_new)} new pairs scored in {out_dir}...", flush=True)

    df_new = pd.DataFrame(rows_new)
    combined = (pd.concat([existing_df, df_new], ignore_index=True)
                if (existing_df is not None and not existing_df.empty) else df_new)
    write_pairs(out_dir, combined)
    df_pairs = combined
    
    judge_array = df_pairs["judge_score"]
    cos_array = df_pairs["cosine"]
    diff_array = df_pairs["diff"] # Difference between cosine score and judge score
    norm_judge_array = (judge_array - judge_array.min()) / judge_array.ptp()
    norm_cos_array = (cos_array - cos_array.min()) / cos_array.ptp()
    diff_norm_array = norm_cos_array - norm_judge_array # Difference between normalized cosine and normalized judge score
    # Store in the data frame 
    df_pairs["judge_score_norm"] = norm_judge_array
    df_pairs["cosine_norm"] = norm_cos_array
    df_pairs["diff_norm"] = diff_norm_array
    
    plots_dir = out_dir / "plots"
    os.makedirs(plots_dir, exist_ok=True)
    handle_plotting(label, plots_dir, judge_array, cos_array, diff_array, norm_judge_array, norm_cos_array, diff_norm_array)
    write_discordant(
                    out_dir, 
                    df_pairs, 
                    int(os.environ['DISCORDANT_DIFFERENCE_STDEV']), 
                    )

def run() -> None:
    vec_dir  = Path(os.environ['VECTORS_DIR'])
    out_dir  = Path(os.environ['ANALYSIS_DIR'])
    ensure_dir(out_dir)
    
    vec_files = vec_dir.glob("**/*.npy")
    vec_map = {Path(vec_file).stem : Path(vec_file) for vec_file in vec_files}
    cos_full, cos_summary, cos_medications, cos_diagnoses = all_cos_funcs(vec_map)

    rnd = random.Random(int(os.environ['SEED']))
    cos_funcs = {"full": cos_full, "summary":cos_summary, "medications": cos_medications, "diagnoses": cos_diagnoses}
    for label, cos_func in cos_funcs.items():
        run_analysis(rnd, label, cos_func, out_dir / label, scrub=True)

    print("[Stage3] complete.", flush=True)