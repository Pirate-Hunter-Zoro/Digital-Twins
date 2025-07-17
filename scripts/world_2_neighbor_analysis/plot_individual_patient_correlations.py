# scripts/world_2_neighbor_analysis/plot_individual_patient_correlations.py

import os
import sys
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
from pathlib import Path

# --- Dynamic sys.path adjustment! ---
current_script_dir = Path(__file__).resolve().parent
project_root = current_script_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.common.config import setup_config, get_global_config

def main():
    # --- ✨ A magnificent new argument parser! ✨ ---
    import argparse
    parser = argparse.ArgumentParser(description="Plot individual patient correlations from a pairwise analysis.")
    parser.add_argument("--representation_method", default="visit_sentence")
    parser.add_argument("--vectorizer_method", required=True)
    parser.add_argument("--distance_metric", default="cosine")
    parser.add_argument("--num_visits", type=int, default=6)
    parser.add_argument("--num_patients", type=int, default=5000)
    parser.add_argument("--num_neighbors", type=int, default=5)
    parser.add_argument("--num_patients_to_plot", type=int, default=4)
    args = parser.parse_args()

    setup_config(
        representation_method=args.representation_method,
        vectorizer_method=args.vectorizer_method,
        distance_metric=args.distance_metric,
        num_visits=args.num_visits,
        num_patients=args.num_patients,
        num_neighbors=args.num_neighbors,
    )
    config = get_global_config()

    print("🔬 Activating the Multi-Scope Inspector! 🔬")

    # --- Path Setup (Now dynamic based on args!) ---
    base_dir = project_root / "data" / config.representation_method / config.vectorizer_method
    vectors_dir = base_dir / f"visits_{config.num_visits}" / f"patients_{config.num_patients}"
    data_dir = vectors_dir / f"metric_{config.distance_metric}" / f"neighbors_{config.num_neighbors}"
    
    # It now knows exactly which results file to look for!
    pairwise_results_path = data_dir / "pairwise_correlation_results.json" 
    output_path = data_dir / "individual_patient_correlation_grid.png"

    if not pairwise_results_path.exists():
        print(f"❌ OH NOES! I can't find the pairwise results at: {pairwise_results_path}")
        return
    
    print(f"📂 Loading pairwise results from {pairwise_results_path}...")
    df = pd.read_json(pairwise_results_path)

    all_patient_ids = df['patient_1'].unique()
    if len(all_patient_ids) < args.num_patients_to_plot:
        selected_patient_ids = all_patient_ids
    else:
        selected_patient_ids = random.sample(list(all_patient_ids), args.num_patients_to_plot)
    
    print(f"Selected patients for plotting: {selected_patient_ids}")
    
    plot_df = df[df['patient_1'].isin(selected_patient_ids)]

    print("🎨 Generating the multi-scope masterpiece...")
    
    g = sns.FacetGrid(plot_df, col="patient_1", col_wrap=2, height=5, sharex=False, sharey=False)
    g.map(sns.scatterplot, "cosine_similarity", "llm_relevance_score", s=100, alpha=0.8)
    
    g.set_titles("Patient: {col_name}", size=14)
    g.set_axis_labels("Cosine Similarity", "LLM Similarity Score", size=12)
    g.fig.suptitle("Per-Patient View of Neighbor Similarity vs. LLM Score", size=20, y=1.03)
    
    plt.tight_layout()

    print(f"💾 Saving the beautiful grid plot to: {output_path}")
    plt.savefig(output_path, dpi=300)
    plt.close()

    print("\n🎉 Individual patient analysis complete!")

if __name__ == "__main__":
    main()