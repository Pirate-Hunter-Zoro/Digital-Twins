import os
import sys
import json
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from pathlib import Path
from itertools import combinations

# --- Dynamic sys.path adjustment ---
current_script_dir = Path(__file__).resolve().parent
project_root = current_script_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.common.config import setup_config, get_global_config

def main():
    print("🎨✨ Powering up the Correlation Matrix Megaplotter! ✨🎨")
    parser = argparse.ArgumentParser(description="Generate scatter plots and heatmaps from pre-computed pairwise metrics.")
    parser.add_argument("--embedder_type", required=True, choices=["gru", "transformer"], help="The type of pre-computed metrics to use.")
    parser.add_argument("--num_visits", type=int, default=6)
    parser.add_argument("--sample_size", type=int, default=50, help="Number of patient visit sequences to sample for plotting.")
    args = parser.parse_args()

    setup_config("visit_sentence", args.embedder_type, "multiple", args.num_visits, 5000, 0)
    config = get_global_config()

    # --- Path Setup ---
    base_dir = project_root / "data" / "visit_sentence"
    data_dir = base_dir / f"visits_{config.num_visits}" / "patients_5000"
    metrics_path = data_dir / f"all_pairwise_metrics_{args.embedder_type}.json"
    output_dir = data_dir / "plots"
    os.makedirs(output_dir, exist_ok=True)

    if not metrics_path.exists():
        raise FileNotFoundError(f"❌ OH NO! The metrics file for '{args.embedder_type}' was not found at {metrics_path}.")

    print(f"📂 Loading pre-computed metrics from {metrics_path}...")
    df = pd.read_json(metrics_path)

    # --- Sampling Logic ---
    print(f"🔬 Taking a random sample of {args.sample_size} patient visit sequences...")
    all_keys = pd.unique(df[['patient_1', 'patient_2']].values.ravel('K'))
    if len(all_keys) < args.sample_size:
        print(f"⚠️ Warning: Total unique patients ({len(all_keys)}) is less than sample size ({args.sample_size}). Using all patients.")
        sampled_keys = all_keys
    else:
        sampled_keys = random.sample(list(all_keys), args.sample_size)
    
    # Filter the dataframe to only include pairs from our sampled keys
    df_sample = df[df['patient_1'].isin(sampled_keys) & df['patient_2'].isin(sampled_keys)].copy()
    print(f"✅ Created a sample with {len(df_sample)} pairs for analysis.")

    # --- 1, 2, 3: Scatter Plots with Spearman Rho! ---
    metric_pairs = [
        ("euclidean_distance", "cosine_similarity"),
        ("llm_relevance_score", "euclidean_distance"),
        ("llm_relevance_score", "cosine_similarity")
    ]

    for x_metric, y_metric in metric_pairs:
        print(f"🎨 Generating scatter plot: {y_metric} vs. {x_metric}")

        # Calculate Spearman Rho
        rho, p_value = spearmanr(df_sample[x_metric], df_sample[y_metric])

        g = sns.jointplot(data=df_sample, x=x_metric, y=y_metric, kind="reg", height=8)
        g.fig.suptitle(f"{y_metric.replace('_', ' ').title()} vs. {x_metric.replace('_', ' ').title()}\nSpearman's Rho: {rho:.3f} (p={p_value:.3g})", y=1.03)
        
        plot_path = output_dir / f"scatter_{y_metric}_vs_{x_metric}_{args.embedder_type}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Saved plot to {plot_path}")


    # --- 4, 5, 6: Heat Maps! ---
    metrics_for_heatmap = ["euclidean_distance", "cosine_similarity", "llm_relevance_score"]
    
    # We need to create a complete dataframe for pivoting, including self-pairs
    all_pairs_df = pd.DataFrame(list(combinations(sampled_keys, 2)), columns=['patient_1', 'patient_2'])
    
    # Merge with our sampled data
    merged_df = pd.merge(all_pairs_df, df_sample, on=['patient_1', 'patient_2'], how='left')
    
    # Add the reverse pairs to make the matrix symmetric for pivoting
    reversed_df = merged_df.rename(columns={'patient_1': 'patient_2', 'patient_2': 'patient_1'})
    full_symmetric_df = pd.concat([merged_df, reversed_df], ignore_index=True)


    for metric in metrics_for_heatmap:
        print(f"🎨 Generating heatmap for: {metric}")
        
        # Pivot the data to create a square matrix
        heatmap_data = full_symmetric_df.pivot(index='patient_1', columns='patient_2', values=metric)
        
        # Fill diagonal with appropriate values (0 for distance, 1 for similarity)
        np.fill_diagonal(heatmap_data.values, 1 if "similarity" in metric or "score" in metric else 0)

        plt.figure(figsize=(16, 12))
        sns.heatmap(heatmap_data, cmap="viridis")
        plt.title(f"Pairwise {metric.replace('_', ' ').title()} Heatmap\n(Embedder: {args.embedder_type})", fontsize=16)
        
        plot_path = output_dir / f"heatmap_{metric}_{args.embedder_type}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Saved heatmap to {plot_path}")

    print("\n🎉 YAY! The Megaplotter has finished its magnificent work!")

if __name__ == "__main__":
    main()