# scripts/world_2_neighbor_analysis/plot_sanity_set_results.py

import os
import sys
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# --- Dynamic sys.path adjustment! ---
current_script_dir = Path(__file__).resolve().parent
project_root = current_script_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def main():
    """
    The main engine of our Graph-O-Matic! It visualizes the results
    from our sanity set calibration to see if our metrics are working!
    """
    print("🎨✨ Powering up the Graph-O-Matic 5000! ✨🎨")

    # --- Path Setup ---
    data_dir = project_root / "data"
    results_path = data_dir / "sanity_set_calibration_results.json"
    output_path = data_dir / "sanity_set_calibration_plot.png"

    # --- Check for our results file! ---
    if not results_path.exists():
        print(f"❌ OH NOES! I can't find the calibration results at: {results_path}")
        print("Please make sure you've run the calibrate_on_sanity_set.py script first!")
        return

    # --- Load the magnificent calibration data! ---
    print(f"📂 Loading results from {results_path}...")
    df = pd.read_json(results_path)
    print(f"✅ Loaded {len(df)} patient pairs to plot!")

    # --- Let's make some ART! ---
    print("🖌️  Generating the calibration masterpiece...")

    # --- THE BIG CHANGE! ---
    # We're plotting "cosine_similarity" now and updating the labels!
    g = sns.jointplot(
        data=df,
        x="cosine_similarity",
        y="llm_relevance_score",
        hue="pair_type",
        palette={"similar": "blue", "dissimilar": "red"},
        height=10,
        s=50,
        alpha=0.6
    )

    g.fig.suptitle(
        "Calibration of Similarity Metrics on Sanity Set\n(Known Similar vs. Dissimilar Patient Pairs)",
        fontsize=18,
        y=1.03
    )

    g.set_axis_labels("Cosine Similarity (Higher is More Similar)", "LLM Relevance Score (Higher is More Similar)", fontsize=14)

    # --- Save our masterpiece! ---
    print(f"💾 Saving the beautiful plot to: {output_path}")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print("\n🎉 YAY! The plot has been generated! Now we can really SEE the data!")

if __name__ == "__main__":
    main()