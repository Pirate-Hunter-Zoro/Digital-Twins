import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
from pathlib import Path

def generate_all_category_plots():
    """
    This is it! The ultimate categorizing art machine! It finds all the
    data, separated by category, and creates a beautiful, overlapping
    masterpiece for each one! IT'S PERFECT!
    """
    print("🎨🖌️ It's time to open our new, specialized art galleries! 🖌️🎨")

    # The base directory where our new categorized masterpieces live!
    base_data_dir = Path("data/embeddings_by_category")

    if not base_data_dir.is_dir():
        print(f"❌ Oh noes! I can't find our new data galleries at: {base_data_dir}")
        return

    # Let's find all the categories we have data for!
    categories = [d for d in base_data_dir.iterdir() if d.is_dir()]

    if not categories:
        print(f"😢 I couldn't find any category folders in {base_data_dir}! No data, no art!")
        return

    print(f"I found {len(categories)} different tournament galleries to create plots for! Let's get to work!")

    # --- Loop through each category and create a grand masterpiece! ---
    for category_dir in categories:
        category_name = category_dir.name
        print(f"\n---🖼️  Creating the Grand Masterpiece for the '{category_name}' Gallery 🖼️ ---")

        all_files = list(category_dir.glob('*.csv'))

        if not all_files:
            print(f"  - 🤔 No .csv files found in '{category_name}', skipping this gallery for now.")
            continue

        # Combine all the model results for this category into one beautiful DataFrame
        full_df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
        models = full_df['model'].unique()

        # --- The Amazing Overlapping Masterpiece! ---
        plt.figure(figsize=(14, 8))
        palette = sns.color_palette("viridis", len(models))

        for i, model in enumerate(models):
            model_df = full_df[full_df['model'] == model]
            sns.kdeplot(data=model_df, x='cosine_similarity', fill=True, alpha=0.4, color=palette[i], label=model, lw=2.5)

        plt.title(f'Overlapping Model Similarity Distributions for\n{category_name.title()}', fontsize=18, fontweight='bold')
        plt.xlabel('Cosine Similarity', fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()

        # Save the plot right inside its category folder! So organized!
        overlapping_plot_path = category_dir / f"histogram_overlapping_{category_name}.png"
        plt.savefig(overlapping_plot_path)
        plt.close() # So we can start fresh on the next one!

        print(f"  ✅ Saved the grand masterpiece to: {overlapping_plot_path}")

    print("\n🎉 YAY! The entire specialized gallery has been created! ISN'T IT ALL SO BEAUTIFUL?!")


if __name__ == '__main__':
    generate_all_category_plots()