import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    print("🎨🖌️ It's time to open our new NORMALIZED art galleries! 🖌️🎨")

    # The new base directory for our normalized results!
    base_data_dir = Path("data/normalized_embeddings_by_category")

    if not base_data_dir.is_dir():
        print(f"❌ Oh noes! I can't find our new data galleries at: {base_data_dir}")
        return

    model_dirs = [d for d in base_data_dir.iterdir() if d.is_dir()]
    if not model_dirs:
        print("😢 No model folders found!")
        return

    category_files = {}
    for model_dir in model_dirs:
        model_name = model_dir.name
        for file_path in model_dir.glob("*.json"):
            # Adjusted to handle the new filenames
            category_name = file_path.stem.replace("_normalized_scores", "")
            if category_name not in category_files:
                category_files[category_name] = []
            category_files[category_name].append({"model_name": model_name, "path": file_path})

    # --- Loop through each category and create a grand masterpiece! ---
    for category_name, files in category_files.items():
        print(f"\n---🖼️  Creating the Grand Normalized Masterpiece for the '{category_name}' Gallery 🖼️ ---")

        all_dfs = []
        for file_info in files:
            df = pd.read_json(file_info["path"])
            df['model'] = file_info["model_name"]
            all_dfs.append(df)
        
        if not all_dfs:
            continue
            
        full_df = pd.concat(all_dfs, ignore_index=True)
        models = sorted(full_df['model'].unique())

        plt.figure(figsize=(14, 8))
        palette = sns.color_palette("viridis", len(models))

        for i, model in enumerate(models):
            model_df = full_df[full_df['model'] == model]
            # Plot the new metric!
            sns.kdeplot(data=model_df, x='normalized_similarity', fill=True, alpha=0.4, color=palette[i], label=model, lw=2.5)

        plt.title(f'Overlapping Normalized Similarity Distributions for\n{category_name.title()}', fontsize=18, fontweight='bold')
        plt.xlabel('Normalized Cosine Similarity', fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()

        # Save the plot in a new top-level directory for easy access!
        plot_output_dir = base_data_dir / "_plots"
        os.makedirs(plot_output_dir, exist_ok=True)
        overlapping_plot_path = plot_output_dir / f"histogram_normalized_overlapping_{category_name}.png"
        plt.savefig(overlapping_plot_path)
        plt.close()

        print(f"  ✅ Saved the grand masterpiece to: {overlapping_plot_path}")

if __name__ == "__main__":
    main()