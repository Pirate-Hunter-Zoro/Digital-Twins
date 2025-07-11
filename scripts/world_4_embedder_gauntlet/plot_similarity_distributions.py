import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re

# --- The Magnificent Fix! ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ---------------------------

def generate_all_plots():
    """
    This is it! The final version! The ultimate machine! It finds all the data,
    and then, in one beautiful, glorious sequence, it creates our entire art gallery!
    AHAHAHA! IT'S PERFECT!
    """
    print("IT'S TIME! Let's create an entire art gallery of data!")

    data_dir = os.path.join(project_root, "data", "embeddings")

    if not os.path.isdir(data_dir):
        print(f"❌ Oh noes! I can't find our data-friends at: {data_dir}")
        return

    all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]

    if not all_files:
        print(f"😢 I couldn't find any .csv files in {data_dir}! We need data to make art!")
        return

    print(f"I found {len(all_files)} result files! It's time to get to work!")

    full_df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)

    print("The data-family is all together! Now for the art show!")

    # --- 1. The Magnificent Overall Plot! ---
    print("🎨 First up: The beautiful group photo!")
    plt.figure(figsize=(10, 6))
    sns.histplot(data=full_df, x='cosine_similarity', bins=50, kde=True)
    plt.title('Overall Distribution of All Cosine Similarities', fontsize=16)
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    overall_plot_path = os.path.join(data_dir, "histogram_overall.png")
    plt.savefig(overall_plot_path, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved the group photo to: {overall_plot_path}")

    # --- 2. The Glorious Individual Portraits (in a loop!) ---
    print("🎨 Next: The individual portraits! Everyone gets their own frame!")
    models = full_df['model'].unique()
    for model_name in models:
        print(f"  - Creating a portrait for our star: {model_name}!")
        model_df = full_df[full_df['model'] == model_name]
        plt.figure(figsize=(10, 6))
        sns.histplot(data=model_df, x='cosine_similarity', bins=40, kde=True)
        plt.title(f'Similarity Distribution for\n{model_name}', fontsize=16)
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Frequency')
        plt.xlim(-0.4, 1.2)
        safe_filename = re.sub(r'[^a-zA-Z0-9_.-]', '-', model_name)
        individual_plot_path = os.path.join(data_dir, f"histogram_{safe_filename}.png")
        plt.savefig(individual_plot_path, bbox_inches='tight')
        plt.close()
        print(f"    ✅ Saved a beautiful portrait to: {individual_plot_path}")

    # --- 3. The Amazing Overlapping Masterpiece! ---
    print("🎨 And for the grand finale: The magnificent overlapping masterpiece!")
    plt.figure(figsize=(12, 8))
    sns.histplot(data=full_df, x='cosine_similarity', bins=50, stat='density', kde=True, color='gray', alpha=0.4, label='Overall Distribution')
    palette = sns.color_palette("husl", len(models))
    for i, model in enumerate(models):
        sns.kdeplot(data=full_df[full_df['model'] == model], x='cosine_similarity', fill=True, alpha=0.2, color=palette[i], label=model)
    plt.title('Overlapping Model Distributions vs. Overall', fontsize=16)
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Density')
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    overlapping_plot_path = os.path.join(data_dir, "histograms_overlapping.png")
    plt.savefig(overlapping_plot_path, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved the grand masterpiece to: {overlapping_plot_path}")

    print("\n🎉 YAY! The entire gallery has been created! ISN'T IT BEAUTIFUL?!")


if __name__ == '__main__':
    generate_all_plots()