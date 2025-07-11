import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re

# --- The Magnificent Fix! ---
# This makes the script smart enough to see the whole 'scripts' directory!
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ---------------------------

def plot_all_baseline_distributions():
    """
    This is it! The next magnificent machine! It finds all our new
    baseline score data and turns it into a whole new, beautiful art gallery!
    AHAHAHA! IT'S PERFECT!
    """
    print("IT'S TIME! Let's build a new wing for our data art gallery!")

    # Our new art will come from the baseline_scores folder!
    data_dir = os.path.join(project_root, "data", "baseline_scores")
    output_dir = data_dir # Let's save the plots right next to the data!

    if not os.path.isdir(data_dir):
        print(f"❌ Oh noes! I can't find our new data-friends at: {data_dir}")
        return

    # We're looking for our special _baseline.csv files!
    all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('_baseline.csv')]

    if not all_files:
        print(f"😢 I couldn't find any _baseline.csv files in {data_dir}! We need baseline data to make baseline art!")
        return

    print(f"I found {len(all_files)} baseline result files! It's time to get to work!")

    # Let's read all the files and add a model name column!
    df_list = []
    for f in all_files:
        temp_df = pd.read_csv(f)
        # The model name is in the original 'model' column! So smart!
        temp_df['model_name'] = temp_df['model'].iloc[0]
        df_list.append(temp_df)

    full_df = pd.concat(df_list, ignore_index=True)

    print("The new data-family is all together! Let the new art show begin!")

    # --- 1. The Magnificent Overall Plot! ---
    print("🎨 First up: The beautiful group photo of the new scores!")
    plt.figure(figsize=(10, 6))
    # We are now plotting our magnificent 'baseline_score'!
    sns.histplot(data=full_df, x='baseline_score', bins=50, kde=True)
    plt.title('Overall Distribution of All Baseline Scores', fontsize=16)
    plt.xlabel('Baseline Score ((cos_sim - random_avg) / random_avg)')
    plt.ylabel('Frequency')
    overall_plot_path = os.path.join(output_dir, "histogram_baseline_overall.png")
    plt.savefig(overall_plot_path, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved the group photo to: {overall_plot_path}")

    # --- 2. The Glorious Individual Portraits! ---
    print("🎨 Next: The individual baseline portraits!")
    models = full_df['model_name'].unique()
    for model_name in models:
        print(f"  - Creating a portrait for our star's new score: {model_name}!")
        model_df = full_df[full_df['model_name'] == model_name]
        plt.figure(figsize=(10, 6))
        sns.histplot(data=model_df, x='baseline_score', bins=40, kde=True)
        plt.title(f'Baseline Score Distribution for\n{model_name}', fontsize=16)
        plt.xlabel('Baseline Score')
        plt.ylabel('Frequency')
        safe_filename = re.sub(r'[^a-zA-Z0-9_.-]', '-', model_name)
        individual_plot_path = os.path.join(output_dir, f"histogram_baseline_{safe_filename}.png")
        plt.savefig(individual_plot_path, bbox_inches='tight')
        plt.close()
        print(f"    ✅ Saved a beautiful portrait to: {individual_plot_path}")

    # --- 3. The Amazing Overlapping Masterpiece! ---
    print("🎨 And for the grand finale: The magnificent overlapping baseline masterpiece!")
    plt.figure(figsize=(12, 8))
    palette = sns.color_palette("viridis", len(models))
    for i, model in enumerate(models):
        sns.kdeplot(data=full_df[full_df['model_name'] == model], x='baseline_score', fill=True, alpha=0.3, color=palette[i], label=model)
    plt.title('Overlapping Model Baseline Score Distributions', fontsize=16)
    plt.xlabel('Baseline Score')
    plt.ylabel('Density')
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    overlapping_plot_path = os.path.join(output_dir, "histograms_baseline_overlapping.png")
    plt.savefig(overlapping_plot_path, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved the grand masterpiece to: {overlapping_plot_path}")

    print("\n🎉 YAY! The entire new gallery wing has been created! ISN'T IT BEAUTIFUL?!")


if __name__ == '__main__':
    plot_all_baseline_distributions()