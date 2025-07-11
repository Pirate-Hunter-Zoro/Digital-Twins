import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def generate_all_the_histograms():
    """
    This is my magnificent creation! It finds all our data, makes a home for
    our art, and then creates THREE different, beautiful kinds of pictures!
    It's the best machine ever! WOOOO!
    """
    print("IT'S TIME! Let's make some of the best data-art the world has ever seen!")

    # --- This is the super-smart, self-aware pathing part! ---
    # It finds where it lives, so it can always find its way home!
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # The project's main folder is two steps up!
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    # And our beautiful data lives in the data/embeddings folder!
    data_dir = os.path.join(project_root, "data", "embeddings")

    # A quick check to make sure the data is actually there!
    if not os.path.isdir(data_dir):
        print(f"❌ Oh noes! I can't find our data-friends at: {data_dir}")
        return

    # Let's go find all our beautiful CSV files!
    all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]

    if not all_files:
        print(f"😢 I couldn't find any .csv files in {data_dir}! We need data to make art!")
        return

    print(f"I found {len(all_files)} result files! It's time to get to work!")

    # Let's smash all the data together into one giant, happy data-family!
    full_df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)

    print("The data-family is all together! Now for the art show!")

    # --- 1. The Magnificent Overall Plot! ---
    print("🎨 First up: The beautiful group photo!")
    plt.figure(figsize=(10, 6))
    sns.histplot(data=full_df, x='cosine_similarity', bins=30, kde=True)
    plt.title('Overall Distribution of All Cosine Similarities', fontsize=16)
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    overall_plot_path = os.path.join(data_dir, "histogram_overall.png")
    plt.savefig(overall_plot_path)
    plt.close() # We close it so it doesn't show up in our next picture!
    print(f"✅ Saved the group photo to: {overall_plot_path}")

    # --- 2. The Glorious Individual Portraits! ---
    print("🎨 Next: The individual portraits! Everyone gets a turn to shine!")
    g = sns.FacetGrid(full_df, col="model", col_wrap=4, sharex=True, sharey=False, height=4)
    g.map(sns.histplot, "cosine_similarity", bins=30, kde=False)
    g.set_axis_labels("Cosine Similarity", "Frequency")
    g.set_titles(col_template="{col_name}", size=8)
    g.fig.suptitle("Individual Similarity Distributions by Model", y=1.03, size=16)
    individual_plot_path = os.path.join(data_dir, "histograms_individual.png")
    plt.savefig(individual_plot_path, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved all the beautiful portraits to: {individual_plot_path}")

    # --- 3. The Amazing Overlapping Masterpiece! ---
    print("🎨 And for the grand finale: The magnificent overlapping masterpiece!")
    plt.figure(figsize=(12, 8))
    # First, we draw the big, beautiful overall distribution in the background!
    sns.histplot(data=full_df, x='cosine_similarity', bins=30, stat='density', kde=True, color='gray', alpha=0.4, label='Overall Distribution')
    
    # Now, we loop through and draw each model's distribution on top!
    models = full_df['model'].unique()
    palette = sns.color_palette("husl", len(models)) # A beautiful color for everyone!
    
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

if __name__ == '__main__':
    generate_all_the_histograms()