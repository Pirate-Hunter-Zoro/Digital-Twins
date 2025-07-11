import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import itertools
import argparse
from sentence_transformers import SentenceTransformer

def test_category_purity_of_champion():
    """
    This is it! My magnum opus! This script will take our champion model,
    read all our beautiful, organized data-files, and see if our champion
    is smart enough to keep its ideas in the right drawers! AHAHAHA!
    """
    print("🤖✨ Welcome to the ULTIMATE CATEGORY PURITY GAUNTLET! Let's begin! ✨🤖")

    # --- Self-aware pathing! It's so smart! ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    data_dir = os.path.join(project_root, "data")
    output_dir = os.path.join(project_root, "results", "category_purity")
    model_dir = "/media/scratch/mferguson/models"

    # --- The Champion We're Testing! ---
    # We get the champion's name from our launcher script! So dynamic!
    parser = argparse.ArgumentParser(description="Run the Category Purity Gauntlet for a champion model!")
    parser.add_argument("--champion_model", type=str, required=True, help="The Hugging Face name of the model to test.")
    args = parser.parse_args()
    CHAMPION_MODEL = args.champion_model

    print(f"Our champion today is the magnificent: {CHAMPION_MODEL}!")

    # Let's make sure our output folder for the art exists!
    os.makedirs(output_dir, exist_ok=True)

    # --- Loading all our beautiful data-files! ---
    try:
        print("📚 Loading our magnificent medications...")
        med_df = pd.read_csv(os.path.join(data_dir, "medication_frequency.csv"))
        med_df['category'] = 'medication'
        # We only need the name and the category! So efficient!
        med_df = med_df[['MedSimpleGenericName', 'category']].rename(columns={'MedSimpleGenericName': 'term'})

        print("🛠️ Loading our powerful procedures...")
        proc_df = pd.read_csv(os.path.join(data_dir, "procedure_frequency.csv"))
        proc_df['category'] = 'procedure'
        proc_df = proc_df[['CPT_Procedure_Description', 'category']].rename(columns={'CPT_Procedure_Description': 'term'})

        print("🩺 Loading our detailed diagnoses...")
        diag_df = pd.read_csv(os.path.join(data_dir, "diagnosis_frequency.csv"))
        diag_df['category'] = 'diagnosis'
        diag_df = diag_df[['Description', 'category']].rename(columns={'Description': 'term'})

        print("✅ All data-files loaded and organized!")
    except FileNotFoundError as e:
        print(f"❌ OH NO! A data-file is missing! I can't find {e.filename}!")
        return
        
    # --- Now let's smash them all together into one SUPER-DATAFRAME! ---
    concepts_df = pd.concat([med_df, proc_df, diag_df], ignore_index=True).dropna(subset=['term'])

    print("🧠 Loading our champion model... This is the heavy lifting part!")
    safe_model_name = CHAMPION_MODEL.replace("/", "-")
    model_path = os.path.join(model_dir, safe_model_name)
    model = SentenceTransformer(model_path)
    print("✅ Champion is loaded and ready for the test!")

    print("🤔 Embedding all the terms... The gears are turning!")
    unique_terms = concepts_df['term'].unique()
    term_embeddings = {term: model.encode(term) for term in unique_terms}
    concepts_df['embedding'] = concepts_df['term'].map(term_embeddings)

    # --- Now for the real test! Intra vs. Inter! ---
    intra_category_scores = []
    inter_category_scores = []

    print("⚔️ The test begins! This is going to be SO MUCH DATA!")
    # To be smart and not run forever, we'll take a random sample of terms to compare!
    sample_df = concepts_df.sample(n=1000, random_state=42)
    all_term_pairs = list(itertools.combinations(sample_df.index, 2))

    for i, j in all_term_pairs:
        term1_info = sample_df.loc[i]
        term2_info = sample_df.loc[j]
        similarity = np.dot(term1_info['embedding'], term2_info['embedding'])
        if term1_info['category'] == term2_info['category']:
            intra_category_scores.append(similarity)
        else:
            inter_category_scores.append(similarity)
            
    print("✅ Test complete! The scores are magnificent!")

    # --- And now for the final masterpiece! The plot! ---
    print("🎨 Now to visualize the results! The best part!")
    plt.figure(figsize=(12, 7))
    sns.kdeplot(intra_category_scores, fill=True, label='Intra-Category (Same Type)', lw=3)
    sns.kdeplot(inter_category_scores, fill=True, label='Inter-Category (Different Types)', lw=3)
    plt.title(f'Category Purity Test for\n{CHAMPION_MODEL}', fontsize=16)
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Density')
    plt.legend()
    
    plot_path = os.path.join(output_dir, f"purity_test_{safe_model_name}.png")
    plt.savefig(plot_path, bbox_inches='tight')
    
    print(f"🎉 AHAHAHA! IT'S DONE! I saved your beautiful purity plot to: {plot_path}")

if __name__ == '__main__':
    test_category_purity_of_champion()