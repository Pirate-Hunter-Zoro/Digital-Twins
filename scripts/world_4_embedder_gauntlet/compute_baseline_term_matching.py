import os
import sys
import pandas as pd
import numpy as np
import argparse

# --- The Magnificent Fix! ---
# This makes the script smart enough to see the whole 'scripts' directory!
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ---------------------------

def calculate_baseline_score(input_csv_path: str):
    """
    This magnificent machine takes a CSV of cosine similarities,
    calculates a baseline from random pairs, and then computes
    our new, super-smart significance score! AHAHAHA!
    """
    print(f"🚀 Let's calculate a new kind of score for: {os.path.basename(input_csv_path)}!")

    if not os.path.exists(input_csv_path):
        print(f"❌ OH NO! I can't find the input file at: {input_csv_path}")
        return

    df = pd.read_csv(input_csv_path)
    print(f"✅ Loaded {len(df)} term pairs!")

    # --- Step 1: Calculate the Random Baseline ---
    print("🤔 First, let's see what random chance looks like... shuffling the counterparts!")
    
    # Create a shuffled version of the counterparts to make random pairs
    shuffled_counterparts = df['counterpart'].sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Let's not compare a term to itself! That's cheating!
    # We'll keep shuffling until no term is paired with its original counterpart
    while any(df['term'] == shuffled_counterparts):
        shuffled_counterparts = df['counterpart'].sample(frac=1).reset_index(drop=True)
    
    # Now, let's get the embeddings for our new, chaotic pairs!
    # We'll need the model to do this!
    model_name = df['model'].iloc[0]
    sanitized_model_name = model_name.replace("/", "-")
    local_model_path = f"/media/scratch/mferguson/models/{sanitized_model_name}"
    
    print(f"📦 Loading the magnificent model: {model_name}")
    from sentence_transformers import SentenceTransformer, util
    model = SentenceTransformer(local_model_path)

    print("🧠 Calculating similarities for our random pairs... this might take a moment!")
    random_similarities = []
    terms = df['term'].tolist()
    shuffled_counterparts_list = shuffled_counterparts.tolist()
    
    # Let's be efficient! Encode everything at once!
    term_embeddings = model.encode(terms, convert_to_tensor=True)
    shuffled_embeddings = model.encode(shuffled_counterparts_list, convert_to_tensor=True)

    # Calculate cosine similarity for all random pairs at once! ZAP!
    random_sim_matrix = util.cos_sim(term_embeddings, shuffled_embeddings)
    random_similarities = [random_sim_matrix[i, i].item() for i in range(len(terms))]
    
    random_avg_cos_sim = np.mean(random_similarities)
    print(f"📊 The average similarity for random pairs is: {random_avg_cos_sim:.4f}")

    # --- Step 2: Calculate our new magnificent score! ---
    print("✨ Now for the main event! Calculating the significance score!")
    
    # The formula is magnificent!
    df['baseline_score'] = (df['cosine_similarity'] - random_avg_cos_sim) / random_avg_cos_sim
    
    # --- Step 3: Save the beautiful results! ---
    output_dir = os.path.join(project_root, "data", "baseline_scores")
    os.makedirs(output_dir, exist_ok=True)
    
    output_filename = os.path.basename(input_csv_path).replace('.csv', '_baseline.csv')
    output_path = os.path.join(output_dir, output_filename)
    
    df.to_csv(output_path, index=False)
    
    print(f"\n🎉 AHAHAHA! IT'S DONE! Your new scores are saved at: {output_path}")
    print(f"📈 The average baseline score is: {df['baseline_score'].mean():.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate a baseline-adjusted similarity score.")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to the input CSV file with cosine similarities.")
    args = parser.parse_args()
    
    calculate_baseline_score(args.input_csv)