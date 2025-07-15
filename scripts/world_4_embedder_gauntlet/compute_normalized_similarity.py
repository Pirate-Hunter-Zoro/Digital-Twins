import os
import sys
import json
import pickle
import argparse
import numpy as np
import random
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations

# --- Dynamic sys.path adjustment ---
current_script_dir = Path(__file__).resolve().parent
project_root = current_script_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def calculate_average_similarity(model, vocabulary, sample_size=300):
    """
    Calculates the average cosine similarity for a model based on a random sample of its vocabulary.
    """
    if len(vocabulary) < sample_size:
        print(f"  - Vocabulary smaller than sample size, using all {len(vocabulary)} terms.")
        sample_size = len(vocabulary)

    print(f"  - Taking a random sample of {sample_size} terms to find baseline similarity...")
    sample_terms = random.sample(vocabulary, sample_size)

    embeddings = model.encode(sample_terms, show_progress_bar=False, batch_size=128)

    # Calculate all-vs-all similarity matrix
    sim_matrix = cosine_similarity(embeddings)

    # Get the upper triangle of the matrix (without the diagonal) to avoid duplicates and self-similarity
    upper_triangle_indices = np.triu_indices_from(sim_matrix, k=1)
    all_sims = sim_matrix[upper_triangle_indices]

    avg_sim = np.mean(all_sims)
    print(f"  - Calculated baseline average similarity for this model: {avg_sim:.4f}")
    return avg_sim

def main():
    parser = argparse.ArgumentParser(description="Compute normalized similarity scores for term pairs.")
    parser.add_argument("--model", type=str, required=True, help="Name of the SentenceTransformer model to use.")
    args = parser.parse_args()

    # --- Path setup ---
    data_dir = project_root / "data"
    pairs_path = data_dir / "term_pairs_by_category.json"

    model_name_safe = args.model.replace("/", "-")
    # A new home for our new metric!
    output_dir = data_dir / "normalized_embeddings_by_category" / model_name_safe
    os.makedirs(output_dir, exist_ok=True)

    # --- Load Model ---
    model_path = f"/media/scratch/mferguson/models/{model_name_safe}"
    print(f"🗺️ Loading model from: {model_path}")
    model = SentenceTransformer(model_path, local_files_only=True)

    # --- Load Term Pairs and create a master vocabulary ---
    print(f"📂 Loading term pairs from: {pairs_path}")
    with open(pairs_path, 'r') as f:
        term_pairs_by_category = json.load(f)

    master_vocab = set()
    for category, pairs in term_pairs_by_category.items():
        for t1, t2 in pairs:
            master_vocab.add(t1)
            master_vocab.add(t2)

    # --- Calculate the magnificent avg_cos_sim for this model ---
    avg_cos_sim = calculate_average_similarity(model, list(master_vocab))

    # --- Process each category ---
    for category, pairs in term_pairs_by_category.items():
        print(f"\n--- Processing category: {category} ---")
        if not pairs:
            continue

        terms1, terms2 = zip(*pairs)
        unique_terms = sorted(list(set(terms1 + terms2)))

        embeddings = model.encode(unique_terms, show_progress_bar=True, batch_size=128)
        embedding_map = {term: emb for term, emb in zip(unique_terms, embeddings)}

        results_data = []
        for t1, t2 in pairs:
            e1 = embedding_map.get(t1)
            e2 = embedding_map.get(t2)
            if e1 is None or e2 is None:
                continue
            cos_sim = cosine_similarity(e1.reshape(1, -1), e2.reshape(1, -1))[0][0]
            # The new metric calculation!
            norm_cos_sim = (cos_sim - avg_cos_sim) / avg_cos_sim if avg_cos_sim != 0 else 0
            results_data.append({
                "term_1": t1,
                "term_2": t2,
                "cosine_similarity": cos_sim,
                "normalized_similarity": norm_cos_sim
            })

        # --- Save results for the category ---
        category_output_path = output_dir / f"{category}_normalized_scores.json"
        with open(category_output_path, 'w') as f:
            json.dump(results_data, f, indent=2)

        print(f"✅ Saved normalized similarity scores for {category} to {category_output_path}")

    print("\n🎉 Glorious success! All categories have been processed with the new metric!")

if __name__ == "__main__":
    main()