import os
import sys
import json
import argparse
import numpy as np
import random
from pathlib import Path
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations

# --- Dynamic sys.path adjustment ---
current_script_dir = Path(__file__).resolve().parent
project_root = current_script_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def load_legacy_model(model_path, is_word2vec):
    """Loads a GloVe or Word2Vec model."""
    print(f"-> Loading legacy model from {model_path}...")
    binary = is_word2vec
    model = KeyedVectors.load_word2vec_format(model_path, binary=binary)
    print("-> Model loaded!")
    return model

def calculate_average_similarity_legacy(model, sample_size=300):
    """Calculates baseline similarity for legacy models."""
    vocab = list(model.index_to_key)
    if len(vocab) < sample_size:
        sample_size = len(vocab)
        
    print(f"  - Taking a random sample of {sample_size} terms for baseline similarity...")
    sample_terms = random.sample(vocab, sample_size)
    
    embeddings = np.array([model[term] for term in sample_terms])
    sim_matrix = cosine_similarity(embeddings)
    upper_triangle_indices = np.triu_indices_from(sim_matrix, k=1)
    avg_sim = np.mean(sim_matrix[upper_triangle_indices])
    
    print(f"  - Calculated baseline average similarity: {avg_sim:.4f}")
    return avg_sim

def main():
    parser = argparse.ArgumentParser(description="Compute NORMALIZED similarity scores for LEGACY models.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--is_word2vec", action="store_true", help="Flag if the model is in Word2Vec binary format.")
    args = parser.parse_args()

    # --- Path setup ---
    data_dir = project_root / "data"
    pairs_path = data_dir / "term_pairs_by_category.json"
    output_dir = data_dir / "normalized_embeddings_by_category" / args.model_name
    os.makedirs(output_dir, exist_ok=True)

    # --- Load Model and Term Pairs ---
    model = load_legacy_model(args.model_path, args.is_word2vec)
    with open(pairs_path, 'r') as f:
        term_pairs_by_category = json.load(f)

    # --- Calculate Baseline Similarity ---
    avg_cos_sim = calculate_average_similarity_legacy(model)
    
    # --- Process Each Category ---
    for category, pairs in term_pairs_by_category.items():
        print(f"\n--- Processing category: {category} ---")
        
        results_data = []
        for t1, t2 in pairs:
            if t1 in model and t2 in model:
                e1 = model[t1].reshape(1, -1)
                e2 = model[t2].reshape(1, -1)
                cos_sim = cosine_similarity(e1, e2)[0][0]
                norm_cos_sim = (cos_sim - avg_cos_sim) / avg_cos_sim if avg_cos_sim != 0 else 0
                results_data.append({
                    "term_1": t1,
                    "term_2": t2,
                    "cosine_similarity": float(cos_sim),
                    "normalized_similarity": float(norm_cos_sim)
                })

        # Save results
        category_output_path = output_dir / f"{category}_normalized_scores.json"
        with open(category_output_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        print(f"✅ Saved normalized scores for {category} to {category_output_path}")

    print("\n🎉 Glorious success! All legacy models processed with the new metric!")

if __name__ == "__main__":
    main()