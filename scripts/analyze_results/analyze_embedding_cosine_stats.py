import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import pickle

from pathlib import Path

# --- Dynamic sys.path adjustment ---
current_script_dir = Path(__file__).resolve().parent
scripts_dir = current_script_dir.parent
project_root = scripts_dir.parent
data_dir = project_root / "data"

if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))
# --- End dynamic pathing ---

from config import setup_config, get_global_config

def compute_within_category_cosines(embeddings: np.ndarray):
    sim_matrix = cosine_similarity(embeddings)
    n = len(sim_matrix)
    return [sim_matrix[i][j] for i in range(n) for j in range(i + 1, n)]

def compute_cross_category_cosines(cat1: np.ndarray, cat2: np.ndarray, num_samples=5000):
    indices1 = np.random.choice(len(cat1), size=num_samples, replace=True)
    indices2 = np.random.choice(len(cat2), size=num_samples, replace=True)
    sims = cosine_similarity(cat1[indices1], cat2[indices2])
    return sims.flatten().tolist()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--representation_method", required=True)
    parser.add_argument("--vectorizer_method", required=True)
    parser.add_argument("--distance_metric", default="euclidean")
    parser.add_argument("--num_visits", type=int, default=6)
    parser.add_argument("--num_patients", type=int, default=5000)
    parser.add_argument("--num_neighbors", type=int, default=5)
    parser.add_argument("--model_name", type=str, default="medgemma")
    args = parser.parse_args()

    setup_config(
        representation_method=args.representation_method,
        vectorizer_method=args.vectorizer_method,
        distance_metric=args.distance_metric,
        num_visits=args.num_visits,
        num_patients=args.num_patients,
        num_neighbors=args.num_neighbors,
        model_name=args.model_name,
    )

    config = get_global_config()

    # === Updated embedding path with all config params ===
    embedding_filename = (
        f"term_embedding_library_by_category_"
        f"{config.num_patients}_{config.num_visits}_{config.representation_method}_"
        f"{config.vectorizer_method}_{config.distance_metric}_{config.model_name}.pkl"
    )
    embedding_path = data_dir / embedding_filename
    if not embedding_path.exists():
        raise FileNotFoundError(f"Embedding file not found: {embedding_path}")

    with open(embedding_path, "rb") as f:
        term_library = pickle.load(f)

    for cat in term_library:
        print(f"🔍 Analyzing category: {cat}")
        terms = term_library[cat]
        term_vecs = np.vstack([item["embedding"] for item in terms])

        within = compute_within_category_cosines(term_vecs)

        other_cats = [c for c in term_library if c != cat]
        random_other_cat = np.random.choice(other_cats)
        other_vecs = np.vstack([item["embedding"] for item in term_library[random_other_cat]])
        cross = compute_cross_category_cosines(term_vecs, other_vecs)

        plt.figure(figsize=(10, 6))
        plt.hist(within, bins=50, alpha=0.7, label=f"Within {cat}", color="blue")
        plt.hist(cross, bins=50, alpha=0.7, label=f"Random vs {random_other_cat}", color="red")
        plt.title(f"Cosine Similarity Distribution: {cat}")
        plt.xlabel("Cosine Similarity")
        plt.ylabel("Frequency")
        plt.legend()

        output_path = data_dir / f"cosine_similarity_{cat}_{config.num_patients}_{config.num_visits}_{config.representation_method}_{config.vectorizer_method}_{config.distance_metric}_{config.model_name}.png"
        plt.savefig(output_path)
        plt.close()
        print(f"📈 Saved plot to: {output_path}")

if __name__ == "__main__":
    main()
