import os
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import argparse
from pathlib import Path

# --- Argument Parser ---
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectorizer_method", required=True, help="Name of the vectorizer model used.")
    return parser.parse_args()


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
    args = parse_args()

    ROOT_DIR = Path(__file__).resolve().parents[2]
    DATA_DIR = ROOT_DIR / "data"

    embedding_file = DATA_DIR / f"term_embedding_library_by_category_{args.vectorizer_method}.pkl"
    with open(embedding_file, "rb") as f:
        term_library = pickle.load(f)

    print(f"📥 Loaded embeddings from: {embedding_file}")

    os.makedirs(DATA_DIR, exist_ok=True)
    categories = list(term_library.keys())

    for cat in categories:
        print(f"🔍 Analyzing category: {cat}")
        terms = term_library[cat]
        term_vecs = np.vstack([entry["embedding"] for entry in terms])

        within = compute_within_category_cosines(term_vecs)

        other_cats = [c for c in categories if c != cat]
        random_other_cat = random.choice(other_cats)
        other_vecs = np.vstack([entry["embedding"] for entry in term_library[random_other_cat]])
        cross = compute_cross_category_cosines(term_vecs, other_vecs)

        plt.figure(figsize=(10, 6))
        plt.hist(within, bins=50, alpha=0.7, label=f"Within {cat}", color="blue")
        plt.hist(cross, bins=50, alpha=0.7, label=f"Vs {random_other_cat}", color="red")
        plt.title(f"Cosine Similarity Distribution: {cat} (Vectorizer: {args.vectorizer_method})")
        plt.xlabel("Cosine Similarity")
        plt.ylabel("Frequency")
        plt.legend()
        output_path = DATA_DIR / f"cosine_similarity_{cat}_{args.vectorizer_method}.png"
        plt.savefig(output_path)
        plt.close()

        print(f"📊 Saved plot to: {output_path}")


if __name__ == "__main__":
    main()
