import os
import pickle
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

# === Dynamic Project Paths ===
current_script_dir = Path(__file__).resolve().parent
project_root = current_script_dir.parents[2]
data_dir = project_root / "data"
os.makedirs(data_dir, exist_ok=True)

# === Config Import ===
from scripts.config import setup_config, get_global_config

# === Argument Parsing ===
parser = argparse.ArgumentParser()
parser.add_argument("--representation_method", required=True)
parser.add_argument("--vectorizer_method", required=True)
parser.add_argument("--distance_metric", required=True)
parser.add_argument("--num_visits", type=int, required=True)
parser.add_argument("--num_patients", type=int, required=True)
parser.add_argument("--num_neighbors", type=int, required=True)
parser.add_argument("--model_name", type=str, required=True)
args = parser.parse_args()

# === Setup Global Config ===
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

# === Embedding Path from Config ===
embedding_path = data_dir / (
    f"term_embedding_library_by_category_"
    f"{config.num_patients}_{config.num_visits}_"
    f"{config.representation_method}_{config.vectorizer_method}_{config.distance_metric}.pkl"
)

if not embedding_path.exists():
    raise FileNotFoundError(f"Embedding file not found: {embedding_path}")

# === Load Embeddings ===
with open(embedding_path, "rb") as f:
    term_library = pickle.load(f)

# === Cosine Computation Functions ===
def compute_within_category_cosines(embeddings: np.ndarray):
    sim_matrix = cosine_similarity(embeddings)
    n = len(sim_matrix)
    return [sim_matrix[i][j] for i in range(n) for j in range(i + 1, n)]

def compute_cross_category_cosines(cat1: np.ndarray, cat2: np.ndarray, num_samples=5000):
    indices1 = np.random.choice(len(cat1), size=num_samples, replace=True)
    indices2 = np.random.choice(len(cat2), size=num_samples, replace=True)
    sims = cosine_similarity(cat1[indices1], cat2[indices2])
    return sims.flatten().tolist()

# === Main Plot Loop ===
categories = list(term_library.keys())

for cat in categories:
    print(f"Analyzing category: {cat}")

    terms = term_library[cat]
    term_vecs = np.vstack([item['embedding'] for item in terms])

    within = compute_within_category_cosines(term_vecs)

    other_cats = [c for c in categories if c != cat]
    random_other_cat = random.choice(other_cats)
    other_vecs = np.vstack([item['embedding'] for item in term_library[random_other_cat]])
    cross = compute_cross_category_cosines(term_vecs, other_vecs)

    # === Plotting ===
    plt.figure(figsize=(10, 6))
    plt.hist(within, bins=50, alpha=0.7, label=f"Within {cat}", color="blue")
    plt.hist(cross, bins=50, alpha=0.7, label=f"Random vs {random_other_cat}", color="red")
    plt.title(f"Cosine Similarity: {cat}")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.legend()

    output_name = (
        f"cosine_similarity_{cat}_"
        f"{config.num_patients}_{config.num_visits}_"
        f"{config.representation_method}_{config.vectorizer_method}_{config.distance_metric}.png"
    )
    output_path = data_dir / output_name
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Saved plot to: {output_path}")
