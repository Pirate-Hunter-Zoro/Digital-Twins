import os
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# === Load embeddings ===
embedding_path = "data/term_embedding_library_by_category.pkl"
with open(embedding_path, "rb") as f:
    term_library = pickle.load(f)

os.makedirs("data", exist_ok=True)

def compute_within_category_cosines(embeddings: np.ndarray):
    sim_matrix = cosine_similarity(embeddings)
    n = len(sim_matrix)
    return [sim_matrix[i][j] for i in range(n) for j in range(i + 1, n)]

def compute_cross_category_cosines(cat1: np.ndarray, cat2: np.ndarray, num_samples=5000):
    indices1 = np.random.choice(len(cat1), size=num_samples, replace=True)
    indices2 = np.random.choice(len(cat2), size=num_samples, replace=True)
    sims = cosine_similarity(cat1[indices1], cat2[indices2])
    return sims.flatten().tolist()

# === Iterate through categories ===
categories = list(term_library.keys())

for cat in categories:
    print(f"Analyzing category: {cat}")

    term_dict = term_library[cat]
    term_vecs = np.vstack(list(term_dict.values()))

    # Compute within-category cosine similarities
    within = compute_within_category_cosines(term_vecs)

    # Compute random inter-category similarities
    other_cats = [c for c in categories if c != cat]
    random_other_cat = random.choice(other_cats)
    other_vecs = np.vstack(list(term_library[random_other_cat].values()))
    cross = compute_cross_category_cosines(term_vecs, other_vecs)

    # === Plotting ===
    plt.figure(figsize=(10, 6))
    plt.hist(within, bins=50, alpha=0.7, label=f"Within {cat}", color="blue")
    plt.hist(cross, bins=50, alpha=0.7, label=f"Random vs {random_other_cat}", color="red")
    plt.title(f"Cosine Similarity Distribution: {cat}")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.legend()
    output_path = f"data/cosine_similarity_{cat}.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot to: {output_path}")
