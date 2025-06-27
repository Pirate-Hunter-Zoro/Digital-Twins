import pickle
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path

def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-8)
    return np.dot(normalized, normalized.T)

def main():
    ROOT_DIR = Path(__file__).resolve().parents[2]
    INPUT_PATH = ROOT_DIR / "data" / "term_embedding_library_by_category.pkl"
    OUTPUT_JSON = ROOT_DIR / "data" / "cosine_similarity_stats_by_category.json"
    PLOT_DIR = ROOT_DIR / "data" / "cosine_similarity_histograms"
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    with open(INPUT_PATH, "rb") as f:
        grouped_embeddings = pickle.load(f)

    summary = {}
    for category, embedding_dict in grouped_embeddings.items():
        terms = list(embedding_dict.keys())
        embeddings = np.stack([embedding_dict[term] for term in terms])

        if embeddings.shape[0] < 2:
            continue

        sim_matrix = cosine_similarity_matrix(embeddings)
        mask = ~np.eye(sim_matrix.shape[0], dtype=bool)
        similarities = sim_matrix[mask]

        stats = {
            "mean": float(np.mean(similarities)),
            "median": float(np.median(similarities)),
            "std": float(np.std(similarities)),
            "min": float(np.min(similarities)),
            "max": float(np.max(similarities)),
            "count": int(similarities.size)
        }

        summary[category] = stats

        # Save histogram
        plt.figure(figsize=(10, 5))
        plt.hist(similarities, bins=100, color='skyblue', edgecolor='gray')
        plt.title(f"Distribution of Cosine Similarities — {category}")
        plt.xlabel("Cosine Similarity")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(PLOT_DIR / f"cosine_similarities_{category}.png")
        plt.close()

    with open(OUTPUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"✅ Saved stats to {OUTPUT_JSON}")
    print(f"📊 Plots saved in {PLOT_DIR}")

if __name__ == "__main__":
    main()
