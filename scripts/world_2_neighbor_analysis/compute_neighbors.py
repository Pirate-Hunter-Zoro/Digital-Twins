import sys
import os
import argparse
import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# TODO - refactor - no more sorting necessary

# --- Dynamic sys.path adjustment ---
current_script_dir = Path(__file__).resolve().parent
project_root = current_script_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.common.config import setup_config, get_global_config

def main():
    print("🤖 Activating the Neighbor-Finding Machine (Behemoth Edition)! 🤖")
    parser = argparse.ArgumentParser(description="Compute and save ALL ranked neighbors from pre-computed vectors.")
    # --- ✨ A NEW, SUPER IMPORTANT ARGUMENT! ✨ ---
    parser.add_argument("--embedder_type", required=True, choices=["gru", "transformer"], help="The type of pre-computed vectors to use.")
    parser.add_argument("--num_visits", type=int, default=6)
    args = parser.parse_args()

    # We use a placeholder for vectorizer_method here because the real info is in the embedder_type!
    setup_config("visit_sentence", args.embedder_type, "cosine", args.num_visits, 5000, 5)
    config = get_global_config()

    # --- Path Setup ---
    # The paths now depend on which embedder's vectors we are using!
    base_dir = project_root / "data" / "visit_sentence"
    vectors_dir = base_dir / f"visits_{config.num_visits}" / "patients_5000"
    output_dir = vectors_dir # The neighbors file will live alongside the vectors file
    
    os.makedirs(output_dir, exist_ok=True)

    # It now knows exactly which vector file and which output file to use!
    vectors_path = vectors_dir / f"all_vectors_{args.embedder_type}.pkl"
    neighbors_path = output_dir / f"all_ranked_neighbors_{args.embedder_type}.pkl"

    if not vectors_path.exists():
        raise FileNotFoundError(f"❌ OH NO! The vector file for '{args.embedder_type}' was not found at {vectors_path}. Please run the vectorization script first!")

    if os.path.exists(neighbors_path):
        print(f"✅ All ranked neighbors for '{args.embedder_type}' already exist! Nothing to do!")
        return

    print(f"📂 Loading pre-computed vectors from {vectors_path}...")
    with open(vectors_path, "rb") as f:
        vectors_dict = pickle.load(f)

    print("🔍 Computing and ranking ALL neighbors for every patient...")
    keys = list(vectors_dict.keys())
    matrix = np.vstack([vectors_dict[k] for k in keys])
    sim_matrix = cosine_similarity(matrix)

    all_ranked_neighbors = {}
    for i, key in enumerate(keys):
        sims = sim_matrix[i]
        potential_neighbors = [(keys[j], sims[j]) for j in range(len(keys)) if i != j]
        ranked_list = sorted(potential_neighbors, key=lambda x: x[1], reverse=True)
        all_ranked_neighbors[key] = ranked_list

    with open(neighbors_path, "wb") as f:
        pickle.dump(all_ranked_neighbors, f)
    print(f"✅ Saved the Flexible Behemoth (all ranked neighbors) to {neighbors_path}")

if __name__ == "__main__":
    main()