import sys
import os
import argparse
import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from itertools import combinations
import json

# --- Dynamic sys.path adjustment ---
current_script_dir = Path(__file__).resolve().parent
project_root = current_script_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.common.config import setup_config, get_global_config
from scripts.common.data_loading.load_patient_data import load_patient_data
from scripts.common.llm.llm_helper import get_narrative, get_relevance_score


def main():
    print("🤖✨ Activating the Universal Metric-Calculating Behemoth! ✨🤖")
    parser = argparse.ArgumentParser(description="Compute pairwise distance and similarity metrics for all patient vectors.")
    parser.add_argument("--embedder_type", required=True, choices=["gru", "transformer"], help="The type of pre-computed vectors to use.")
    parser.add_argument("--num_visits", type=int, default=6)
    args = parser.parse_args()

    # We use a placeholder for other config values
    setup_config("visit_sentence", args.embedder_type, "multiple", args.num_visits, 5000, 0)
    config = get_global_config()

    # --- Path Setup ---
    base_dir = project_root / "data" / "visit_sentence"
    data_dir = base_dir / f"visits_{config.num_visits}" / "patients_5000"
    output_dir = data_dir
    os.makedirs(output_dir, exist_ok=True)

    vectors_path = data_dir / f"all_vectors_{args.embedder_type}.pkl"
    # The new output file for our magnificent metrics!
    metrics_output_path = output_dir / f"all_pairwise_metrics_{args.embedder_type}.json"

    if not vectors_path.exists():
        raise FileNotFoundError(f"❌ OH NO! The vector file for '{args.embedder_type}' was not found at {vectors_path}.")

    if metrics_output_path.exists():
        print(f"✅ All pairwise metrics for '{args.embedder_type}' already exist! Nothing to do!")
        return

    print(f"📂 Loading pre-computed vectors from {vectors_path}...")
    with open(vectors_path, "rb") as f:
        vectors_dict = pickle.load(f)

    print("📂 Loading patient data for narrative generation...")
    patient_lookup = {p["patient_id"]: p for p in load_patient_data()}

    keys = list(vectors_dict.keys())
    all_results = []

    print(f"⚙️  Starting computation for {len(keys)} patient visit sequences... This will create a LOT of pairs!")

    # Using itertools.combinations to get all unique pairs! So efficient!
    for (key1, key2) in combinations(keys, 2):
        vector1 = vectors_dict[key1]
        vector2 = vectors_dict[key2]

        # --- Metric 1: Cosine Similarity ---
        # Reshape for the function, then extract the single value!
        cos_sim = cosine_similarity(vector1.reshape(1, -1), vector2.reshape(1, -1))[0][0]

        # --- Metric 2: Euclidean Distance ---
        euclidean_dist = euclidean_distances(vector1.reshape(1, -1), vector2.reshape(1, -1))[0][0]

        # --- Metric 3: LLM Semantic Similarity ---
        patient_id1, visit_idx1 = key1
        patient_id2, visit_idx2 = key2

        # Generate the little stories for our LLM friend!
        narrative1 = get_narrative(patient_lookup[patient_id1]['visits'][:visit_idx1 + 1])
        narrative2 = get_narrative(patient_lookup[patient_id2]['visits'][:visit_idx2 + 1])

        llm_score = get_relevance_score(narrative1, narrative2)

        all_results.append({
            "patient_1": key1,
            "patient_2": key2,
            "cosine_similarity": float(cos_sim),
            "euclidean_distance": float(euclidean_dist),
            "llm_relevance_score": float(llm_score)
        })

        if len(all_results) % 100 == 0:
            print(f"  ...processed {len(all_results)} pairs so far...")

    print(f"\n💾 Saving all {len(all_results)} metric sets to: {metrics_output_path}")
    with open(metrics_output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print("✅ The Behemoth is finished! All metrics have been calculated!")


if __name__ == "__main__":
    main()