import os
import sys
import argparse
import pickle
from scipy.stats import spearmanr
import numpy as np

# === Dynamic Path Fix ===
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.config import setup_config, get_global_config

def load_scores(score_file):
    with open(score_file, "rb") as f:
        return pickle.load(f)

def extract_llm_and_distance_scores(data):
    llm_scores = []
    distances = []

    for patient_key, neighbors in data.items():
        for _, _, dist, score in neighbors:
            llm_scores.append(score)
            distances.append(dist)

    return distances, llm_scores

def main(args):
    # 💾 Construct filename dynamically from args
    filename = f"data/eval_patient_neighbors_{args.num_patients}_{args.num_visits}_{args.representation_method}_{args.vectorizer_method}_{args.distance_metric}_k{args.num_neighbors}.pkl"
    
    if not os.path.exists(filename):
        print(f"❌ ERROR: File not found: {filename}")
        return

    print(f"📦 Loading evaluation data from: {filename}")
    data = load_scores(filename)
    
    distances, llm_scores = extract_llm_and_distance_scores(data)

    print(f"🔍 Computing Spearman's rho on {len(llm_scores)} neighbor pairs...")
    rho, pval = spearmanr(distances, llm_scores)

    print("📊 Spearman Correlation Results:")
    print(f"rho = {rho:.4f}")
    print(f"p-value = {pval:.4e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--representation_method", type=str, required=True)
    parser.add_argument("--vectorizer_method", type=str, required=True)
    parser.add_argument("--distance_metric", type=str, default="cosine")
    parser.add_argument("--num_patients", type=int, default=5000)
    parser.add_argument("--num_visits", type=int, default=6)
    parser.add_argument("--num_neighbors", type=int, default=5)

    args = parser.parse_args()

    # Set up global config for any downstream dependencies
    setup_config(
        representation_method=args.representation_method,
        vectorizer_method=args.vectorizer_method,
        distance_metric=args.distance_metric,
        num_visits=args.num_visits,
        num_patients=args.num_patients,
        num_neighbors=args.num_neighbors,
    )

    main(args)
