import os
import sys
import json
import pickle
import numpy as np
from scipy.stats import spearmanr
from scipy.spatial.distance import mahalanobis
from numpy.linalg import inv
from collections import defaultdict

# --- Dynamic sys.path adjustment ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.config import setup_config, get_global_config
from scripts.llm.llm_helper import get_relevance_score, get_narrative
from scripts.read_data.load_patient_data import load_patient_data

# === Memoization Paths ===
def get_cache_paths(config, model_name):
    stem = f"{config.num_patients}_{config.num_visits}_{config.representation_method}_{config.vectorizer_method}_{config.distance_metric}_{model_name}"
    return {
        "relevance_scores": os.path.join("data", f"relevance_cache_{stem}.jsonl"),
        "narratives": os.path.join("data", f"narratives_cache_{stem}.json"),
        "spearman_output": os.path.join("data", f"spearman_rho_{stem}.json"),
        "progress": os.path.join("data", f"processed_patients_{stem}.txt"),
    }

def load_cached_narratives(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}

def save_cached_narratives(path, cache):
    with open(path, "w") as f:
        json.dump(cache, f)

def append_relevance_log(path, entry):
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")

def load_processed_patient_ids(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return set(f.read().splitlines())
    return set()

def append_processed_patient_id(path, patient_key):
    with open(path, "a") as f:
        f.write(f"{patient_key}\n")

def compute_mahalanobis_distance_to_group(patient_vec, neighbor_vecs):
    if len(neighbor_vecs) < 2:
        return float("nan")

    mean_vec = np.mean(neighbor_vecs, axis=0)
    cov_matrix = np.cov(neighbor_vecs, rowvar=False)

    try:
        inv_cov = inv(cov_matrix)
        return mahalanobis(patient_vec, mean_vec, inv_cov)
    except np.linalg.LinAlgError:
        return float("nan")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--representation_method", required=True)
    parser.add_argument("--vectorizer_method", required=True)
    parser.add_argument("--distance_metric", default="euclidean")
    parser.add_argument("--num_visits", type=int, default=6)
    parser.add_argument("--num_patients", type=int, default=5000)
    parser.add_argument("--num_neighbors", type=int, default=5)
    parser.add_argument("--model_name", type=str, default="medgemma")
    parser.add_argument("--max_patients", type=int, default=500)
    args = parser.parse_args()

    setup_config(
        representation_method=args.representation_method,
        vectorizer_method=args.vectorizer_method,
        distance_metric=args.distance_metric,
        num_visits=args.num_visits,
        num_patients=args.num_patients,
        num_neighbors=args.num_neighbors,
    )

    config = get_global_config()
    paths = get_cache_paths(config, args.model_name)

    os.makedirs("data", exist_ok=True)
    narrative_cache = load_cached_narratives(paths["narratives"])
    processed_ids = load_processed_patient_ids(paths["progress"])

    patient_data = load_patient_data()
    patient_lookup = {p["patient_id"]: p for p in patient_data}

    with open(os.path.join("data", f"neighbors_{config.num_patients}_{config.num_visits}_{config.representation_method}_{config.vectorizer_method}_{config.distance_metric}.pkl"), "rb") as f:
        neighbors_dict = pickle.load(f)

    with open(os.path.join("data", f"all_vectors_{config.num_patients}_{config.num_visits}_{config.representation_method}_{config.vectorizer_method}.pkl"), "rb") as f:
        vector_dict = pickle.load(f)

    results = []
    processed = 0

    print(f"🧪 Resuming analysis up to {args.max_patients} patients...")

    for (patient_id, visit_idx), patient_vector in vector_dict.items():
        patient_key = f"{patient_id}:{visit_idx}"
        if processed >= args.max_patients:
            break
        if patient_key in processed_ids:
            continue

        neighbors = neighbors_dict.get((patient_id, visit_idx))
        if not neighbors:
            continue

        patient = patient_lookup.get(patient_id)
        if not patient:
            continue

        if patient_key not in narrative_cache:
            try:
                narrative_cache[patient_key] = get_narrative(patient["visits"][:visit_idx + 1])
            except Exception:
                print(f"Skipping {patient_id} — narrative generation failed")
                continue

        patient_narrative = narrative_cache[patient_key]
        relevance_scores = []
        neighbor_vectors = []

        for (neighbor_id, neighbor_vidx), _, neighbor_vec in neighbors[:config.num_neighbors]:
            neighbor_key = f"{neighbor_id}:{neighbor_vidx}"
            if neighbor_key not in narrative_cache:
                neighbor = patient_lookup.get(neighbor_id)
                if not neighbor:
                    continue
                try:
                    narrative_cache[neighbor_key] = get_narrative(neighbor["visits"][:neighbor_vidx + 1])
                except Exception:
                    continue

            neighbor_narrative = narrative_cache[neighbor_key]
            relevance = get_relevance_score(patient_narrative, neighbor_narrative)
            relevance_scores.append(relevance)
            neighbor_vectors.append(neighbor_vec)

            append_relevance_log(paths["relevance_scores"], {
                "patient": patient_key,
                "neighbor": neighbor_key,
                "score": relevance,
            })

        if len(relevance_scores) < 2:
            continue

        avg_relevance = np.mean(relevance_scores)
        mahal_dist = compute_mahalanobis_distance_to_group(patient_vector, np.vstack(neighbor_vectors))

        if not np.isnan(mahal_dist):
            results.append((avg_relevance, mahal_dist))
            processed += 1
            append_processed_patient_id(paths["progress"], patient_key)

    save_cached_narratives(paths["narratives"], narrative_cache)

    if not results:
        print("❌ No valid results, exiting.")
        return

    relevance_vals, mahalanobis_vals = zip(*results)
    rho, pval = spearmanr(relevance_vals, mahalanobis_vals)

    summary = {
        "spearman_rho": rho,
        "p_value": pval,
        "num_patients": len(relevance_vals)
    }

    with open(paths["spearman_output"], "w") as f:
        json.dump(summary, f, indent=2)

    print("✅ Spearman correlation complete!")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
