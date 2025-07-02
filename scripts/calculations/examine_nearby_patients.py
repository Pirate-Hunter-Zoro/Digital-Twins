import os
import sys
import json
import pickle
import numpy as np
from scipy.stats import spearmanr
from scipy.spatial.distance import mahalanobis
from numpy.linalg import inv

# --- Dynamic sys.path adjustment ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.config import setup_config, get_global_config
from scripts.llm.llm_helper import get_relevance_score, get_narrative

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

    data_dir = os.path.join(project_root, "data")
    neighbors_path = os.path.join(data_dir, f"neighbors_{config.num_patients}_{config.num_visits}_{config.representation_method}_{config.vectorizer_method}_{config.distance_metric}.pkl")
    all_vectors_path = os.path.join(data_dir, f"all_vectors_{config.num_patients}_{config.num_visits}_{config.representation_method}_{config.vectorizer_method}.pkl")
    patient_data_path = os.path.join(data_dir, "all_patients_combined.json")

    with open(neighbors_path, "rb") as f:
        neighbors_dict = pickle.load(f)

    with open(all_vectors_path, "rb") as f:
        vector_dict = pickle.load(f)

    from scripts.read_data.load_patient_data import load_patient_data
    patient_data = load_patient_data()
    patient_lookup = {p["patient_id"]: p for p in patient_data}

    print("--- Beginning correlation computation ---")
    results = []

    for (patient_id, visit_idx), patient_vector in vector_dict.items():
        if (patient_id, visit_idx) not in neighbors_dict:
            print(f"Skipping {patient_id} @ visit {visit_idx} — no neighbors found.")
            continue

        patient = patient_lookup.get(patient_id)
        if not patient:
            print(f"Skipping {patient_id} — patient not found.")
            continue

        try:
            patient_narrative = get_narrative(patient["visits"][:visit_idx + 1])
        except Exception as e:
            print(f"Skipping {patient_id} — narrative generation failed: {e}")
            continue

        neighbors = neighbors_dict[(patient_id, visit_idx)][:config.num_neighbors]

        relevance_scores = []
        neighbor_vectors = []

        for (neighbor_id, neighbor_vidx), _, neighbor_vec in neighbors:
            neighbor = patient_lookup.get(neighbor_id)
            if not neighbor:
                continue
            try:
                neighbor_narrative = get_narrative(neighbor["visits"][:neighbor_vidx + 1])
                relevance = get_relevance_score(patient_narrative, neighbor_narrative)
                relevance_scores.append(relevance)
                neighbor_vectors.append(neighbor_vec)
            except Exception as e:
                print(f"Skipping neighbor {neighbor_id} — relevance failed: {e}")
                continue

        if len(relevance_scores) < 2:
            print(f"Skipping {patient_id} — not enough valid neighbors (found {len(relevance_scores)}).")
            continue

        avg_relevance = np.mean(relevance_scores)
        neighbor_vec_matrix = np.vstack(neighbor_vectors)
        mahal_dist = compute_mahalanobis_distance_to_group(patient_vector, neighbor_vec_matrix)

        if np.isnan(mahal_dist):
            print(f"Skipping {patient_id} — Mahalanobis distance invalid.")
            continue

        results.append((avg_relevance, mahal_dist))

    if not results:
        print("⚠️ No valid patient results found. Exiting early.")
        return

    relevance_vals, mahalanobis_vals = zip(*results)
    rho, pval = spearmanr(relevance_vals, mahalanobis_vals)

    summary = {
        "spearman_rho": rho,
        "p_value": pval,
        "num_patients": len(relevance_vals)
    }

    output_path = os.path.join(data_dir, f"spearman_rho_{config.num_patients}_{config.num_visits}_{config.representation_method}_{config.vectorizer_method}_{config.distance_metric}.json")
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("✅ Done!")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
