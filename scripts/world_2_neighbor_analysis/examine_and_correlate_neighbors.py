import os
import sys
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from scipy.spatial.distance import mahalanobis
from numpy.linalg import inv
from pathlib import Path

# --- Dynamic sys.path adjustment ---
current_script_dir = Path(__file__).resolve().parent
project_root = current_script_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.common.config import setup_config, get_global_config
from scripts.common.llm.llm_helper import get_relevance_score, get_narrative
from scripts.common.data_loading.load_patient_data import load_patient_data

# --- Biomni Integration! ---
try:
    from biomni.agent import A1
    agent = A1()
    BIOMNI_AVAILABLE = True
    print("🤖 Biomni agent initialized successfully!")
except ImportError:
    print("⚠️ Biomni package not found. Proceeding without Biomni-powered narrative enhancements.")
    BIOMNI_AVAILABLE = False
except Exception as e:
    print(f"🔬 Biomni agent failed to initialize: {e}. Proceeding without enhancements.")
    BIOMNI_AVAILABLE = False


def get_enhanced_narrative(visits, patient_id):
    """
    Generates a narrative for a patient's visits, enhanced with
    pathway information from Biomni's database tools if available!
    """
    basic_narrative = get_narrative(visits)
    if not BIOMNI_AVAILABLE:
        return basic_narrative
    try:
        key_diagnosis = None
        for visit in reversed(visits):
            if visit.get("diagnoses"):
                key_diagnosis = visit["diagnoses"][0].get("Diagnosis_Name")
                if key_diagnosis:
                    break
        if key_diagnosis:
            print(f"🧬 Querying Biomni for pathway info on: {key_diagnosis}")
            pathway_info = agent.go(f"Find KEGG pathway for {key_diagnosis}")
            return f"{basic_narrative} The primary diagnosis, {key_diagnosis}, is associated with the {pathway_info}."
    except Exception as e:
        print(f"Biomni call failed: {e}")
    return basic_narrative


def compute_mahalanobis_distance_to_group(patient_vec, neighbor_vecs):
    if len(neighbor_vecs) < 2:
        return float('nan')
    mean_vec = np.mean(neighbor_vecs, axis=0)
    try:
        jitter = np.random.rand(neighbor_vecs.shape[1]) * 1e-9
        cov_matrix = np.cov(neighbor_vecs + jitter, rowvar=False)
        inv_cov = inv(cov_matrix)
        return mahalanobis(patient_vec, mean_vec, inv_cov)
    except np.linalg.LinAlgError:
        print("⚠️ Warning: Covariance matrix is singular. Could not compute Mahalanobis distance.")
        return float('nan')

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze neighbor relevance using Mahalanobis distance and LLM scoring.")
    parser.add_argument("--representation_method", required=True)
    parser.add_argument("--vectorizer_method", required=True)
    parser.add_argument("--distance_metric", default="euclidean")
    parser.add_argument("--num_visits", type=int, default=6)
    parser.add_argument("--num_patients", type=int, default=5000)
    parser.add_argument("--num_neighbors", type=int, default=5)
    parser.add_argument("--model_name", type=str, default="medgemma")
    parser.add_argument("--max_patients_to_process", type=int, default=500, help="Maximum number of new patients to process in this run.")
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

    # --- ✨ Build the hyper-structured directories ---
    base_dir = project_root / "data" / config.representation_method / config.vectorizer_method
    vectors_dir = base_dir / f"visits_{config.num_visits}" / f"patients_{config.num_patients}"
    data_dir = vectors_dir / f"metric_{config.distance_metric}" / f"neighbors_{config.num_neighbors}"
    
    os.makedirs(data_dir, exist_ok=True)
    print(f"📂 Using data directory: {data_dir}")
    
    # --- ✨ Use base filenames from their correct, specific locations ---
    vectors_path = vectors_dir / "all_vectors.pkl"
    neighbors_path = data_dir / "neighbors.pkl"
    graph_output_path = data_dir / "correlation_results.png"
    summary_output_path = data_dir / "correlation_summary.json"

    print("--- 📂 Loading Data ---")
    patient_data = load_patient_data()
    patient_lookup = {p["patient_id"]: p for p in patient_data}
    
    with open(neighbors_path, "rb") as f:
        neighbors_dict = pickle.load(f)
    print(f"✅ Loaded neighbors from {neighbors_path}")

    with open(vectors_path, "rb") as f:
        vector_dict = pickle.load(f)
    print(f"✅ Loaded vectors from {vectors_path}")

    print("\n--- 🚀 Starting Analysis ---")
    results = []
    
    for (patient_id, visit_idx), patient_vector in vector_dict.items():
        if len(results) >= args.max_patients_to_process:
            print(f"🏁 Reached processing limit of {args.max_patients_to_process} patients.")
            break

        print(f"\nProcessing Patient: {patient_id}, Visit Index: {visit_idx}")
        
        neighbors = neighbors_dict.get((patient_id, visit_idx), [])
        if not neighbors:
            print("  - No neighbors found. Skipping.")
            continue
            
        patient_narrative = get_enhanced_narrative(patient_lookup[patient_id]["visits"][:visit_idx + 1], patient_id)
        
        relevance_scores = []
        neighbor_vectors = []

        for (neighbor_id, neighbor_vidx), _, neighbor_vec in neighbors[:config.num_neighbors]:
            neighbor_narrative = get_enhanced_narrative(patient_lookup[neighbor_id]["visits"][:neighbor_vidx + 1], neighbor_id)
            relevance = get_relevance_score(patient_narrative, neighbor_narrative)
            print(f"  - Neighbor {neighbor_id} Relevance: {relevance:.2f}")
            relevance_scores.append(relevance)
            neighbor_vectors.append(neighbor_vec)

        if len(relevance_scores) < 2:
            print("  - Not enough valid neighbors to proceed. Skipping.")
            continue

        avg_relevance = np.mean(relevance_scores)
        mahal_dist = compute_mahalanobis_distance_to_group(patient_vector, np.array(neighbor_vectors))

        if not np.isnan(mahal_dist):
            results.append((avg_relevance, mahal_dist))
            print(f"  - Avg. Relevance: {avg_relevance:.2f}, Mahalanobis Dist: {mahal_dist:.2f}")

    print("\n--- 📊 Final Correlation and Graph Generation ---")
    if not results:
        print("❌ No valid results were generated to calculate correlation or create a graph.")
        return

    relevance_vals, mahalanobis_vals = zip(*results)
    rho, pval = spearmanr(relevance_vals, mahalanobis_vals)
    
    # --- ✨ NEW GRAPH-O-MATIC MODULE! ✨ ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    scatter = ax.scatter(mahalanobis_vals, relevance_vals, alpha=0.6, edgecolors="w", s=80)
    
    title = (
        f'LLM Relevance vs. Mahalanobis Distance\n'
        f'{config.representation_method} | {config.vectorizer_method}\n'
        f'{config.num_visits} Visits | {config.distance_metric} | {config.num_neighbors} Neighbors'
    )
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel('Mahalanobis Distance to Neighbors', fontsize=14)
    ax.set_ylabel('Average LLM Relevance Score', fontsize=14)
    
    stats_text = f"Spearman's Rho: {rho:.4f}\nP-value: {pval:.4f}\nN: {len(results)}"
    ax.text(0.95, 0.05, stats_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
            
    fig.tight_layout()
    
    plt.savefig(graph_output_path, dpi=300)
    print(f"\n✅ Magnificent graph saved to: {graph_output_path}")

    summary = {
        "spearman_rho": rho,
        "p_value": pval,
        "num_data_points": len(results)
    }
    with open(summary_output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Summary statistics saved to {summary_output_path}")


if __name__ == "__main__":
    main()