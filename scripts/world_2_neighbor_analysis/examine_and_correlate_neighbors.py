import os
import sys
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from scipy.spatial.distance import cosine # We only need the simple cosine distance now!
from pathlib import Path

# --- Dynamic sys.path adjustment ---
current_script_dir = Path(__file__).resolve().parent
project_root = current_script_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.common.config import setup_config, get_global_config
from scripts.common.llm.query_llm import query_llm # We'll build the prompt here!
from scripts.common.data_loading.load_patient_data import load_patient_data
from scripts.common.utils import get_visit_term_lists # Using our new, better helper!

def get_similarity_score(narrative1: str, narrative2: str) -> float:
    """
    A new, improved function that asks for SIMILARITY!
    """
    prompt = (
        "You are an expert clinical researcher. On a scale from 0 to 10, "
        "how clinically SIMILAR are the following two patient summaries? "
        "A score of 10 means they are almost identical clinical cases.\n\n"
        f"Patient A:\n{narrative1}\n\n"
        f"Patient B:\n{narrative2}\n\n"
        "Similarity Score (0-10):"
    )
    # This uses a simple regex to find the first number in the LLM's response
    # It's a quick and easy way to parse the score!
    import re
    response = query_llm(prompt)
    match = re.search(r"(\d+\.?\d*|\d*\.?\d+)", response)
    return float(match.group(1)) if match else 0.0

def get_narrative_from_visits(visits: list[dict]) -> str:
    """
    Creates a simple narrative string from a list of visit dictionaries.
    """
    # This is a simplified version of your llm_helper's get_narrative.
    # We can make this more complex later if we need to!
    narrative_parts = []
    for visit in visits:
        # A simple summary of the visit
        summary = f"Visit on {visit.get('StartVisit', 'unknown date')}: "
        diags = [d.get('Diagnosis_Name') for d in visit.get('diagnoses', []) if d.get('Diagnosis_Name')]
        if diags:
            summary += f"Diagnosed with {', '.join(diags)}. "
        meds = [m.get('MedSimpleGenericName') for m in visit.get('medications', []) if m.get('MedSimpleGenericName')]
        if meds:
            summary += f"Prescribed {', '.join(meds)}."
        narrative_parts.append(summary)
    return " ".join(narrative_parts)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze neighbor SIMILARITY using pairwise comparison.")
    parser.add_argument("--representation_method", default="visit_sentence")
    parser.add_argument("--vectorizer_method", required=True)
    parser.add_argument("--distance_metric", default="cosine")
    parser.add_argument("--num_visits", type=int, default=6)
    parser.add_argument("--num_patients", type=int, default=5000)
    parser.add_argument("--num_neighbors", type=int, default=5)
    parser.add_argument("--model_name", type=str, default="medgemma")
    parser.add_argument("--max_patients_to_process", type=int, default=100) # Let's keep this small for testing!
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

    base_dir = project_root / "data" / config.representation_method / config.vectorizer_method
    vectors_dir = base_dir / f"visits_{config.num_visits}" / f"patients_{config.num_patients}"
    data_dir = vectors_dir / f"metric_{config.distance_metric}" / f"neighbors_{config.num_neighbors}"
    
    os.makedirs(data_dir, exist_ok=True)
    
    vectors_path = vectors_dir / "all_vectors.pkl"
    neighbors_path = data_dir / "neighbors.pkl"
    graph_output_path = data_dir / "pairwise_correlation_results.png" # A new name for a new plot!
    summary_output_path = data_dir / "pairwise_correlation_summary.json"

    print("--- 📂 Loading Data ---")
    patient_data = load_patient_data()
    patient_lookup = {p["patient_id"]: p for p in patient_data}
    
    with open(neighbors_path, "rb") as f: neighbors_dict = pickle.load(f)
    with open(vectors_path, "rb") as f: vector_dict = pickle.load(f)

    print("\n--- 🚀 Starting Pairwise Analysis ---")
    results = []
    
    patients_processed = 0
    for (patient_id, visit_idx), patient_vector in vector_dict.items():
        if patients_processed >= args.max_patients_to_process:
            break
        
        print(f"\nProcessing Patient: {patient_id}")
        neighbors = neighbors_dict.get((patient_id, visit_idx), [])
        if not neighbors: continue
            
        patient_narrative = get_narrative_from_visits(patient_lookup[patient_id]["visits"][:visit_idx + 1])
        
        for (neighbor_id, neighbor_vidx), _, neighbor_vec in neighbors[:config.num_neighbors]:
            neighbor_narrative = get_narrative_from_visits(patient_lookup[neighbor_id]["visits"][:neighbor_vidx + 1])
            
            # --- THE NEW WAY! ONE PAIR, TWO METRICS! ---
            similarity_score = get_similarity_score(patient_narrative, neighbor_narrative)
            cosine_dist = cosine(patient_vector, neighbor_vec)
            
            print(f"  - Neighbor {neighbor_id}: LLM Sim = {similarity_score:.2f}, Cosine Dist = {cosine_dist:.2f}")
            results.append((similarity_score, cosine_dist))

        patients_processed += 1

    print("\n--- 📊 Final Correlation and Graph Generation ---")
    if not results:
        print("❌ No valid results were generated.")
        return

    llm_scores, vector_distances = zip(*results)
    rho, pval = spearmanr(llm_scores, vector_distances)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.scatter(vector_distances, llm_scores, alpha=0.6, edgecolors="w", s=80)
    
    ax.set_title("Pairwise LLM Similarity vs. Vector Distance", fontsize=16)
    ax.set_xlabel("Cosine Distance (Lower is More Similar)", fontsize=14)
    ax.set_ylabel("LLM Similarity Score", fontsize=14)
    
    stats_text = f"Spearman's Rho: {rho:.4f}\nP-value: {pval:.4f}\nN Pairs: {len(results)}"
    ax.text(0.95, 0.05, stats_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
            
    fig.tight_layout()
    plt.savefig(graph_output_path, dpi=300)
    print(f"\n✅ New pairwise graph saved to: {graph_output_path}")

    summary = {"spearman_rho": rho, "p_value": pval, "num_pairs": len(results)}
    with open(summary_output_path, "w") as f: json.dump(summary, f, indent=2)
    print(f"✅ New summary saved to {summary_output_path}")

if __name__ == "__main__":
    main()