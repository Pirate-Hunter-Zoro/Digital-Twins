import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from scipy.spatial.distance import cosine
from pathlib import Path

# --- Dynamic sys.path adjustment ---
current_script_dir = Path(__file__).resolve().parent
project_root = current_script_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.common.config import setup_config, get_global_config
from scripts.common.llm.query_llm import query_llm
from scripts.common.data_loading.load_patient_data import load_patient_data

def get_similarity_score(narrative1: str, narrative2: str) -> float:
    prompt = (
        "On a scale from 0 to 10, how clinically SIMILAR are the following two patient summaries? "
        f"Patient A:\n{narrative1}\n\nPatient B:\n{narrative2}\n\nSimilarity Score (0-10):"
    )
    import re
    response = query_llm(prompt)
    match = re.search(r"(\d+\.?\d*|\d*\.?\d+)", response)
    return float(match.group(1)) if match else 0.0

def get_narrative_from_visits(visits: list[dict]) -> str:
    narrative_parts = []
    for visit in visits:
        summary = f"On {visit.get('StartVisit', 'a visit')}, "
        diags = [d.get('Diagnosis_Name') for d in visit.get('diagnoses', []) if d.get('Diagnosis_Name')]
        if diags: summary += f"diagnosed with {', '.join(diags)}. "
        meds = [m.get('MedSimpleGenericName') for m in visit.get('medications', []) if m.get('MedSimpleGenericName')]
        if meds: summary += f"Prescribed {', '.join(meds)}."
        narrative_parts.append(summary)
    return " ".join(narrative_parts) if narrative_parts else "No visit information."

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze pairwise similarity for near AND far neighbors from the full ranked list.")
    parser.add_argument("--vectorizer_method", required=True)
    parser.add_argument("--num_visits", type=int, default=6)
    parser.add_argument("--k_neighbors", type=int, default=5, help="Number of nearest/farthest neighbors to analyze.")
    args = parser.parse_args()

    setup_config("visit_sentence", args.vectorizer_method, "cosine", args.num_visits, 5000, args.k_neighbors)
    config = get_global_config()

    # --- Setup Paths ---
    base_dir = project_root / "data" / "visit_sentence" / config.vectorizer_method.replace('/', '-')
    data_dir = base_dir / f"visits_{config.num_visits}" / "patients_5000"
    neighbor_dir = data_dir / "metric_cosine"
    
    vectors_path = data_dir / "all_vectors.pkl"
    # --- ✨ Pointing to our new Behemoth file! ✨ ---
    neighbors_path = neighbor_dir / "all_ranked_neighbors.pkl"
    graph_output_path = neighbor_dir / f"full_range_correlation_k{config.num_neighbors}.png"
    summary_output_path = neighbor_dir / f"full_range_correlation_summary_k{config.num_neighbors}.json"

    print("--- 📂 Loading Data ---")
    patient_lookup = {p["patient_id"]: p for p in load_patient_data()}
    with open(neighbors_path, "rb") as f: all_ranked_neighbors = pickle.load(f)
    with open(vectors_path, "rb") as f: vector_dict = pickle.load(f)
    
    print("\n--- 🚀 Starting Full Range Pairwise Analysis ---")
    results = []
    
    patients_to_process = random.sample(list(all_ranked_neighbors.keys()), k=min(25, len(all_ranked_neighbors)))

    for patient_key in patients_to_process:
        patient_id, visit_idx = patient_key
        patient_vector = vector_dict[patient_key]
        print(f"\nProcessing Patient: {patient_id}")

        ranked_list = all_ranked_neighbors.get(patient_key, [])
        if not ranked_list: continue

        # --- ✨ Brilliant Slicing Logic! ✨ ---
        near_neighbors = ranked_list[:config.num_neighbors]
        far_neighbors = ranked_list[-config.num_neighbors:]

        patient_narrative = get_narrative_from_visits(patient_lookup[patient_id]["visits"][:visit_idx + 1])
        
        for pair_type, neighbor_list in [("near", near_neighbors), ("far", far_neighbors)]:
            for neighbor_key, similarity_from_ranking in neighbor_list:
                neighbor_vec = vector_dict[neighbor_key]
                neighbor_id, neighbor_vidx = neighbor_key
                
                neighbor_narrative = get_narrative_from_visits(patient_lookup[neighbor_id]["visits"][:neighbor_vidx + 1])
                
                llm_score = get_similarity_score(patient_narrative, neighbor_narrative)
                
                # We can reuse the similarity from our big matrix calculation! So efficient!
                cosine_sim = similarity_from_ranking

                results.append({
                    "type": pair_type,
                    "llm_score": llm_score,
                    "cosine_similarity": cosine_sim
                })

    # --- 📊 Final Correlation and Graph Generation ---
    if not results:
        print("❌ No results to plot!")
        return
        
    df = pd.DataFrame(results)
    
    print("\n--- Generating New Plot ---")
    sns.jointplot(data=df, x="cosine_similarity", y="llm_score", hue="type", palette={"near": "blue", "far": "red"}, height=10, alpha=0.7)
    plt.suptitle("LLM Similarity vs. Vector Similarity for Near and Far Pairs", y=1.02, size=16)
    plt.savefig(graph_output_path, dpi=300)
    print(f"✅ Magnificent new graph saved to: {graph_output_path}")

    summary = df.groupby('type').agg(['mean', 'std']).to_dict()
    with open(summary_output_path, "w") as f: json.dump(summary, f, indent=2)
    print(f"✅ Summary statistics saved to {summary_output_path}")

if __name__ == "__main__":
    main()