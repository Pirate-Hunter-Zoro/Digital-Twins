import os
import sys
import json
from collections import defaultdict

# --- Dynamic sys.path adjustment ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End of sys.path adjustment ---

from scripts.config import setup_config, get_global_config
from scripts.eval.idf_utils import load_idf_registry
from scripts.eval.load_neighbors import load_neighbors_for_config
from scripts.eval.load_patient_results import load_all_patient_results
from scripts.read_data.load_patient_data import load_patient_data
from scripts.eval.scoring_utils import compute_weighted_cosine_score

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Setup
setup_config(
    representation_method="visit_sentence",
    vectorizer_method="biobert-mnli-mednli",
    distance_metric="euclidean",
    num_visits=6,
    num_patients=5000,
    num_neighbors=5,
)

# Load data
config = get_global_config()
neighbors_by_patient = load_neighbors_for_config(config)
predictions = load_all_patient_results(config)
idf_registry = load_idf_registry()
all_patients = {p["patient_id"]: p for p in load_patient_data()}

# Load vectorizer once
vectorizer = SentenceTransformer(
    f"/home/librad.laureateinstitute.org/mferguson/models/{config.vectorizer_method}",
    local_files_only=True
)

# Prepare output paths
os.makedirs("logs", exist_ok=True)
os.makedirs("data", exist_ok=True)
log_path = "logs/neighbor_analysis_errors.txt"
output_path = "data/neighbor_analysis_summary.json"

log_lines = []
output_data = []

print("--- Examining neighbors ---")
for patient_id, neighbors in neighbors_by_patient.items():
    patient = all_patients.get(patient_id)
    predicted = predictions.get(patient_id)

    if not patient or not predicted:
        log_lines.append(f"[SKIP] Missing data for patient {patient_id}")
        continue

    actual = patient["visits"][config.num_visits - 1]

    try:
        actual_terms = set(actual["diagnoses"] + actual["medications"] + actual["treatments"])
        predicted_terms = set(predicted["diagnoses"] + predicted["medications"] + predicted["treatments"])

        if not actual_terms or not predicted_terms:
            log_lines.append(f"[SKIP] No terms to compare for patient {patient_id}")
            continue

        actual_vecs = vectorizer.encode(list(actual_terms), convert_to_numpy=True)
        predicted_vecs = vectorizer.encode(list(predicted_terms), convert_to_numpy=True)

        sim_matrix = cosine_similarity(actual_vecs, predicted_vecs)
        scores = compute_weighted_cosine_score(
            sim_matrix, list(actual_terms), list(predicted_terms), idf_registry
        )

        output_data.append({
            "patient_id": patient_id,
            "top_neighbors": [n[0][0] for n in neighbors[:3]],
            "prediction_scores": scores
        })
    except Exception as e:
        log_lines.append(f"[ERROR] Failed scoring for patient {patient_id}: {str(e)}")

# Save outputs
with open(output_path, "w") as f:
    json.dump(output_data, f, indent=2)

with open(log_path, "w") as f:
    f.write("\n".join(log_lines))

print(f"✅ Done. {len(output_data)} patients processed.")
print(f"📄 Output: {output_path}")
print(f"🪵 Log: {log_path}")
