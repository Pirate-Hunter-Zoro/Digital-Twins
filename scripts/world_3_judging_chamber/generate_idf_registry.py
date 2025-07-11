import os
import sys
import argparse
import json
from collections import defaultdict
from math import log

# --- The Magnificent Fix! ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ---------------------------

from scripts.common.config import setup_config, get_global_config
from scripts.common.utils import clean_term

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--representation_method", required=True)
    parser.add_argument("--vectorizer_method", required=True)
    parser.add_argument("--distance_metric", required=True)
    parser.add_argument("--num_visits", type=int, required=True)
    parser.add_argument("--num_patients", type=int, required=True)
    parser.add_argument("--num_neighbors", type=int, required=True)
    parser.add_argument("--model_name", required=True)
    args = parser.parse_args()

    setup_config(
        representation_method=args.representation_method,
        vectorizer_method=args.vectorizer_method,
        distance_metric=args.distance_metric,
        num_visits=args.num_visits,
        num_patients=args.num_patients,
        num_neighbors=args.num_neighbors,
        model_name=args.model_name,
    )
    config = get_global_config()

    filename_suffix = f"{config.num_patients}_{config.num_visits}_{config.representation_method}_{config.vectorizer_method}_{config.distance_metric}_{config.model_name}"
    input_path = os.path.join(project_root, "data", f"patient_results_{filename_suffix}.json")
    output_path = os.path.join(project_root, "data", f"idf_registry_{filename_suffix}.json")

    print(f"📥 Loading patient results from: {input_path}")
    with open(input_path, "r") as f:
        patient_results = json.load(f)

    term_counts = defaultdict(int)
    total_docs = 0

    for patient_id, data in patient_results.items():
        all_terms = set()
        for section in ["predicted", "actual"]:
            for category in ["diagnoses", "medications", "treatments"]:
                terms = data.get(section, {}).get(category, [])
                cleaned_terms = {clean_term(t) for t in terms if clean_term(t)}
                all_terms |= cleaned_terms
        for term in all_terms:
            term_counts[term] += 1
        total_docs += 1

    idf_scores = {
        term: log(total_docs / (count + 1))  # Added smoothing! So robust!
        for term, count in term_counts.items()
    }

    with open(output_path, "w") as f:
        json.dump(idf_scores, f, indent=2)

    print(f"✅ Saved IDF registry to: {output_path}")

if __name__ == "__main__":
    main()