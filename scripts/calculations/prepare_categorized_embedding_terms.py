# scripts/calculations/prepare_categorized_embedding_terms.py

import json
import re
import os
import sys
from collections import defaultdict
from pathlib import Path

# --- Dynamic sys.path adjustment for module imports ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End of sys.path adjustment ---

from scripts.config import setup_config, get_global_config

# --- Term cleaner ---
def clean_term(term: str) -> str:
    term = term.lower().strip()

    # Strip all types of quotes and leading/trailing slashes
    term = term.replace('"', '').replace("'", '')
    term = re.sub(r"^\\+|\\+$", "", term)

    # Remove known annotation patterns like (hhs/hcc), (cms/hcc), ICD-9/10 codes in parentheses
    term = re.sub(r"\([^)]*hcc[^)]*\)", "", term)
    term = re.sub(r"\(cms[^)]*\)", "", term)
    term = re.sub(r"\b\(\d{3}(\.\d+)?\)", "", term)

    # Remove redundant commas
    term = re.sub(r",+", "", term)

    # Common low-information descriptors
    blacklist = [
        "initial encounter", "unspecified", "nos", "nec",
        "<none>", "<None>", ";", ":", 
        "at \d+ oclock position", "during pregnancy.*",  # generalize weird suffixes
        "due to.*", "with late onset", "with dysplasia", "without dysplasia"
    ]
    for noise in blacklist:
        term = re.sub(noise, "", term)

    # Remove trailing or repeated whitespace
    term = re.sub(r"\s+", " ", term)

    return term.strip()


# --- Main term categorizer ---
def main():
    config = get_global_config()
    ROOT_DIR = Path(__file__).resolve().parents[2]

    json_filename = f"patient_results_{config.num_patients}_{config.num_visits}_{config.representation_method}_{config.vectorizer_method}_{config.distance_metric}.json"
    TERM_JSON_PATH = ROOT_DIR / "data" / json_filename
    OUTPUT_PATH = ROOT_DIR / "data" / "grouped_terms_by_category.json"

    with open(TERM_JSON_PATH, "r") as f:
        results = json.load(f)

    print(f"📥 Loaded {len(results)} patient prediction records from {TERM_JSON_PATH.name}")

    categorized_terms = defaultdict(set)
    for patient_id, data in results.items():
        for section in ["predicted", "actual"]:
            for category in ["diagnoses", "medications", "treatments"]:
                terms = data.get(section, {}).get(category, [])
                for term in terms:
                    cleaned = clean_term(term)
                    if cleaned:
                        categorized_terms[category].add(cleaned)

    # Flatten sets to sorted lists
    final_terms = {cat: sorted(list(terms)) for cat, terms in categorized_terms.items()}

    with open(OUTPUT_PATH, "w") as f:
        json.dump(final_terms, f, indent=2)

    print(f"✅ Saved grouped terms to {OUTPUT_PATH}")

if __name__ == "__main__":
    setup_config(num_visits=6, num_neighbors=10, num_patients=5000, vectorizer_method='sentence_transformer',
                 distance_metric='euclidean', representation_method='bag_of_codes')
    main()
