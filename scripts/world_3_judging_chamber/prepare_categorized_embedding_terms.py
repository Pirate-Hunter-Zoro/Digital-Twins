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

from scripts.common.config import setup_config, get_global_config

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


import os
import sys
import re
import json
from pathlib import Path
from collections import defaultdict
import argparse

# --- Dynamic sys.path adjustment ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = Path(current_script_dir).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# --- Term cleaner ---
def clean_term(term: str) -> str:
    term = term.lower().strip()
    term = term.replace('"', '').replace("'", '')
    term = re.sub(r"^\\+|\\+$", "", term)
    term = re.sub(r"\([^)]*hcc[^)]*\)", "", term)
    term = re.sub(r"\(cms[^)]*\)", "", term)
    term = re.sub(r"\b\(\d{3}(\.\d+)?\)", "", term)
    term = re.sub(r",+", "", term)

    blacklist = [
        "initial encounter", "unspecified", "nos", "nec",
        "<none>", "<None>", ";", ":", 
        "at \d+ oclock position", "during pregnancy.*",
        "due to.*", "with late onset", "with dysplasia", "without dysplasia"
    ]
    for noise in blacklist:
        term = re.sub(noise, "", term)

    return re.sub(r"\s+", " ", term).strip()


def main(args):
    data_dir = project_root / "data"

    json_filename = (
        f"patient_results_{args.num_patients}_{args.num_visits}_"
        f"visit_sentence_{args.vectorizer_method}_{args.distance_metric}.json"
    )
    TERM_JSON_PATH = data_dir / json_filename
    OUTPUT_PATH = data_dir / f"grouped_terms_by_category_{args.vectorizer_method}_{args.num_patients}_{args.num_visits}.json"

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

    final_terms = {cat: sorted(list(terms)) for cat, terms in categorized_terms.items()}

    with open(OUTPUT_PATH, "w") as f:
        json.dump(final_terms, f, indent=2)

    print(f"✅ Saved grouped terms to {OUTPUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectorizer_method", required=True)
    parser.add_argument("--num_visits", type=int, default=6)
    parser.add_argument("--num_patients", type=int, default=5000)
    parser.add_argument("--distance_metric", default="euclidean")
    args = parser.parse_args()

    main(args)