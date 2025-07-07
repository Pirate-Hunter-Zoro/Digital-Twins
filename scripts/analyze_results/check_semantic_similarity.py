import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from numpy.linalg import norm

import sys
from pathlib import Path

current_script_dir = Path(__file__).resolve().parent
scripts_dir = current_script_dir.parent
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

from calculations.prepare_categorized_embedding_terms import clean_term
from config import setup_config, get_global_config
from models.embedder import FallbackEmbedder

from transformers import AutoTokenizer, AutoModel
import torch
from sentence_transformers import SentenceTransformer

# --- Cosine similarity ---
def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b) + 1e-8)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectorizer_method", required=True)
    args = parser.parse_args()

    project_loc = Path(__file__).resolve().parents[3]
    model_path = project_loc / "models" / args.vectorizer_method

    test_pairs = {
        "diagnoses": [
            ("heart attack", "myocardial infarction"),
            ("high blood pressure", "hypertension"),
            ("alzheimers disease", "major neurocognitive disorder due to probable alzheimers disease")
        ],
        "treatments": [
            ("mri brain", "magnetic resonance imaging of the head"),
            ("xr wrist", "x-ray of the hand")
        ],
        "medications": [
            ("acetaminophen", "tylenol"),
            ("ibuprofen", "advil")
        ]
    }

    embedder, embed_type = FallbackEmbedder(model_path)

    results = {}
    for category, pairs in test_pairs.items():
        cat_results = []
        for term_a, term_b in pairs:
            a_clean = clean_term(term_a)
            b_clean = clean_term(term_b)

            vec_a = embedder.encode(a_clean) if embed_type == "sentence-transformers" else embedder(a_clean)
            vec_b = embedder.encode(b_clean) if embed_type == "sentence-transformers" else embedder(b_clean)

            sim_score = cosine_similarity(vec_a, vec_b)
            cat_results.append({
                "term_a": term_a,
                "term_b": term_b,
                "cosine_similarity": float(f"{sim_score:.4f}"),
                "term_a_cleaned": a_clean,
                "term_b_cleaned": b_clean,
            })
        results[category] = cat_results

    output_path = project_loc / "Digital-Twins" / "data" / f"semantic_similarity_examples_{args.vectorizer_method}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Results saved to {output_path}")

if __name__ == "__main__":
    main()
