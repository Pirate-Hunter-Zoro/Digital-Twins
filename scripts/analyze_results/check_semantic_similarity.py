import os
import sys
import json
import pickle
import numpy as np
from pathlib import Path
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer

# --- Dynamic path setup ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Imports ---
from scripts.calculations.prepare_categorized_embedding_terms import clean_term
from scripts.config import setup_config, get_global_config

# --- Cosine similarity ---
def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b) + 1e-8)

# --- Fallback vector generator ---
def embed_term(term, model):
    return model.encode(term, convert_to_numpy=True, normalize_embeddings=True)

def main():
    # --- Config setup ---
    setup_config(num_visits=6, num_neighbors=10, num_patients=5000,
                 vectorizer_method='biobert-mnli-mednli',
                 distance_metric='euclidean', representation_method='bag_of_codes')
    config = get_global_config()
    ROOT_DIR = Path(__file__).resolve().parents[2]
    PROJ_LOC = Path(__file__).resolve().parents[3]

    # --- Load model and embeddings ---
    print(f"🧠 Loading model from: {config.vectorizer_method}")
    model = SentenceTransformer(str(PROJ_LOC / "models" / config.vectorizer_method))
    with open(ROOT_DIR / "data" / "term_embedding_library_by_category.pkl", "rb") as f:
        grouped_data = pickle.load(f)

    # --- Define term pairs ---
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

    # --- Process and save similarities ---
    results = {}
    for category, pairs in test_pairs.items():
        cat_results = []
        embeddings = grouped_data.get(category, {})
        for term_a, term_b in pairs:
            a_clean = clean_term(term_a)
            b_clean = clean_term(term_b)

            vec_a = embeddings.get(a_clean)
            vec_b = embeddings.get(b_clean)

            if vec_a is None:
                vec_a = embed_term(a_clean, model)
            if vec_b is None:
                vec_b = embed_term(b_clean, model)

            sim_score = cosine_similarity(vec_a, vec_b)
            cat_results.append({
                "term_a": term_a,
                "term_b": term_b,
                "cosine_similarity": float(f"{sim_score:.4f}"),
                "term_a_cleaned": a_clean,
                "term_b_cleaned": b_clean,
                "generated_a": a_clean not in embeddings,
                "generated_b": b_clean not in embeddings
            })
        results[category] = cat_results

    # --- Save results ---
    output_path = ROOT_DIR / "data" / "semantic_similarity_examples.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Results saved to {output_path}")

if __name__ == "__main__":
    main()
