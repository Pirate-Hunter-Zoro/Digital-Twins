import os
import sys
import json
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import argparse

# --- Dynamic sys.path adjustment ---
current_script_dir = Path(__file__).resolve().parent
project_root = current_script_dir.parents[2]
project_loc = current_script_dir.parents[3]  # Where /models lives

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.analyze_results.prepare_categorized_embedding_terms import clean_term

def main(args):
    model_dir = project_loc / "models" / args.model_name
    model = SentenceTransformer(str(model_dir), local_files_only=True)

    input_path = project_root / "data" / f"grouped_terms_by_category_{args.model_name}_{args.num_patients}_{args.num_visits}.json"
    output_path = project_root / "data" / f"term_embedding_library_by_category_{args.model_name}_{args.num_patients}_{args.num_visits}.pkl"

    with open(input_path, "r") as f:
        grouped_terms = json.load(f)

    embedding_library = {}
    for category, terms in grouped_terms.items():
        cleaned_terms = [clean_term(t) for t in terms]
        embeddings = model.encode(cleaned_terms, show_progress_bar=True, convert_to_numpy=True)
        embedding_library[category] = [
            {"term": t, "embedding": emb} for t, emb in zip(cleaned_terms, embeddings)
        ]

    with open(output_path, "wb") as f:
        pickle.dump(embedding_library, f)

    print(f"✅ Saved embeddings to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--num_patients", type=int, default=5000)
    parser.add_argument("--num_visits", type=int, default=6)
    args = parser.parse_args()
    main(args)
