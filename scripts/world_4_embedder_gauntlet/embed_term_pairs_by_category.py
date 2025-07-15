import os
import sys
import pandas as pd
import argparse
from pathlib import Path
from sentence_transformers import SentenceTransformer

# --- Dynamic sys.path adjustment ---
current_script_dir = Path(__file__).resolve().parent
project_root = current_script_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.common.config import setup_config, get_global_config

def main():
    parser = argparse.ArgumentParser(description="Embed term pairs and calculate similarity for a given model.")
    parser.add_argument("--model", type=str, required=True, help="Name of the SentenceTransformer model to use.")
    args = parser.parse_args()

    # --- ✨ NEW: Pre-computation Check ✨ ---
    # We'll check if the output for the LAST category already exists.
    # If it does, we can assume the whole script has run for this model.
    model_name_safe = args.model.replace("/", "-")
    output_check_path = project_root / "data" / "embeddings_by_category" / "procedure" / f"{model_name_safe}.csv"

    if os.path.exists(output_check_path):
        print(f"✅ Hooray! Results for {args.model} already exist at {output_check_path}. Skipping!")
        return

    print(f"🚀 Starting embedding process for model: {args.model}")

    setup_config(vectorizer_method=args.model)
    config = get_global_config()

    # --- Path setup ---
    data_dir = project_root / "data"
    pairs_path = data_dir / "term_pairs_by_category.json"
    output_base_dir = data_dir / "embeddings_by_category"

    # --- Load Model ---
    model_path = f"/media/scratch/mferguson/models/{model_name_safe}"
    print(f"🗺️ Loading model from: {model_path}")
    model = SentenceTransformer(model_path, local_files_only=True)

    # --- Load Term Pairs ---
    print(f"📂 Loading term pairs from: {pairs_path}")
    with open(pairs_path, 'r') as f:
        term_pairs_by_category = json.load(f)

    # --- Process each category ---
    for category, pairs in term_pairs_by_category.items():
        print(f"\n--- Processing category: {category} ---")
        if not pairs:
            print("No pairs to process. Skipping.")
            continue

        output_dir = output_base_dir / category
        os.makedirs(output_dir, exist_ok=True)
        output_path = output_dir / f"{model_name_safe}.csv"

        terms1, terms2 = zip(*pairs)

        unique_terms = sorted(list(set(terms1 + terms2)))
        print(f"Found {len(unique_terms)} unique terms to embed.")

        embeddings = model.encode(unique_terms, show_progress_bar=True, batch_size=128)
        embedding_map = {term: emb for term, emb in zip(unique_terms, embeddings)}

        results_data = []
        for t1, t2 in pairs:
            if t1 in embedding_map and t2 in embedding_map:
                e1 = embedding_map[t1]
                e2 = embedding_map[t2]
                similarity = SentenceTransformer.util.cos_sim(e1, e2).item()
                results_data.append({"term_1": t1, "term_2": t2, "cosine_similarity": similarity})

        results_df = pd.DataFrame(results_data)
        results_df.to_csv(output_path, index=False)
        print(f"✅ Saved similarity scores for {category} to {output_path}")

    print("\n🎉 Glorious success! All categories have been processed!")

if __name__ == "__main__":
    main()