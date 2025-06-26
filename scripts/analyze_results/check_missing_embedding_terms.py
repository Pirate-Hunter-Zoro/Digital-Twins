
import json
import pickle
from pathlib import Path

# --- Config paths ---
project_root = Path(__file__).resolve().parent.parent.parent
data_dir = project_root / "data"

results_path = data_dir / "patient_results_5000_6_bag_of_codes_sentence_transformer_euclidean.json"
embeddings_path = data_dir / "term_embedding_library_mpnet.pkl"

# --- Load data ---
print(f"📦 Loading results from: {results_path.name}")
with open(results_path, "r") as f:
    results = json.load(f)

print(f"📦 Loading embeddings from: {embeddings_path.name}")
with open(embeddings_path, "rb") as f:
    embedding_library = pickle.load(f)

# --- Gather all terms ---
all_terms = set()
for entry in results.values():
    for cat in ["predicted", "actual"]:
        cat_terms = entry.get(cat, {})
        for subcat in ["diagnoses", "medications", "treatments"]:
            all_terms.update(cat_terms.get(subcat, []))

# --- Check which are missing ---
missing_terms = [term for term in all_terms if term not in embedding_library]

print(f"🔍 Total unique terms from result file: {len(all_terms)}")
print(f"❌ Terms missing from embedding library: {len(missing_terms)}")

if missing_terms:
    print("\n🧟‍♂️ Terms with NO embeddings:")
    for term in sorted(missing_terms)[:30]:  # Show first 30
        print(f" - {term}")
    if len(missing_terms) > 30:
        print(f"...and {len(missing_terms) - 30} more.")
else:
    print("✅ All terms are covered by the embedding library!")
