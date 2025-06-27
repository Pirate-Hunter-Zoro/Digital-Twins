import json
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# 🧼 Term cleaner (safe fallback if not already done)
def clean_term(term: str) -> str:
    return term.strip().lower()

def main():
    # 📂 Dynamic paths
    ROOT_DIR = Path(__file__).resolve().parents[2]
    PROJ_LOC = Path(__file__).resolve().parents[3]
    TERM_JSON_PATH = ROOT_DIR / "data" / "grouped_terms_by_category.json"
    OUTPUT_PATH = ROOT_DIR / "data" / "term_embedding_library_by_category.pkl"
    MODEL_PATH = PROJ_LOC / "models" / "biobert-mnli-mednli"

    print("🧬 Beginning embedding generation by category...")
    print(f"📥 Loading terms from: {TERM_JSON_PATH}")
    print(f"🧠 Loading model from: {MODEL_PATH}")

    model = SentenceTransformer(str(MODEL_PATH))

    with open(TERM_JSON_PATH, "r") as f:
        categorized_terms = json.load(f)

    embedding_library = {}

    for category, term_list in categorized_terms.items():
        print(f"\n📦 Processing category: {category} ({len(term_list)} terms)")
        cleaned_terms = list({clean_term(t) for t in term_list if t.strip()})
        embeddings = model.encode(cleaned_terms, show_progress_bar=True, batch_size=64)

        embedding_library[category] = {
            term: vec for term, vec in zip(cleaned_terms, embeddings)
        }

    print(f"\n💾 Saving categorized embedding library to: {OUTPUT_PATH}")
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(embedding_library, f)

    print("✅ All category embeddings generated and saved!")

if __name__ == "__main__":
    main()
