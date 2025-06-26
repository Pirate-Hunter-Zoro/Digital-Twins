import json
import pickle
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# 🔧 Paths
TERM_JSON_PATH = "data/combined_terms_for_embedding.json"
OUTPUT_PATH = "data/term_embedding_library_mpnet_combined.pkl"
MODEL_PATH = "/home/librad.laureateinstitute.org/mferguson/models/biobert-mnli-mednli"

import re

def clean_term(term: str) -> str:
    term = term.lower().strip()
    term = re.sub(r"\s*\([^)]*hcc[^)]*\)", "", term)
    term = re.sub(r"\b\d{3}\.\d{1,2}\b", "", term)
    blacklist = ["initial encounter", "unspecified", "nos", "nec", "<none>", "<None>", ";", ":"]
    for noise in blacklist:
        term = term.replace(noise, "")
    term = re.sub(r"\s+", " ", term)
    return term.strip()

print("🧬 MPNet Term Embedding — Enhanced Twin Edition")
print(f"📥 Loading terms from: {TERM_JSON_PATH}")

with open(TERM_JSON_PATH, "r") as f:
    all_terms = json.load(f)

terms_to_embed = sorted(set(clean_term(term) for term in all_terms if term and term.strip()))

print(f"🔢 Total terms to embed: {len(terms_to_embed)}")
print(f"🧠 Loading MPNet model from: {MODEL_PATH}")
model = SentenceTransformer(MODEL_PATH)

print("💥 Embedding in progress...")
embeddings = model.encode(terms_to_embed, batch_size=64, show_progress_bar=True)

term_embeddings = {term: vector for term, vector in zip(terms_to_embed, embeddings)}

print(f"💾 Saving to: {OUTPUT_PATH}")
with open(OUTPUT_PATH, "wb") as f:
    pickle.dump(term_embeddings, f)

print("✅ Embedding complete! No term shall be left behind!")
