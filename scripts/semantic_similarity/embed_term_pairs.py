import os
import json
import argparse
from sentence_transformers import SentenceTransformer, util

# === Argument parsing ===
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="Model name from HuggingFace Hub")
args = parser.parse_args()
model_name = args.model
print(f"🔍 Using model: {model_name}")

# === Load the model ===
model = SentenceTransformer(model_name)

# === Load term pairs ===
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))
term_pairs_path = os.path.join(root_dir, "data", "term_pairs.json")

with open(term_pairs_path, "r") as f:
    term_pairs = json.load(f)

# === Compute similarities ===
results = []
for pair in term_pairs:
    term = pair["term"]
    counterpart = pair["counterpart"]
    term_embedding = model.encode(term, convert_to_tensor=True)
    counterpart_embedding = model.encode(counterpart, convert_to_tensor=True)
    cosine_sim = util.cos_sim(term_embedding, counterpart_embedding).item()

    results.append({
        "term": term,
        "commercial": counterpart,
        "cosine_similarity": cosine_sim,
        "model": model_name
    })

# === Save to file ===
output_path = os.path.join(root_dir, "data", "embeddings", model_name.replace("/", "-") + ".json")
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"✅ Done. Saved to {output_path}")
