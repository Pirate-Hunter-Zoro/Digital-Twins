import os
import sys
import json
import csv
import argparse
from sentence_transformers import SentenceTransformer, util

# --- The Magnificent Fix! ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ---------------------------

# === Argument parsing ===
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="Model name from HuggingFace Hub")
args = parser.parse_args()
model_name = args.model
print(f"🔍 Using model: {model_name}")

# === Load term pairs ===
term_pairs_path = os.path.join(project_root, "data", "term_pairs.json")

# === Load the model ===
sanitized_model_name = model_name.replace("/", "-")
local_model_path = f"/media/scratch/mferguson/models/{sanitized_model_name}"

print(f"📦 Loading local model from: {local_model_path}")
model = SentenceTransformer(local_model_path)

with open(term_pairs_path, "r") as f:
    term_pairs = json.load(f)

# === Compute similarities and WRITE to the CSV! ===
output_dir = os.path.join(project_root, "data", "embeddings")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, sanitized_model_name + ".csv")

with open(output_path, 'w', newline='', encoding='utf-8') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(['term', 'counterpart', 'cosine_similarity', 'model'])

    print("🧠 Okay, let's get to work! Calculating similarities!")
    for pair in term_pairs:
        term = pair["term"]
        counterpart = pair["counterpart"]
        term_embedding = model.encode(term, convert_to_tensor=True)
        counterpart_embedding = model.encode(counterpart, convert_to_tensor=True)
        cosine_sim = util.cos_sim(term_embedding, counterpart_embedding).item()
        writer.writerow([term, counterpart, cosine_sim, model_name])

print(f"✅ YAY! Done! Your beautiful new CSV is saved at {output_path}")