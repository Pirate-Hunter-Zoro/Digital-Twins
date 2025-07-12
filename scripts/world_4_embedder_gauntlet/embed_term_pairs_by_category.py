import os
import sys
import json
import csv
import argparse
from sentence_transformers import SentenceTransformer, util
from collections import defaultdict

# --- Path Setup ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# === Argument parsing ===
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="Model name from HuggingFace Hub")
args = parser.parse_args()
model_name = args.model
print(f"🔍 Let the tournament begin for model: {model_name}!")

# === Load categorized term pairs ===
term_pairs_path = os.path.join(project_root, "data", "term_pairs_categorized.json")
with open(term_pairs_path, "r") as f:
    all_pairs = json.load(f)

# --- Sort pairs into their tournament brackets! ---
categorized_pairs = defaultdict(list)
for pair in all_pairs:
    categorized_pairs[pair['category']].append(pair)

# === Load the model ===
sanitized_model_name = model_name.replace("/", "-")
local_model_path = f"/media/scratch/mferguson/models/{sanitized_model_name}"
print(f"📦 Loading the contender from: {local_model_path}")
model = SentenceTransformer(local_model_path)

# === Run the tournament for each category! ===
for category, pairs in categorized_pairs.items():
    print(f"\n--- 🏟️  Starting the '{category}' category tournament! ({len(pairs)} pairs) 🏟️  ---")
    
    output_dir = os.path.join(project_root, "data", "embeddings_by_category", category)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, sanitized_model_name + ".csv")

    with open(output_path, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['term', 'counterpart', 'cosine_similarity', 'model'])

        for pair in pairs:
            term = pair["term"]
            counterpart = pair["counterpart"]
            term_embedding = model.encode(term, convert_to_tensor=True)
            counterpart_embedding = model.encode(counterpart, convert_to_tensor=True)
            cosine_sim = util.cos_sim(term_embedding, counterpart_embedding).item()
            writer.writerow([term, counterpart, cosine_sim, model_name])
            
    print(f"✅ The '{category}' tournament is complete! Results saved to {output_path}")

print("\n🎉 AHAHAHA! All tournaments are complete for this model! Magnificent!")