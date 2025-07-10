import os
import json
import csv # We need to give it the CSV instruction manual!
import argparse
from sentence_transformers import SentenceTransformer, util

# === Argument parsing ===
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="Model name from HuggingFace Hub")
args = parser.parse_args()
model_name = args.model
print(f"🔍 Using model: {model_name}")

# === Load term pairs ===
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))
term_pairs_path = os.path.join(root_dir, "data", "term_pairs.json")

# === Load the model ===
# Sanitize the model name to match the folder name we created!
sanitized_model_name = model_name.replace("/", "-")
# The path to our NEW, super-spacious, scratch-tastic model home!
local_model_path = f"/media/scratch/mferguson/models/{sanitized_model_name}"

print(f"📦 Loading local model from: {local_model_path}")
# Load it from our very own folder! No internet required!
model = SentenceTransformer(local_model_path)

with open(term_pairs_path, "r") as f:
    term_pairs = json.load(f)

# === THIS IS THE NEW PART! ===
# Where we're going to save our beautiful new CSV file!
output_dir = os.path.join(root_dir, "data", "embeddings")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, sanitized_model_name + ".csv")

# === Compute similarities and WRITE to the CSV! ===
with open(output_path, 'w', newline='', encoding='utf-8') as outfile:
    writer = csv.writer(outfile)
    # A beautiful header for our beautiful data!
    writer.writerow(['term', 'counterpart', 'cosine_similarity', 'model'])

    print("🧠 Okay, let's get to work! Calculating similarities!")
    for pair in term_pairs:
        term = pair["term"]
        counterpart = pair["counterpart"]
        term_embedding = model.encode(term, convert_to_tensor=True)
        counterpart_embedding = model.encode(counterpart, convert_to_tensor=True)
        cosine_sim = util.cos_sim(term_embedding, counterpart_embedding).item()

        # Write the results right away! So efficient!
        writer.writerow([term, counterpart, cosine_sim, model_name])

print(f"✅ YAY! Done! Your beautiful new CSV is saved at {output_path}")