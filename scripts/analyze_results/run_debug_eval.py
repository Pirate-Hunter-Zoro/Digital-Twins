import json
import pickle
import random
import os
from evaluate import evaluate_prediction_by_category

# 🔧 Update these paths as needed
RESULTS_PATH = "data/patient_results_5000_6_bag_of_codes_sentence_transformer_euclidean.json"
IDF_REGISTRY_PATH = "data/term_idf_registry.json"
EMBEDDINGS_PATH = "data/term_embedding_library_mpnet_combined.pkl"

print(f"Loading results from: {os.path.abspath(RESULTS_PATH)}")
with open(RESULTS_PATH, "r") as f:
    results = json.load(f)

print(f"Loading IDF registry from: {os.path.abspath(IDF_REGISTRY_PATH)}")
with open(IDF_REGISTRY_PATH, "r") as f:
    idf_registry = json.load(f)

print(f"Loading term embeddings from: {os.path.abspath(EMBEDDINGS_PATH)}")
with open(EMBEDDINGS_PATH, "rb") as f:
    term_embeddings = pickle.load(f)

print("\n========== DEBUGGING SAMPLE PATIENTS ==========\n")

sample_patients = random.sample(list(results.keys()), 5)
for pid in sample_patients:
    print(f"🔬 Evaluating Patient ID: {pid}\n")
    predicted = results[pid]["predicted"]
    actual = results[pid]["actual"]

    print("📦 Raw prediction content:")
    print(json.dumps(predicted, indent=2))
    print("\n📦 Ground truth content:")
    print(json.dumps(actual, indent=2))

    scores = evaluate_prediction_by_category(predicted, actual, term_embeddings, similarity_threshold=0.4)
    print("📊 Score Breakdown:", json.dumps(scores, indent=2))
    print("\n" + "="*80 + "\n")
