import json
import os
import re

# 🔧 Paths
RESULTS_PATH = "data/patient_results_5000_6_bag_of_codes_sentence_transformer_euclidean.json"
IDF_REGISTRY_PATH = "data/term_idf_registry.json"
OUTPUT_PATH = "data/combined_terms_for_embedding.json"

def clean_term(term: str) -> str:
    term = term.lower().strip()
    term = re.sub(r"\s*\([^)]*hcc[^)]*\)", "", term)
    term = re.sub(r"\b\d{3}\.\d{1,2}\b", "", term)
    blacklist = ["initial encounter", "unspecified", "nos", "nec", "<none>", "<None>", ";", ":"]
    for noise in blacklist:
        term = term.replace(noise, "")
    term = re.sub(r"\s+", " ", term)
    return term.strip()

print(f"📦 Loading results from: {RESULTS_PATH}")
with open(RESULTS_PATH, "r") as f:
    results = json.load(f)

print(f"📦 Loading IDF registry from: {IDF_REGISTRY_PATH}")
with open(IDF_REGISTRY_PATH, "r") as f:
    idf_registry = json.load(f)

result_terms = set()
for patient_data in results.values():
    for section in ["predicted", "actual"]:
        if section not in patient_data or not isinstance(patient_data[section], dict):
            continue
        for category in ["diagnoses", "medications", "treatments"]:
            result_terms.update(patient_data[section].get(category, []))

idf_terms = set(idf_registry.keys())

all_terms = result_terms.union(idf_terms)
cleaned_terms = sorted(set(clean_term(t) for t in all_terms if t and t.strip()))

print(f"🔢 Total cleaned terms: {len(cleaned_terms)}")
print(f"💾 Saving to: {OUTPUT_PATH}")
with open(OUTPUT_PATH, "w") as f:
    json.dump(cleaned_terms, f, indent=2)

print("✅ Combined term list ready for embedding!")
