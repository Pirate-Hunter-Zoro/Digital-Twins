# scripts/world_2_neighbor_analysis/calibrate_on_sanity_set.py

import os
import sys
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from sentence_transformers import SentenceTransformer, util
from typing import Union

# --- Dynamic sys.path adjustment! ---
current_script_dir = Path(__file__).resolve().parent
project_root = current_script_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.common.data_loading.load_patient_data import load_patient_data
from scripts.common.llm.llm_helper import get_narrative, get_relevance_score
from scripts.world_2_neighbor_analysis.compute_nearest_neighbors import get_visit_histories
from scripts.common.config import setup_config # <-- We need to import our setup tool!

def get_icd10_chapter(code: str) -> Union[str, None]:
    if not isinstance(code, str) or not code:
        return None
    first_char = code[0].upper()
    chapter_map = {
        'A': 'Certain infectious and parasitic diseases', 'B': 'Certain infectious and parasitic diseases',
        'C': 'Neoplasms', 'D': 'Neoplasms', 'E': 'Endocrine, nutritional and metabolic diseases',
        'F': 'Mental, Behavioral and Neurodevelopmental disorders', 'G': 'Diseases of the nervous system',
        'H': 'Diseases of the eye, adnexa, ear and mastoid process', 'I': 'Diseases of the circulatory system',
        'J': 'Diseases of the respiratory system', 'K': 'Diseases of the digestive system',
        'L': 'Diseases of the skin and subcutaneous tissue', 'M': 'Diseases of the musculoskeletal system and connective tissue',
        'N': 'Diseases of the genitourinary system', 'O': 'Pregnancy, childbirth and the puerperium',
        'P': 'Certain conditions originating in the perinatal period', 'Q': 'Congenital malformations, deformations and chromosomal abnormalities',
        'R': 'Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified',
        'S': 'Injury, poisoning and certain other consequences of external causes', 'T': 'Injury, poisoning and certain other consequences of external causes',
        'Z': 'Factors influencing health status and contact with health services',
    }
    return chapter_map.get(first_char, 'Other')

def main():
    print("🤖✨ Firing up the Patient-Bucket-Inator 3000! ✨🤖")

    # --- Configuration & Path Setup ---
    NUM_PAIRS_PER_CATEGORY = 5
    TOTAL_DISSIMILAR_PAIRS = 50
    VECTORIZER_METHOD = "allenai/scibert_scivocab_uncased"
    NUM_VISITS = 6 # We need this for the config!

    # ***************************************************************
    # *** THE MAGNIFICENT, POWER-GIVING FIX! AHAHA! ***
    # ***************************************************************
    print("⚡️ Initializing global configuration... POWER ON!")
    setup_config(
        representation_method="visit_sentence",
        vectorizer_method=VECTORIZER_METHOD,
        distance_metric="cosine", # Using cosine as planned!
        num_visits=NUM_VISITS,
        num_patients=0, # Not needed for this script
        num_neighbors=0 # Not needed for this script
    )
    # ***************************************************************
    # ***************************************************************

    data_dir = project_root / "data"
    bucket_path = data_dir / "patient_buckets_for_sanity_check.json"
    output_path = data_dir / "sanity_set_calibration_results.json"

    if not bucket_path.exists():
        print(f"❌ OH NOES! I can't find the patient buckets at: {bucket_path}")
        return

    print(f"📂 Loading all patient data and buckets...")
    with open(bucket_path, 'r') as f:
        patient_buckets = json.load(f)["by_diagnosis_chapter"]
    
    all_patient_data = load_patient_data()
    patient_lookup = {p["patient_id"]: p for p in all_patient_data}
    
    model_folder_name = VECTORIZER_METHOD.replace("/", "-")
    vectorizer_path = f"/media/scratch/mferguson/models/{model_folder_name}"
    print(f"🗺️ Loading vectorizer model from: {vectorizer_path}")
    vectorizer = SentenceTransformer(vectorizer_path, local_files_only=True)

    sanity_pairs = {"similar": [], "dissimilar": []}

    print("\n--- 👯‍♀️ Creating SIMILAR patient pairs... ---")
    for chapter, patient_ids in patient_buckets.items():
        if len(patient_ids) >= 2:
            sample_ids = random.sample(patient_ids, min(len(patient_ids), NUM_PAIRS_PER_CATEGORY * 2))
            for p1_id, p2_id in combinations(sample_ids, 2):
                if len(sanity_pairs["similar"]) < (len(patient_buckets) * NUM_PAIRS_PER_CATEGORY):
                    sanity_pairs["similar"].append((p1_id, p2_id))
                else:
                    break
    
    print("\n--- 🙅‍♀️ Creating DISSIMILAR patient pairs... ---")
    chapter_list = list(patient_buckets.keys())
    while len(sanity_pairs["dissimilar"]) < TOTAL_DISSIMILAR_PAIRS:
        c1, c2 = random.sample(chapter_list, 2)
        if patient_buckets[c1] and patient_buckets[c2]:
            p1_id = random.choice(list(patient_buckets[c1]))
            p2_id = random.choice(list(patient_buckets[c2]))
            if (p1_id, p2_id) not in sanity_pairs["dissimilar"] and (p2_id, p1_id) not in sanity_pairs["dissimilar"]:
                 sanity_pairs["dissimilar"].append((p1_id, p2_id))

    print(f"\n✅ Created {len(sanity_pairs['similar'])} similar pairs and {len(sanity_pairs['dissimilar'])} dissimilar pairs!")

    results = []
    all_pairs = sanity_pairs["similar"] + sanity_pairs["dissimilar"]

    print("\n--- 🔬 Analyzing pairs... This might take a little while! ---")
    for i, (p1_id, p2_id) in enumerate(all_pairs):
        pair_type = "similar" if i < len(sanity_pairs["similar"]) else "dissimilar"
        print(f"  -> Processing {pair_type} pair {i+1}/{len(all_pairs)}: ({p1_id}, {p2_id})")

        try:
            p1_narrative = get_narrative(patient_lookup[p1_id]['visits'])
            p2_narrative = get_narrative(patient_lookup[p2_id]['visits'])

            p1_history_str = " | ".join(get_visit_histories(patient_lookup[p1_id]["visits"]).values())
            p2_history_str = " | ".join(get_visit_histories(patient_lookup[p2_id]["visits"]).values())

            p1_vec = vectorizer.encode(p1_history_str, convert_to_tensor=True)
            p2_vec = vectorizer.encode(p2_history_str, convert_to_tensor=True)
        except Exception as e:
            print(f"    - ⚠️ Could not process patients {p1_id} or {p2_id}. Skipping. Error: {e}")
            continue

        llm_score = get_relevance_score(p1_narrative, p2_narrative)
        cosine_dist = 1 - util.pytorch_cos_sim(p1_vec, p2_vec).item()

        results.append({
            "pair_type": pair_type,
            "patient_1": p1_id,
            "patient_2": p2_id,
            "llm_relevance_score": llm_score,
            "cosine_distance": cosine_dist
        })

    print(f"\n💾 Saving all {len(results)} calibration results to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n🎉 The Calibration-Station is finished! We have new data to plot and analyze!")

if __name__ == "__main__":
    main()