# scripts/world_2_neighbor_analysis/generate_patient_buckets.py

import os
import sys
import json
import pandas as pd
from collections import defaultdict
from pathlib import Path
from typing import Union # <-- THE MAGNIFICENT FIX!

# --- Dynamic sys.path adjustment! Gotta find our tools! ---
current_script_dir = Path(__file__).resolve().parent
project_root = current_script_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# --- Helper Functions for our Sorting Machine! ---

def get_icd10_chapter(code: str) -> Union[str, None]:
    """
    A simple little cog to determine the ICD-10 chapter from a code.
    This is a simplified map! A real one would be much, much bigger!
    """
    if not isinstance(code, str) or not code:
        return None
    
    # Let's just look at the first letter! It's a great start!
    first_char = code[0].upper()
    
    # A tiny map of letters to chapters! ISN'T IT CUTE?!
    chapter_map = {
        'A': 'Certain infectious and parasitic diseases',
        'B': 'Certain infectious and parasitic diseases',
        'C': 'Neoplasms',
        'D': 'Neoplasms',
        'E': 'Endocrine, nutritional and metabolic diseases',
        'F': 'Mental, Behavioral and Neurodevelopmental disorders',
        'G': 'Diseases of the nervous system',
        'H': 'Diseases of the eye, adnexa, ear and mastoid process',
        'I': 'Diseases of the circulatory system',
        'J': 'Diseases of the respiratory system',
        'K': 'Diseases of the digestive system',
        'L': 'Diseases of the skin and subcutaneous tissue',
        'M': 'Diseases of the musculoskeletal system and connective tissue',
        'N': 'Diseases of the genitourinary system',
        'O': 'Pregnancy, childbirth and the puerperium',
        'P': 'Certain conditions originating in the perinatal period',
        'Q': 'Congenital malformations, deformations and chromosomal abnormalities',
        'R': 'Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified',
        'S': 'Injury, poisoning and certain other consequences of external causes',
        'T': 'Injury, poisoning and certain other consequences of external causes',
        'Z': 'Factors influencing health status and contact with health services',
    }
    return chapter_map.get(first_char, 'Other')


def main():
    """
    The main engine of our Patient-Bucket-Inator! It sorts patients into
    buckets based on shared diagnoses and medication classes. FOR SCIENCE!
    """
    print("🤖✨ Firing up the Patient-Bucket-Inator 3000! ✨🤖")

    # --- Path Setup ---
    data_dir = project_root / "data"
    patient_data_path = data_dir / "all_patients_combined.json"
    output_path = data_dir / "patient_buckets_for_sanity_check.json"

    # --- Check for our ingredients! ---
    if not patient_data_path.exists():
        print(f"❌ OH NOES! I can't find the patient data at: {patient_data_path}")
        print("Please make sure you've run the process_data.py script first!")
        return

    # --- Load the magnificent patient data! ---
    print(f"📂 Loading all patient data from {patient_data_path}...")
    with open(patient_data_path, 'r') as f:
        all_patients = json.load(f)
    print(f"✅ Loaded {len(all_patients)} patients! LET'S GET SORTING!")

    # --- Prepare our empty buckets! ---
    patients_by_dx_chapter = defaultdict(set)
    # We can add medication class logic here later! One step at a time!

    # --- The Main Sorting Loop! This is my favorite part! ---
    for patient in all_patients:
        patient_id = patient['patient_id']
        
        # This set will hold all the unique chapters for this one patient!
        unique_chapters_for_patient = set()

        for encounter in patient.get('encounters', []):
            for diagnosis in encounter.get('diagnoses', []):
                dx_code = diagnosis.get('Diagnosis_1_Code')
                if dx_code:
                    chapter = get_icd10_chapter(dx_code)
                    if chapter:
                        unique_chapters_for_patient.add(chapter)
        
        # Now, for every unique chapter we found, we add the patient to that bucket!
        for chapter in unique_chapters_for_patient:
            patients_by_dx_chapter[chapter].add(patient_id)

    # --- Convert sets to lists so we can save it as a JSON! ---
    final_buckets = {
        "by_diagnosis_chapter": {
            chapter: sorted(list(p_ids)) for chapter, p_ids in patients_by_dx_chapter.items()
        }
    }
    
    # --- Save our beautiful, sorted buckets! ---
    print(f"\n💾 Saving the sorted patient buckets to: {output_path}")
    with open(output_path, "w") as f:
        json.dump(final_buckets, f, indent=2)

    print("\n🎉 AHAHAHA! The sorting is complete! All the patients are in their little data-homes!")
    print("Now we can pick from these buckets to build our perfect sanity-check dataset!")


if __name__ == "__main__":
    main()