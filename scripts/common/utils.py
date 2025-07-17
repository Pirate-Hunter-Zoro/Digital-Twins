# scripts/common/utils.py (Now with a safety net!)

import sys
import os
import re

# --- Dynamic sys.path adjustment ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End of sys.path adjustment ---

from scripts.common.config import get_global_config

def turn_to_sentence(encounter_obj: dict) -> str:
    """
    Convert an encounter dictionary into a human-readable sentence.
    This is now a central utility function.
    """
    sentences = []
    
    # Diagnoses
    if encounter_obj.get("diagnoses"):
        diagnoses_names = [
            diag.get("Diagnosis_Name")
            for diag in encounter_obj["diagnoses"]
            if diag.get("Diagnosis_Name")
        ]
        if diagnoses_names:
            sentences.append("Diagnoses: " + ", ".join(diagnoses_names))

    # Medications
    if encounter_obj.get("medications"):
        medication_names = [
            med.get("MedSimpleGenericName")
            for med in encounter_obj["medications"]
            if med.get("MedSimpleGenericName")
        ]
        if medication_names:
            sentences.append("Medications: " + ", ".join(medication_names))
    
    # Treatments (Procedures)
    if encounter_obj.get("treatments"):
        treatment_descriptions = [
            proc.get("CPT_Procedure_Description")
            for proc in encounter_obj["treatments"]
            if proc.get("CPT_Procedure_Description")
        ]
        if treatment_descriptions:
            sentences.append("Treatments: " + ", ".join(treatment_descriptions))

    return "; ".join(sentences) if sentences else "No information recorded for this visit."

def clean_term(term: str) -> str:
    """
    Cleans a medical term string for embedding.
    This is our magnificent, centralized term-scrubber!
    """
    # --- THE MAGNIFICENT FIX! ---
    # If the term is None, we just return an empty string and stop!
    if term is None:
        return ""
    # ---------------------------

    term = term.lower().strip()
    term = term.replace('"', '').replace("'", '')
    term = re.sub(r"^\\+|\\+$", "", term)
    term = re.sub(r"\([^)]*hcc[^)]*\)", "", term)
    term = re.sub(r"\(cms[^)]*\)", "", term)
    term = re.sub(r"\b\(\d{3}(\.\d+)?\)", "", term)
    term = re.sub(r",+", "", term)

    blacklist = [
        "initial encounter", "unspecified", "nos", "nec",
        "<none>", "<None>", ";", ":",
        r"at \d+ oclock position", r"during pregnancy.*",
        r"due to.*", r"with late onset", r"with dysplasia", r"without dysplasia"
    ]
    for noise in blacklist:
        term = re.sub(noise, "", term)

    return re.sub(r"\s+", " ", term).strip()

def get_visit_term_lists(patient_visits: list[dict]) -> dict[int, list[str]]:
    """
    Now, instead of one long sentence, this gives us a list of all the cleaned
    medical terms for each visit! PERFECT for our new encoder!
    A list of lists - each inner list is a vist where all the observed terms are shoved in.
    """
    visit_histories = {}
    config = get_global_config()
    history_window_length = config.num_visits

    for i in range(len(patient_visits) - history_window_length + 1):
        relevant_visits = patient_visits[i : i + history_window_length]
        end_idx = i + history_window_length - 1

        # This will now be a list of lists!
        history_term_lists = []
        for visit in relevant_visits:
            visit_terms = []
            visit_terms.extend([clean_term(d.get("Diagnosis_Name", "")) for d in visit.get("diagnoses", [])])
            visit_terms.extend([clean_term(m.get("MedSimpleGenericName", "")) for m in visit.get("medications", [])])
            visit_terms.extend([clean_term(p.get("CPT_Procedure_Description", "")) for p in visit.get("treatments", [])])
            history_term_lists.append([term for term in visit_terms if term])
        
        visit_histories[end_idx] = history_term_lists
        
    return visit_histories