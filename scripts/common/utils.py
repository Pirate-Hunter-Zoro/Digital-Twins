import sys
import os
import re

# --- Dynamic sys.path adjustment ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End of sys.path adjustment ---

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