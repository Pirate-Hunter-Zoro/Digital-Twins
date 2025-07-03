import sys
import os
import json
import random
import textwrap
import re

# --- Dynamic sys.path adjustment ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End of sys.path adjustment ---

# --- FIX APPLIED: No more circular imports! ---
from scripts.calculations.compute_nearest_neighbors import get_neighbors
from scripts.llm.query_llm import query_llm
from scripts.llm.llm_helper import get_narrative
from scripts.read_data.load_patient_data import load_patient_data
from scripts.common.utils import turn_to_sentence
# --- End of Fix ---

from scripts.config import get_global_config

# --- Global Variables ---
patient_data = None
patient_data_lookup = {}
all_response_options = None
nearest_neighbors_data = None


def setup_prompt_generation():
    """
    Initializes all necessary global data structures for generating prompts.
    """
    global patient_data, all_response_options, nearest_neighbors_data, patient_data_lookup

    print("--- Setting up prompt generation environment ---")
    patient_data = load_patient_data()
    patient_data_lookup = {p['patient_id']: p for p in patient_data}
    
    all_diagnoses = set()
    all_medications = set()
    all_treatments = set()

    print("Extracting all unique terms for prompt options...")
    for patient in patient_data:
        for visit in patient.get("visits", []):
            diagnoses_codes = [d.get("Diagnosis_Name") for d in visit.get("diagnoses", []) if d.get("Diagnosis_Name")]
            all_diagnoses.update(diagnoses_codes)
            
            medication_names = [m.get("MedSimpleGenericName") for m in visit.get("medications", []) if m.get("MedSimpleGenericName")]
            all_medications.update(medication_names)
            
            treatment_descriptions = [p.get("CPT_Procedure_Description") for p in visit.get("treatments", []) if p.get("CPT_Procedure_Description")]
            all_treatments.update(treatment_descriptions)
    
    all_response_options = {
        "diagnoses": sorted(list(all_diagnoses)),
        "medications": sorted(list(all_medications)),
        "treatments": sorted(list(all_treatments))
    }
    
    print("Pre-computing nearest neighbors for all patients...")
    nearest_neighbors_data = get_neighbors(patient_data)
    print("--- Prompt generation setup complete! ---")


def summarize_neighbor_narratives(neighbor_narratives: list[str]) -> str:
    """
    Uses the LLM to create a concise summary of neighbor data.
    """
    if not neighbor_narratives:
        return "No similar patient cases were found."

    combined_narratives = "\n\n---\n\n".join(neighbor_narratives)
    
    summary_prompt = textwrap.dedent(f"""
    You are a medical data analyst. The following are several patient case summaries who are similar to a target patient.
    Summarize the key recurring patterns, common diagnoses, and frequent treatments from these cases in 2-3 sentences.

    CASES:
    {combined_narratives}

    SUMMARY OF KEY PATTERNS:
    """)

    return query_llm(summary_prompt, max_tokens=512)


def generate_prompt(patient_dict: dict) -> str:
    """
    Generates a prompt string for the LLM based on patient history.
    """
    from scripts.config import get_global_config
    config = get_global_config()

    visits = patient_dict["visits"][: config.num_visits]
    patient_id = patient_dict["patient_id"]

    # Optional neighbors context
    prompt_sections = []

    if config.include_neighbors and hasattr(config, "nearest_neighbors_data"):
        neighbor_tuples = config.nearest_neighbors_data.get((patient_id, config.num_visits - 1), [])
        for i, (neighbor_key, _, _) in enumerate(neighbor_tuples[: config.num_neighbors]):
            neighbor_id, neighbor_idx = neighbor_key
            neighbor_dict = config.full_patient_lookup.get(neighbor_id)
            if neighbor_dict:
                neighbor_visits = neighbor_dict["visits"][: neighbor_idx + 1]
                neighbor_narrative = get_narrative(neighbor_visits)
                prompt_sections.append(f"[Neighbor {i+1}]\n{neighbor_narrative}")

    # Current patient narrative
    patient_narrative = get_narrative(visits)
    prompt_sections.append(f"[Patient History]\n{patient_narrative}")

    # NEW: Unrestricted instruction
    prompt_sections.append(
        "[Instruction]\n"
        "Based on the patient history above, predict the next clinical visit. "
        "Return the predicted diagnoses, medications, and treatments as structured JSON:"
    )

    # Example format
    prompt_sections.append("""
        [Output Format Example]
        ```json
        {
        "predicted": {
            "diagnoses": ["Example Diagnosis A", "Example Diagnosis B"],
            "medications": ["Example Medication"],
            "treatments": ["Example Treatment"]
        }
        }
    """)

    return "\n\n".join(prompt_sections).strip()


def parse_llm_response(response: str) -> dict[str, set[str]]:
    """
    Parses the structured LLM output to extract diagnoses, medications, and treatments.
    """
    next_visit = {"diagnoses": set(), "medications": set(), "treatments": set()}
    response = " ".join(response.strip().split())

    match = re.search(
        r"Diagnoses:\s*(.*?)(?:;|$)\s*Medications:\s*(.*?)(?:;|$)\s*Treatments:\s*(.*)",
        response,
        re.IGNORECASE | re.DOTALL
    )

    if match:
        diag_str, med_str, treat_str = match.groups()
        for key, raw_str in zip(["diagnoses", "medications", "treatments"], [diag_str, med_str, treat_str]):
            if raw_str and raw_str.strip().lower() not in ['none', 'n/a', '']:
                items = {item.strip() for item in raw_str.split(',') if item.strip()}
                next_visit[key].update(items)

    return next_visit


def force_valid_prediction(prompt: str, max_retries: int = 5) -> dict[str, set[str]]:
    for i in range(max_retries):
        try:
            raw_response = query_llm(prompt)
        except Exception as e:
            print(f"Error during LLM query on attempt {i+1}: {e}")
            continue

        predicted = parse_llm_response(raw_response)
        if any(predicted.values()):
            return predicted

        print(f"Warning: Empty or invalid response from LLM on try {i+1}. Retrying...")

    print(f"Error: Could not get a valid prediction after {max_retries} retries.", file=sys.stderr)
    return {"diagnoses": set(), "medications": set(), "treatments": set()}
