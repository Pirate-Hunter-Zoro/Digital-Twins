import sys
import os
import json
import random
import textwrap
import re

# --- Dynamic sys.path adjustment for module imports ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End of sys.path adjustment ---

from scripts.calculations.compute_nearest_neighbors import get_neighbors, get_visit_strings, turn_to_sentence
from scripts.llm.query_llm import query_llm
from scripts.read_data.load_patient_data import load_patient_data
from scripts.config import get_global_config

# --- Global Variables ---
# These are initialized by setup_prompt_generation()
all_patient_strings = {}
all_medications = set()
all_treatments = set()
all_diagnoses = set()
patient_data = None
all_response_options = None
nearest_neighbors_data = None


def setup_prompt_generation():
    """
    Initializes all necessary global data structures for generating prompts.
    This should be called once before starting the main processing loop.
    """
    global patient_data, all_diagnoses, all_medications, all_treatments
    global all_response_options, nearest_neighbors_data

    print("--- Setting up prompt generation environment ---")
    patient_data = load_patient_data()
    
    all_diagnoses.clear()
    all_medications.clear()
    all_treatments.clear()

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
    NEW FUNCTION: Uses the LLM to create a concise summary of neighbor data,
    preventing the main prompt from becoming too long.
    """
    if not neighbor_narratives:
        return "No similar patient cases were found."

    combined_narratives = "\n\n---\n\n".join(neighbor_narratives)
    
    summary_prompt = textwrap.dedent(f"""
    You are a medical data analyst. The following are several patient case summaries who are similar to a target patient.
    Summarize the key recurring patterns, common diagnoses, and frequent treatments from these cases in 2-3 sentences.
    Focus on what might be most relevant for predicting the next step for a similar patient.

    CASES:
    {combined_narratives}

    SUMMARY OF KEY PATTERNS:
    """)

    # We use a shorter max_tokens for the summary to keep it concise.
    return query_llm(summary_prompt, max_tokens=512)


def generate_prompt(patient: dict) -> str:
    """
    Generate a prompt to predict the patient's (n+1)st visit.
    This version now summarizes neighbor data to prevent excessive prompt length.
    """
    global nearest_neighbors_data, all_response_options

    patient_id = patient["patient_id"]
    num_visits_for_history = get_global_config().num_visits - 1
    
    # Define the key for this specific patient and visit index
    patient_key = (patient_id, num_visits_for_history)

    # --- Generate Patient History Section ---
    history_visits = patient["visits"][:num_visits_for_history]
    if not history_visits:
        history_section = "No historical visits available."
    else:
        history_section = "\n".join(f"Visit {i}: {turn_to_sentence(visit)}" for i, visit in enumerate(history_visits))

    # --- Generate Neighbor Summary Section ---
    relevant_neighbors = nearest_neighbors_data.get(patient_key, [])
    neighbor_narratives = [
        get_narrative(patient_data_lookup[neighbor_id]["visits"][:neighbor_vidx + 1])
        for (neighbor_id, neighbor_vidx), _, _ in relevant_neighbors[:get_global_config().num_neighbors]
        if (patient_data_lookup := {p['patient_id']: p for p in patient_data}).get(neighbor_id)
    ]
    
    # Use our new summarization function!
    neighbor_summary_section = summarize_neighbor_narratives(neighbor_narratives)

    # --- Random Options for LLM (to guide the output format) ---
    random_diagnoses = ', '.join(random.sample(all_response_options["diagnoses"], min(len(all_response_options["diagnoses"]), 3)))
    random_medications = ', '.join(random.sample(all_response_options["medications"], min(len(all_response_options["medications"]), 3)))
    random_treatments = ', '.join(random.sample(all_response_options["treatments"], min(len(all_response_options["treatments"]), 3)))

    # --- Assemble the Final Prompt ---
    prompt = textwrap.dedent(f"""
    Based on the patient's history and a summary of similar cases, predict the diagnoses, medications, and treatments for the patient's next visit.

    PATIENT HISTORY:
    {history_section}

    SUMMARY OF SIMILAR CASES:
    {neighbor_summary_section}

    You MUST choose from the following valid options. Diagnoses are ICD-10 codes.

    Diagnoses Options: {random_diagnoses}
    Medications Options: {random_medications}
    Treatments Options: {random_treatments}

    Respond with ONLY the following format and no additional text or explanation:
    Diagnoses: <comma-separated list>; Medications: <comma-separated list>; Treatments: <comma-separated list>

    Example:
    Diagnoses: E11.9, I10, Z79.4; Medications: Metformin, Lisinopril; Treatments: Physical therapy referral
    
    ### BEGIN RESPONSE
    """)
    
    return prompt


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
                # Split and clean up each item
                items = {item.strip() for item in raw_str.split(',') if item.strip()}
                next_visit[key].update(items)

    return next_visit


def force_valid_prediction(prompt: str, max_retries: int = 5) -> dict[str, set[str]]:
    """
    Queries the LLM and retries if the response is empty, ensuring a validly structured
    prediction is returned.
    """
    for i in range(max_retries):
        raw_response = query_llm(prompt)
        predicted = parse_llm_response(raw_response)
        
        # Check if we got any valid data in any category
        if any(predicted.values()):
            return predicted
        
        print(f"Warning: Empty or invalid response from LLM on try {i+1}. Retrying...")
    
    # If all retries fail, return an empty structure
    print(f"Error: Could not get a valid prediction after {max_retries} retries.", file=sys.stderr)
    return {"diagnoses": set(), "medications": set(), "treatments": set()}

