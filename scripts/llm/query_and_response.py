import sys
import os

# --- Dynamic sys.path adjustment for module imports ---
# Get the absolute path to the directory containing the current script
current_script_dir = os.path.dirname(os.path.abspath(__file__))

# Calculate the project root.
# This assumes your 'config.py' (or other root-level modules like 'main.py')
# is located two directories up from where this script resides.
# Example: If this script is in 'project_root/scripts/read_data/your_script.py'
# then '..' takes you to 'project_root/scripts/'
# and '..', '..' takes you to 'project_root/'
project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..'))

# Add the project root to sys.path if it's not already there.
# Inserting at index 0 makes it the first place Python looks for modules,
# so 'import config' will find it quickly.
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- End of sys.path adjustment ---

import json
from scripts.calculations.compute_nearest_neighbors import get_neighbors, get_visit_strings, turn_to_sentence
from scripts.llm.query_llm import query_llm
from scripts.read_data.load_patient_data import load_patient_data
import random
import textwrap
from scripts.config import get_global_config

all_patient_strings = {}
all_medications = set()
all_treatments = set()
all_diagnoses = set()
patient_data = None
all_response_options = None

def setup_prompt_generation():
    global all_patient_strings
    global all_medications
    global all_treatments
    global all_diagnoses
    global patient_data
    global all_response_options

    patient_data = load_patient_data() # This now loads real data in 'visits' format
    
    # Initialize sets to ensure they're clean on setup
    all_medications.clear()
    all_treatments.clear()
    all_diagnoses.clear()

    for patient in patient_data:
        # Loop through 'visits' (which are now the real data encounters)
        for visit in patient.get("visits", []): # Use .get() for safety
            all_medications.update(visit.get("medications", []))
            all_treatments.update(visit.get("treatments", [])) # Assuming 'treatments' key now exists due to load_patient_data mapping
            all_diagnoses.update(visit.get("diagnoses", []))

    all_response_options = {
        "diagnoses": sorted(all_diagnoses),
        "medications": sorted(all_medications),
        "treatments": sorted(all_treatments)
    }

    # all_patient_strings needs to be generated from the new patient_data format
    # get_visit_strings needs to be adapted for this.
    # It's in compute_nearest_neighbors.py, so ensure it uses the 'encounter_obj' passed to turn_to_sentence.
    from scripts.calculations.compute_nearest_neighbors import get_visit_strings
    all_patient_strings = get_visit_strings(patient_data)
    
def generate_prompt(patient: dict) -> str:
    """
    Generate a prompt to get the patient's (n+1)st visit (of index n).
    """
    global all_patient_strings
    
    nearest_neighbors = get_neighbors(patient_data)

    try:
        with open(f"real_data/all_prompts_{get_global_config().vectorizer_method}_{get_global_config().distance_metric}.json", "r") as f:
            all_prompts = json.load(f)
    except:
        all_prompts = {}

    patient_id = patient["patient_id"]
    
    window_size_used = get_global_config().num_visits-1  # We use the latest visit *up to* the prediction point
    key = f"{patient_id}_{window_size_used}"
    if key not in all_prompts.keys():
        # Then we need to generate the prompt

        if len(patient["visits"]) < get_global_config().num_visits:
            # This patient doesn't have enough history visits for prediction based on config.
            # You might want to skip them or handle this case.
            # For now, let's assume valid patients have at least num_visits visits.
            # If it's the last visit being predicted, it's the one at num_visits-1.
            # If config.num_visits=5, we use visits 0,1,2,3 for history and predict visit 4.

            # Correctly get the history, taking the latest 'num_visits-1' encounters/visits
            history_visits = patient["visits"][:get_global_config().num_visits-1]
            
            # If for some reason history_visits is empty, or too short
            if not history_visits:
                history_section = "No historical visits available."
            else:
                history_section = "\n".join(turn_to_sentence(visit) for visit in history_visits)
        else:
            history_section = "\n".join(
                turn_to_sentence(visit) for visit in patient["visits"][:get_global_config().num_visits-1]
            )

        relevant_neighbors = nearest_neighbors.get((patient_id, window_size_used), [])
        neighbor_section = "\n".join(
            all_patient_strings[neighbor_key_score[0]]
            for neighbor_key_score in relevant_neighbors[:min(len(relevant_neighbors), get_global_config().num_neighbors+1)]
        )

        # If config.num_visits=5, we use visits 0,1,2,3 for history and predict visit 4.

        # Correctly get the history, taking the latest 'num_visits-1' encounters/visits
        history_visits = patient["visits"][:get_global_config().num_visits-1]
        
        # If for some reason history_visits is empty, or too short
        if not history_visits:
            history_section = "No historical visits available."
        else:
            history_section = "\n".join(turn_to_sentence(visit) for visit in history_visits)

        # --- Random options for LLM ---
        # These sets are populated by setup_prompt_generation globally
        # If all_diagnoses, all_medications, all_treatments are empty (e.g. no data), random.sample will error.
        # Add a check for non-empty set before sampling.
        random_diagnoses = ', '.join(random.sample(sorted(list(all_diagnoses)), min(len(all_diagnoses), 3))) if all_diagnoses else "None"
        random_medications = ', '.join(random.sample(sorted(list(all_medications)), min(len(all_medications), 3))) if all_medications else "None"
        random_treatments = ', '.join(random.sample(sorted(list(all_treatments)), min(len(all_treatments), 3))) if all_treatments else "None"
        # Create the prompt
        prompt = textwrap.dedent(f"""
        Here is a list of all the patient's first {get_global_config().num_visits-1} visits:

        {history_section}

        Here are the most similar visit sequences of the same length from other patients, in descending order of closeness:

        {neighbor_section}

        Based on the similar patients and this patient's history, predict the patient's next visit.

        You MUST choose from the following valid options only (note that Diagnoses are ICD-10 codes):

        Diagnoses: {random_diagnoses}
        Medications: {random_medications}
        Treatments: {random_treatments}

        Respond with ONLY the following format and no additional text or explanation:
        Diagnoses: <comma-separated ICD codes>; Medications: <comma-separated medication names>; Treatments: <comma-separated treatment descriptions>

        Example:
        Diagnoses: E11, F33.1, J45.909; Medications: Metformin, Insulin, Lisinopril; Treatments: Referral to endocrinology, Sleep hygiene education, Physical therapy referral

        You are a medical assistant. Do not explain your reasoning. Output only the response in the specified format. Do not include any additional text.
        ### BEGIN RESPONSE\n
        """)

        # Save the prompt for future use
        all_prompts[key] = prompt
        with open(f"real_data/all_prompts_{get_global_config().vectorizer_method}_{get_global_config().distance_metric}.json", "w") as f:
            json.dump(all_prompts, f, indent=4)
            
    return all_prompts[key]

import re

def parse_llm_response(response: str) -> dict[str, set[str]]:
    """
    Parses LLM output that may include messy formatting or merged sections.
    Extracts diagnoses, medications, and treatments from a string.
    """
    next_visit = {"diagnoses": set(), "medications": set(), "treatments": set()}

    # Normalize formatting: remove excess whitespace and collapse to one line
    response = " ".join(response.strip().split())

    # Use regex to extract each labeled section
    match = re.search(
        r"Diagnoses:\s*(.*?);?\s*Medications:\s*(.*?);?\s*Treatments:\s*(.*)",
        response,
        re.IGNORECASE
    )

    if match:
        diag_str, med_str, treat_str = match.groups()

        for key, raw_str in zip(["diagnoses", "medications", "treatments"], [diag_str, med_str, treat_str]):
            canonical_map = {val.lower(): val for val in all_response_options[key]}
            items = [
                canonical_map[x.strip().lower()]
                for x in raw_str.split(",")
                if x.strip().lower() in canonical_map
            ]
            next_visit[key].update(items)

    return next_visit

def force_valid_prediction(prompt: str, max_retries: int = 5) -> dict[str, set[str]]:
    """
    Force the LLM to return a valid prediction by appending a specific instruction.
    """
    predicted = parse_llm_response(query_llm(prompt))
    for _ in range(max_retries):
        if not any(len(predicted[k]) for k in ["diagnoses", "medications", "treatments"]):
            predicted = parse_llm_response(query_llm(prompt))
        else:
            break

    return predicted