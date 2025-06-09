import json
from compute_nearest_neighbors import get_neighbors, get_visit_strings, turn_to_sentence
from generate_patients import load_patient_data
from query_llm import query_llm
import random
import textwrap

patient_data = load_patient_data()
all_medications = set()
all_treatments = set()
all_diagnoses = set()
for patient in patient_data:
    for visit in patient["visits"]:
        all_medications.update(visit.get("medications", []))
        all_treatments.update(visit.get("treatments", []))
        all_diagnoses.update(visit.get("diagnoses", []))

all_response_options = {
    "diagnoses": sorted(all_diagnoses),
    "medications": sorted(all_medications),
    "treatments": sorted(all_treatments)
}

all_patient_strings = get_visit_strings(patient_data, window_len=5)

def generate_prompt(patient: dict, n: int, use_synthetic_data: bool=False, num_neighbors: int=5, vectorizer: str="sentence_transformer", distance_metric: str="cosine") -> str:
    """
    Generate a prompt to get the patient's (n+1)st visit (of index n).
    """
    nearest_neighbors = get_neighbors(patient_data, use_synthetic_data=use_synthetic_data, vectorizer=vectorizer, distance_metric=distance_metric)

    try:
        with open(f"{'synthetic_data' if use_synthetic_data else 'real_data'}/all_prompts_{vectorizer}_{distance_metric}.json", "r") as f:
            all_prompts = json.load(f)
    except:
        all_prompts = {}

    patient_id = patient["patient_id"]
    window_size_used = n  # We use the latest visit *up to* the prediction point
    key = f"{patient_id}_{window_size_used}"
    if key not in all_prompts.keys():
        # Then we need to generate the prompt

        history_section = "\n".join(
            turn_to_sentence(visit) for visit in patient["visits"][:n]
        )

        relevant_neighbors = nearest_neighbors.get((patient_id, window_size_used), [])
        neighbor_section = "\n".join(
            all_patient_strings[neighbor_key_score[0]]
            for neighbor_key_score in relevant_neighbors[:min(len(relevant_neighbors), num_neighbors+1)]
        )

        random_diagnoses = ', '.join(random.sample(sorted(all_diagnoses), min(len(all_diagnoses), 3)))
        random_medications = ', '.join(random.sample(sorted(all_medications), min(len(all_medications), 3)))
        random_treatments = ', '.join(random.sample(sorted(all_treatments), min(len(all_treatments), 3)))
        # Create the prompt
        prompt = textwrap.dedent(f"""
        Here is a list of all the patient's first {n} visits:

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
        with open("all_prompts.json", "w") as f:
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