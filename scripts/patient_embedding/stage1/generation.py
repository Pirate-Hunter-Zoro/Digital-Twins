"""LLM call wrapper for narrative generation.
Formats prompts and calls llama.cpp chat; raises on empty content."""

# scripts/patient_embedding/stage1/generation.py

from __future__ import annotations

import json, math
from typing import List
from pathlib import Path
import os
if __name__=="__main__":
    from dotenv import load_dotenv
    load_dotenv()
    from scripts.common.models.vllm_client import VllmClient  
    from scripts.patient_embedding.shared.prompts import PromptLoader
else:
    from common.models.vllm_client import VllmClient
    from patient_embedding.shared.prompts import PromptLoader
from typing import Iterator
import copy

prompt_loader = PromptLoader()
DEBUG = False
DEBUG_FILE = Path("test_data/debug.txt")

# ---------------- public APIs ----------------

def _yield_patient_json_in_batches(full_patient_json: dict, chunk_size: int) -> Iterator[dict]:
    """
    Yield a list of strings which is the full json partitioned into manageable chunks.
    """
    base_dict = {
        "patient_id": full_patient_json["patient_id"],
        "demographics": full_patient_json["demographics"],
        "anchor_date": full_patient_json["anchor_date"],
        "active_medications": [],
        "encounters": []
    }
    
    current_chunk_medications = []
    current_dict = None
    # Chunk from the base dictionary for all medications
    for medication in full_patient_json["active_medications"]:
        current_chunk_medications.append(medication)
        json_so_far = copy.deepcopy(base_dict)
        json_so_far["active_medications"] = current_chunk_medications
        if len(str(json_so_far)) > chunk_size:
            # Past the limit
            yield current_dict
            current_chunk_medications = [medication]
        else:
            current_dict = json_so_far
    if len(current_chunk_medications) > 0:
        last_dict = copy.deepcopy(base_dict)
        last_dict['active_medications'] = current_chunk_medications
        yield last_dict
    
    # Chunk from the base dictionary for all encounters
    current_dict = None
    current_chunk_encounters = []
    for encounter in full_patient_json["encounters"]:
        current_chunk_encounters.append(encounter)
        json_so_far = copy.deepcopy(base_dict)
        json_so_far["encounters"] = current_chunk_encounters
        if len(str(json_so_far)) > chunk_size:
            # Past the limit
            yield current_dict
            current_chunk_encounters = [encounter]
        else:
            current_dict = json_so_far
    if len(current_chunk_encounters) > 0:
        last_dict = copy.deepcopy(base_dict)
        last_dict['encounters'] = current_chunk_encounters
        yield last_dict

def generate_note(
    client: VllmClient,
    patient_json: dict,
):
    """Rolling narrative generation and storing."""
    
    # Check to see if narrative already exists
    patient_id = patient_json['patient_id']
    narrative_save_path = Path(os.environ['NARRATIVES_DIR']) / f"{patient_id}.md"
    if narrative_save_path.exists():
        return
    
    system_text_initial = prompt_loader.get_narrative_system_initial()
    system_text_extraction = prompt_loader.get_narrative_system_extraction()
    
    first_chunk = True
    base_narrative = None
    if DEBUG:
        sections = []
    cumulative_visits = []
    for chunked_json in _yield_patient_json_in_batches(full_patient_json=patient_json, chunk_size=int(os.environ['PATIENT_JSON_CHUNK_SIZE'])):
        if not chunked_json:
            raise ValueError(f"Invalid json chunk yielded for patient {patient_id}: {chunked_json}")
        if first_chunk:
            first_chunk = False
            user_text  = prompt_loader.render_narrative_user_initial(patient_json=chunked_json)
            messages = [
                {"role": "system", "content": system_text_initial},
                {"role": "user", "content": user_text},
            ]
            resp = client.chat(
                messages,
            )
            if not resp or not isinstance(resp, str):
                raise ValueError(f"Empty content after using the following prompts:\n\nSystem: {system_text_initial}\n\nUser: {user_text}")
            else:
                base_narrative = resp
                if DEBUG:
                    sections.append(base_narrative)
        else:
            user_text  = prompt_loader.render_narrative_user_extraction(patient_json=chunked_json)
            messages = [
                {"role": "system", "content": system_text_extraction},
                {"role": "user", "content": user_text},
            ]
            resp = client.chat(
                messages,
            )
            if not resp or not isinstance(resp, str):
                raise ValueError(f"Empty content after using the following prompts:\n\nSystem: {system_text_extraction}\n\nUser: {user_text}")
            else:
                cumulative_visits.append(resp)
                if DEBUG:
                    sections.append(resp)
    
    system_text_finalization = prompt_loader.get_narrative_system_finalization()
    user_text  = prompt_loader.render_narrative_user_finalization(base_narrative=base_narrative, additional_visits=cumulative_visits)
    messages = [
        {"role": "system", "content": system_text_finalization},
        {"role": "user", "content": user_text},
    ]
    resp = client.chat(
        messages,
    )
    if not resp or not isinstance(resp, str):
        raise ValueError(f"Empty content after using the following prompts:\n\nSystem: {system_text_finalization}\n\nUser: {user_text}")
    else:
        if DEBUG:
            sections.append(resp)
            with open(DEBUG_FILE, 'w') as f:
                f.write("\n\n\n\n=========================================================\n\n".join(sections))
        with open(narrative_save_path, 'w') as f:
            f.write(resp)

if __name__=="__main__":
    
    client = VllmClient()
    id = "40063E1E7CFA9D672D3C62EFB573443F"
    TEST_PATIENT_FILE = f"test_data/patient_{id}.json"

    # Open the file and then load its contents
    with open(TEST_PATIENT_FILE, 'r') as f:
        patient_json = json.load(f)
        with open(TEST_PATIENT_FILE, 'w') as f:
            json.dump(patient_json, f, indent=4)
    
    generate_note(client, patient_json)
    with open(Path(os.environ['NARRATIVES_DIR']) / f"{id}.md", 'r') as f:
        narrative = "\n".join(f.readlines())
        with open(f"test_data/patient_{id}.md", "w") as f:
            f.write(narrative)