"""Stage-1 orchestration.
Runs workers to produce narratives."""

from __future__ import annotations
import os
from typing import Dict, Tuple
import multiprocessing

from scripts.data_loading.deterministic_narrative import generate_deterministic_narrative
from scripts.data_loading.load_patient_data import load_patient_data
RECORD_EVERY = 1000

# Now deterministically parsed narratives

def _get_deterministic_narrative(sliced_and_unsliced_json: Tuple[Dict]) -> bool:
    sliced_patient_json, unsliced_patient_json = sliced_and_unsliced_json
    generate_deterministic_narrative(sliced_json=sliced_patient_json, unsliced_json=unsliced_patient_json)
    # To signify completion
    return True

def generate_deterministic_narratives():
    """
    Use multiprocessing to generate narratives for all the sampled patients
    """
    with multiprocessing.Pool(processes=int(os.environ['NUM_WORKERS_NON_LLM_TASK'])) as thread_pool:
        for i, _ in enumerate(thread_pool.imap_unordered(_get_deterministic_narrative, load_patient_data())):
            if (i + 1) % RECORD_EVERY == 0:
                print(f"Created {i+1} deterministic narratives...", flush=True)