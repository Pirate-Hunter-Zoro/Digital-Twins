"""Stage-1 orchestration.
Runs workers to produce narratives."""

from __future__ import annotations
import os
from typing import Dict
import multiprocessing

from .llm_narrative import generate_llm_narrative
from .deterministic_narrative import generate_deterministic_narrative
from common.data_loading.load_patient_data import load_patient_data
from common.models.vllm_client import VllmClient

RECORD_EVERY = 20

def _get_llm_narrative(sliced_patient_json: Dict, unsliced_patient_json: Dict) -> bool:
    generate_llm_narrative(client=client, patient_json=sliced_patient_json)
    # To signify completion
    return True

def init_worker_llm():
    global client
    client = VllmClient()

def generate_llm_narratives():
    """
    Use multiprocessing to generate narratives for all the sampled patients
    """
    with multiprocessing.Pool(processes=int(os.environ['NUM_WORKERS_LLM_TASK']), initializer=init_worker_llm) as thread_pool:
        for i, _ in enumerate(thread_pool.imap_unordered(_get_llm_narrative, load_patient_data())):
            if (i + 1) % RECORD_EVERY == 0:
                print(f"Created {i+1} llm narratives out of {int(os.environ['NUM_PATIENTS'])}...", flush=True)

# Now deterministically parsed narratives

def _get_deterministic_narrative(sliced_patient_json: Dict, unsliced_patient_json: Dict) -> bool:
    generate_deterministic_narrative(sliced_json=sliced_patient_json, unsliced_json=unsliced_patient_json)
    # To signify completion
    return True

def generate_deterministic_narratives():
    """
    Use multiprocessing to generate narratives for all the sampled patients
    """
    with multiprocessing.Pool(processes=int(os.environ['NUM_WORKERS_NON_LLM_TASK']), initializer=init_worker_llm) as thread_pool:
        for i, _ in enumerate(thread_pool.imap_unordered(_get_deterministic_narrative, load_patient_data())):
            if (i + 1) % RECORD_EVERY == 0:
                print(f"Created {i+1} deterministic narratives out of {int(os.environ['NUM_PATIENTS'])}...", flush=True)