"""Stage-1 orchestration.
Runs workers to produce narratives."""

from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import multiprocessing

from .generation import generate_note
from common.data_loading.load_patient_data import load_patient_data
from common.models.vllm_client import VllmClient

RECORD_EVERY = 20

def _get_narrative(patient_json: Dict) -> Tuple[str, Optional[str]]:
    return (patient_json['patient_id'], generate_note(client, patient_json))

def init_worker():
    global client
    client = VllmClient()

def run_threadpool_stage():
    """
    Use multiprocessing to generate narratives for all the sampled patients
    """
    with multiprocessing.Pool(processes=int(os.environ['NUM_WORKERS_LLM_TASK']), initializer=init_worker) as thread_pool:
        for i, _ in enumerate(thread_pool.imap_unordered(_get_narrative, load_patient_data(years_back=int(os.environ['YEARS_BACK'])))):
            if (i + 1) % RECORD_EVERY == 0:
                print(f"Created {i+1} narratives out of {int(os.environ['NUM_PATIENTS'])}...", flush=True)