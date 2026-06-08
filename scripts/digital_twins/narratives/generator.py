"""Stage-1 orchestration.
Runs workers to produce narratives."""

from __future__ import annotations
import os
import multiprocessing
import pandas as pd
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from scripts.data_loading.deterministic_narrative import (
    extract_fields,
    build_pairings,
    set_donor_pool,
    set_pairings,
    generate_deterministic_narrative,
)
from scripts.data_loading.load_patient_data import load_patient_data

RECORD_EVERY = 1000

def generate_deterministic_narratives():
    """
    Use multiprocessing to generate narratives for all the sampled patients
    """
    sliced_jsons = list(load_patient_data())
    # Create a mapping of each patient to their Dict of extracted fields
    donor_pool = {sliced_json["patient_id"]: extract_fields(sliced_json) for sliced_json in sliced_jsons}
    pairings = build_pairings(list(donor_pool.keys()))
    set_donor_pool(donor_pool)
    set_pairings(pairings)
    narrative_lengths = {'patient_id': [], 'pre_anchor_history_days': []}
    with multiprocessing.Pool(processes=int(os.environ['NUM_WORKERS_NON_LLM_TASK'])) as thread_pool:
        for i, (patient_id, history_length) in enumerate(thread_pool.imap_unordered(generate_deterministic_narrative, sliced_jsons)):
            narrative_lengths['patient_id'].append(patient_id)
            narrative_lengths['pre_anchor_history_days'].append(history_length)
            if (i + 1) % RECORD_EVERY == 0:
                print(f"Created {i+1} deterministic narratives...", flush=True)
    pd.DataFrame(narrative_lengths).to_csv(Path(os.environ['ARTIFACTS_DIR']) / 'narrative_pre_anchor_history_days.csv')
    
    # Now that cohort is established, remove all narratives belonging to non-cohort patients
    for p in Path(os.environ['NARRATIVES_DIR']).glob("*.md"):
        if p.stem not in donor_pool.keys():
            p.unlink(missing_ok=True)