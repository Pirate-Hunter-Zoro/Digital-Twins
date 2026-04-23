"""Stage-1 orchestration.
Runs workers to produce narratives."""

import os
import multiprocessing
from pathlib import Path
import random
import json
import pandas as pd
from typing import Dict, Tuple

from dotenv import load_dotenv
load_dotenv()

from scripts.data_loading.deterministic_vector import (
    generate_deterministic_vector, 
    initialize_categorical_levels,
    CATEGORICAL_LEVELS_PATH,
    
)
from scripts.data_loading.load_patient_data import load_patient_data
RECORD_EVERY = 1000
DETERMINISTIC_DATAFRAME_PATH = Path(os.environ['DETERMINISTIC_DATAFRAME_PATH'])

def _process_patient(sliced_json: Dict) -> Tuple[str, pd.Series]:
    """Return patient's id paired with their deterministic vector

    Args:
        sliced_json (Dict): Patient's sliced json

    Returns:
        Tuple[str, pd.Series]: Paired ID with vector
    """
    return (sliced_json['patient_id'], generate_deterministic_vector(sliced_json))

def _sanity_check_report(df: pd.DataFrame):
    """Sample and perform a sanity check on a random sample of deterministic patient vectors

    Args:
        df (pd.DataFrame): All deterministic vectors
    """
    n_samples = 10
    random.seed(int(os.environ['SEED']))
    sample_ids = random.sample(list(df.index), n_samples)
    with open(CATEGORICAL_LEVELS_PATH, 'r') as f:
        category_levels = json.load(f)
        header = "\n\n\n".join([f"Shape: {df.shape}", df.dtypes.to_string(), json.dumps(category_levels, indent=4)])
        def get_patient_dump(patient_id) -> str:
            narrative_path = Path(os.environ['DETERMINISTIC_NARRATIVES_DIR']) / f"{patient_id}.md"
            with open(narrative_path, 'r') as f:
                narrative = f.read()
                row = df.loc[patient_id]
                return f"=== {patient_id} ===\n\n{row.to_string()}\n\n--- narrative ---\n{narrative}\n"
        report = header + "\n\n\n".join([get_patient_dump(id) for id in sample_ids])
        output_path = Path(os.environ['RESULTS_DIR']) / 'vector_sanity_check.txt' 
        os.makedirs(output_path.parent, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report)

def generate_deterministic_vectors():
    """
    Use multiprocessing to generate vectors for all the sampled patients
    """
    # So threads don't race over the initialization of recording the attributes that are in each vector
    initialize_categorical_levels()
    results = []
    with multiprocessing.Pool(processes=int(os.environ['NUM_WORKERS_NON_LLM_TASK'])) as thread_pool:
        for (i, (patient_id, series)) in enumerate(thread_pool.imap_unordered(_process_patient, load_patient_data())):
            results.append((patient_id, series))
            if (i + 1) % RECORD_EVERY == 0:
                print(f"Created {i+1} deterministic vectors...", flush=True)
    
    print(f"Created all deterministic vectors ({len(results)} patients passed the history-window filter).", flush=True)
    # Index is patient ID
    ids, vectors = zip(*results)
    vectors_df = pd.DataFrame(vectors, index=ids)
    vectors_df.index.name = "patient_id"
    os.makedirs(DETERMINISTIC_DATAFRAME_PATH.parent, exist_ok=True)
    vectors_df.to_parquet(DETERMINISTIC_DATAFRAME_PATH)
    print(f"Saved DataFrame shape {vectors_df.shape} to {DETERMINISTIC_DATAFRAME_PATH}", flush=True)
                
    # Sample the vectors, respective narratives, and the vector index dictionary to state what each entry is for our sanity check
    _sanity_check_report(vectors_df)