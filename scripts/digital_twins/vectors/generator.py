"""Stage-1 orchestration.
Runs workers to produce narratives."""

from __future__ import annotations
import os
import multiprocessing
from pathlib import Path
import random
import json
import numpy as np

from dotenv import load_dotenv
load_dotenv()

from scripts.data_loading.deterministic_vector import (
    generate_deterministic_vector, 
    initialize_attribute_indices,
    PATIENT_ATTRIBUTES_PATH
)
from scripts.data_loading.load_patient_data import load_patient_data
RECORD_EVERY = 1000

# Now deterministically parsed vectors

def generate_deterministic_vectors():
    """
    Use multiprocessing to generate vectors for all the sampled patients
    """
    # So threads don't race over the initialization of recording the attributes that are in each vector
    initialize_attribute_indices()
    with multiprocessing.Pool(processes=int(os.environ['NUM_WORKERS_NON_LLM_TASK'])) as thread_pool:
        for i, _ in enumerate(thread_pool.imap_unordered(generate_deterministic_vector, load_patient_data())):
            if (i + 1) % RECORD_EVERY == 0:
                print(f"Created {i+1} deterministic vectors...", flush=True)
                
    # Sample the vectors, respective narratives, and the vector index dictionary to state what each entry is for our sanity check
    print("Reporting vector sanity check...", flush=True)
    with open(PATIENT_ATTRIBUTES_PATH, 'r') as f:
        attributes = json.load(f)
        n_samples = 10
        all_vectors = Path(os.environ['DETERMINISTIC_VECTORS_DIR']).glob("*.npy")
        random.seed(int(os.environ['SEED']))
        sample_vector_paths = random.sample(list(all_vectors), n_samples)
        sample_ids = [v_path.stem for v_path in sample_vector_paths]
        sample_narrative_paths = [Path(os.environ['DETERMINISTIC_NARRATIVES_DIR']) / f"{id}.md" for id in sample_ids]
        
        report = json.dumps(attributes, indent=4) + '\n'
        def get_patient_dump(id, vec_path, narrative_path) -> str:
            with open(narrative_path, 'r') as f:
                narrative = f.read()
            result = id + '\n' + str(np.load(vec_path).tolist()) + '\n'\
                + narrative
            return result
        report += "\n".join([get_patient_dump(sample_ids[i], sample_vector_paths[i], sample_narrative_paths[i]) for i in range(n_samples)])
        os.makedirs(Path(os.environ['RESULTS_DIR']), exist_ok=True)
        with open(Path(os.environ['RESULTS_DIR']) / 'vector_sanity_check.txt', 'w') as f:
            f.write(report)