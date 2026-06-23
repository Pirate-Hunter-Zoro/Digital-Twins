import pandas as pd
from pathlib import Path
import os
import multiprocessing
import numpy as np

from scripts.pipeline.predictions.trd_predictor import TRDPredictor
from scripts.pipeline.neighbors.neighbor_scheme import RELEVANT_NEIGHBOR_SCHEMES
from scripts.pipeline.predictions.create_train_test_split import create_train_test_split

from dotenv import load_dotenv
load_dotenv()

RESULTS_DIR = Path(os.environ['RESULTS_DIR'])
os.makedirs(RESULTS_DIR, exist_ok=True)
predictor = None
test_ids = create_train_test_split()[1]

def init_worker():
    global predictor # This will be created for each worker
    np.random.seed(int(os.environ['SEED']))
    predictor = TRDPredictor(exclude_ids=test_ids)

def evaluate_patient(patient_id: str):
    """Obtain all the patient's neighborhood prediction information and save it

    Args:
        patient_id (str): Id of the patient of interest
    """
    global predictor
    results = []
    for scheme in RELEVANT_NEIGHBOR_SCHEMES:
        results.extend(predictor.construct_neighborhood_data(index_id=patient_id, scheme=scheme))
    return results

def run():
    """
    Single worker process to run TRD prediction on a sub-sample of patients
    """
    
    # Establish deterministic order over the different Slurm workers
    sorted_test_ids = sorted(list(test_ids))
    
    # Now break up the patient sample to different workers
    slurm_task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    slurm_task_count = int(os.environ['SLURM_ARRAY_TASK_COUNT'])
    chunk_size = len(test_ids) // slurm_task_count
    num_with_extra = len(test_ids) % slurm_task_count
    
    # Grab this worker's chunk of sorted_test_ids
    # start_index inclusive, end_index exclusive
    if slurm_task_id < num_with_extra:
        start_index = slurm_task_id*(chunk_size+1)
        end_index = start_index + chunk_size+1
    else:
        start_index = num_with_extra*(chunk_size+1) + (slurm_task_id - num_with_extra)*chunk_size
        end_index = start_index + chunk_size
    
    chunk_ids = sorted_test_ids[start_index:end_index]
    results = []
    with multiprocessing.Pool(processes=int(os.environ['NUM_WORKERS_LLM_TASK']), initializer=init_worker) as pool:
        done = 0
        for res in pool.imap_unordered(evaluate_patient, chunk_ids):
            results.extend(res)
            done+=1
            print(f"Completed {done} neighborhood constructions out of {len(chunk_ids)} for and SLURM id {slurm_task_id}...", flush=True)
    # Save dataframe of results
    pd.DataFrame(results).to_csv(RESULTS_DIR / f"neighbor_results__{slurm_task_id}.csv")
    
if __name__=="__main__":
    run()