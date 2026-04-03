import pandas as pd
from pathlib import Path
import os
import multiprocessing
import numpy as np

from scripts.digital_twins.predictions.trd_predictor import TRDPredictor
from scripts.digital_twins.neighbors.neighbor_scheme import NeighborScheme
from scripts.digital_twins.predictions.create_train_test_split import create_train_test_split

from dotenv import load_dotenv
load_dotenv()

RESULTS_DIR = Path(os.environ['RESULTS_DIR'])
os.makedirs(RESULTS_DIR, exist_ok=True)
predictor = None
test_ids = create_train_test_split()[1]

def init_worker():
    global predictor
    np.random.seed(int(os.environ['SEED']))
    predictor = TRDPredictor(exclude_ids=test_ids)

def evaluate_patient(patient_id: str) -> list[dict]:
    """
    Obtain all of the patient's neighborhood prediction information
    
    :param patient_id:ID of patient
    :type patient_info: str
    :return: patient neighborhood results
    :rtype: list[dict]
    """
    global predictor
    results = []
    for scheme in NeighborScheme:
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
        for res in pool.imap_unordered(evaluate_patient, chunk_ids):
            results.extend(res)
    # Save dataframe of results
    pd.DataFrame(results).to_csv(RESULTS_DIR / f"neighbor_results_{slurm_task_id}.csv")
    
if __name__=="__main__":
    run()