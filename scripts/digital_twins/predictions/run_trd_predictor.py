import random
import pandas as pd
import sqlite3
from pathlib import Path
import os
from tqdm import tqdm
import multiprocessing

from scripts.digital_twins.predictions.trd_predictor import TRDPredictor

from dotenv import load_dotenv
load_dotenv()

RESULTS_DIR = Path(os.environ['RESULTS_DIR'])
os.makedirs(RESULTS_DIR, exist_ok=True)
predictor = None

def init_worker():
    global predictor
    predictor = TRDPredictor()

def evaluate_patient(narrative_hash_id: str) -> list[dict]:
    """
    Obtain all of the patient's neighborhood prediction information
    
    :param narrative_hash_id: narrative hash ID of patient
    :type patient_info: str
    :return: patient neighborhood results
    :rtype: list[dict]
    """
    global predictor
    results_random = predictor.construct_neighborhood_data(index_id=narrative_hash_id, random=True)
    results = predictor.construct_neighborhood_data(index_id=narrative_hash_id)
    results.extend(results_random)
    return results

def run():
    """
    Single worker process to run TRD prediction on a sub-sample of patients
    """
    vector_db = Path(os.environ['VECTORS_DIR']) / 'vectors.db'
    connection = sqlite3.connect(vector_db)
    cursor = connection.cursor()
    # Randomly pick n patients but ensure that half are TRD positive and half are TRD negative
    cursor.execute("""
SELECT patient_id FROM vectors                          
"""
    ) 
    rows = cursor.fetchall()
    predictor = TRDPredictor()
    n = int(os.environ['TRD_TEST_COUNT'])
    trd_positive = [row for row in rows if predictor.get_trd_status(candidate_patient_id=row[0]) == 1]
    trd_negative = [row for row in rows if predictor.get_trd_status(candidate_patient_id=row[0]) == 0]
    print(f"Found {len(trd_positive)} TRD positive and {len(trd_negative)} TRD negative patients", flush=True)
    
    # Each worker is going to do this, and we need them all to yield the same sample - hence the seeding
    random.seed(int(os.environ['SEED']))
    patient_sample = random.sample(trd_positive, min(len(trd_positive), n//2)) + random.sample(trd_negative, min(len(trd_negative), n//2))
    
    # Now break up the patient sample to different workers
    slurm_task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    slurm_task_count = int(os.environ['SLURM_ARRAY_TASK_COUNT'])
    chunk_length = int(n / slurm_task_count)
    
    # Chunk patient sample for this worker
    start_idx = slurm_task_id*chunk_length
    end_idx = len(patient_sample) if slurm_task_id == slurm_task_count - 1 else start_idx + chunk_length
    # Python handles out of range end_idx
    patient_chunk_for_worker = patient_sample[start_idx: end_idx] # For the worker below, we will only grab the narrative Hash ID; not the patient ID
    
    results = []
    with multiprocessing.Pool(processes=int(os.environ['NUM_WORKERS_LLM_TASK']), initializer=init_worker) as pool:
        for result in tqdm(pool.imap_unordered(evaluate_patient, [info[0] for info in patient_chunk_for_worker]), total=len(patient_chunk_for_worker)):
            results.extend(result)
    # Turn results into a dataframe
    results_df = pd.DataFrame(results)
    
    # Save dataframe to a .csv and the results to a .txt
    results_csv = RESULTS_DIR / f'neighbor_results_{slurm_task_id}.csv'
    results_df.to_csv(results_csv, index=False)
    
if __name__=="__main__":
    run()