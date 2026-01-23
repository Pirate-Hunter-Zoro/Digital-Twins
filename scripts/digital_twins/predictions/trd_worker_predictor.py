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

def evaluate_patient(patient_info: tuple[str, str]) -> dict:
    """
    Evaluate TRD prediction for the patient with the given narrative hash id and patient id
    
    :param patient_info: narrative hash ID and patient ID
    :type patient_info: tuple[str, str]
    :return: TRD prediction results
    :rtype: dict
    """
    global predictor
    narrative_hash_id, patient_id = patient_info
    print("Predicting TRD risk for patient ID:", patient_id, flush=True)
    trd_status = predictor.get_trd_status(candidate_id=patient_id)
    prediction = predictor.predict_risk(index_id=narrative_hash_id)
    random_llm_similarity_scores = predictor.get_random_sample_judge_scores(index_id=narrative_hash_id)
    return {
        'patient_id' : patient_id,
        'actual_trd_status' : trd_status,
        'trd_risk_score_llm' : prediction['risk_score'][0],
        'trd_risk_score_cosine' : prediction['risk_score'][1],
        'trd_risk_score_uniform' : prediction['risk_score'][2],
        'ess_llm' : prediction['confidence_ess'][0],
        'ess_cosine' : prediction['confidence_ess'][1],
        'ess_uniform' : prediction['confidence_ess'][2],
        'nearest_llm_scores' : str(prediction['nearest_llm_scores']),
        'random_llm_scores' : str(random_llm_similarity_scores),
        'neighbors_by_llm_score' : str(prediction['neighbors_sorted_by_llm_weight']),
        'neighbors_by_cos_score' : str(prediction['neighbors_sorted_by_cosine_weight'])
    }

def run():
    """
    Single worker process to run TRD prediction on a sub-sample of patients
    """
    vector_db = Path(os.environ['VECTORS_DIR']) / 'vectors.db'
    connection = sqlite3.connect(vector_db)
    cursor = connection.cursor()
    # Randomly pick n patients but ensure that half are TRD positive and half are TRD negative
    cursor.execute("""
SELECT id, patient_id FROM vectors                          
"""
    ) 
    rows = cursor.fetchall()
    predictor = TRDPredictor()
    n = int(os.environ['TRD_TEST_COUNT'])
    trd_positive = [row for row in rows if predictor.get_trd_status(candidate_id=row[1]) == 1]
    trd_negative = [row for row in rows if predictor.get_trd_status(candidate_id=row[1]) == 0]
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
    patient_chunk_for_worker = patient_sample[start_idx: end_idx]
    
    results = []
    with multiprocessing.Pool(processes=int(os.environ['NUM_WORKERS_LLM_TASK']), initializer=init_worker) as pool:
        for result in tqdm(pool.imap_unordered(evaluate_patient, patient_chunk_for_worker), total=len(patient_chunk_for_worker)):
            results.append(result)
    # Turn results into a dataframe
    results_df = pd.DataFrame(results)
    
    # Save dataframe to a .csv and the results to a .txt
    results_csv = RESULTS_DIR / f'trd_evaluation_results_{slurm_task_id}.csv'
    results_df.to_csv(results_csv, index=False)