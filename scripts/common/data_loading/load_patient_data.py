import os
from typing import Iterator, Dict, Optional
from pathlib import Path
import multiprocessing
import random
from dotenv import load_dotenv
load_dotenv()
import json
import pandas as pd

from common.data_loading.fit_to_anchor import slice_and_convert_time, find_anchor_date

COHORT_PATH = Path(os.environ['COHORT_PATH'])
SEED = int(os.environ['SEED'])
NUM_PATIENTS = int(os.environ['NUM_PATIENTS'])
RAW_JSON_PATH = Path(os.environ['PATIENT_JSON_DIR'])
SLICED_JSON_PATH = Path(os.environ['SLICED_PATIENT_JSON_DIR'])

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS_PATH = PROJECT_ROOT / "artifacts"
ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)
SAMPLED_IDS_PATH = ARTIFACTS_PATH / f"{NUM_PATIENTS}_patients/sampled_patient_ids.txt"
SAMPLED_IDS_PATH.parent.mkdir(parents=True, exist_ok=True)

def _load_one_patient(paths: tuple[Path,Path]) -> Dict:
    sliced_patient_json_file_path, original_patient_json_file_path = paths[0], paths[1]
    if sliced_patient_json_file_path.exists():
        # If we already have the sliced json, just return it
        with open(sliced_patient_json_file_path, 'r') as f:
            return json.load(f)
    else:
        # Otherwise we have to make it
        if not original_patient_json_file_path.exists():
            raise ValueError(f"Tried to sample patient with non-existant json - {original_patient_json_file_path} does not exist.")
        with open(original_patient_json_file_path, 'r') as f:
            patient_json = json.load(f)
            anchor_date = find_anchor_date(patient_json)
            if not anchor_date:
                raise ValueError(f"Patient at path {original_patient_json_file_path} does not have relevant medication.")
            sliced_json = slice_and_convert_time(patient_json, anchor_date, int(os.environ['YEARS_BACK']))
            
            sliced_patient_json_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(sliced_patient_json_file_path, 'w') as f:
                # Store the sliced json now that we have created it
                json.dump(sliced_json, f, indent=4)
            return sliced_json
    
def _check_anchor_for_id(patient_id: str) -> Optional[str]:
    # Worker function
    if _see_if_anchor(RAW_JSON_PATH / f"patient_{patient_id}.json"):
        return patient_id
    return None
        
def _see_if_anchor(json_path: Path) -> bool:
    # Helper function FOR the worker function
    try:
        with open(json_path, 'r') as f:
            patient_dict = json.load(f)
            if find_anchor_date(patient_dict):
                return True
            return False
    except Exception:
        return False
    
def load_patient_data() -> Iterator[Dict]:
    
    if not SAMPLED_IDS_PATH.exists():
        cohort_df = pd.read_csv(COHORT_PATH)
        patient_ids_with_mdd = cohort_df['PatientEpicId_SH'].tolist()
        patient_ids_with_anchor = []
        with multiprocessing.Pool(processes=int(os.environ['NUM_WORKERS_NON_LLM_TASK'])) as pool:
            for id in pool.imap_unordered(_check_anchor_for_id, patient_ids_with_mdd):
                if id:
                    patient_ids_with_anchor.append(id)
        
        rnd = random.Random(SEED)
        sampled_patients = rnd.sample(patient_ids_with_anchor, k=min(NUM_PATIENTS, len(patient_ids_with_anchor)))
        os.makedirs(SAMPLED_IDS_PATH.parent, exist_ok=True)
        with open(SAMPLED_IDS_PATH, 'w') as f:
            f.write("\n".join(sampled_patients))
        
    with open(SAMPLED_IDS_PATH, 'r') as f:
        # Ignore empty lines
        cohort_ids = [line.strip() for line in f if line.strip()]
        # Create list of tuples - the first element is the patient's sliced json path, and the second is the patient's original json path
        sliced_and_unsliced_paths = [(SLICED_JSON_PATH / f'patient_{id}.json', RAW_JSON_PATH / f'patient_{id}.json') for id in cohort_ids]
        with multiprocessing.Pool(processes=int(os.environ['NUM_WORKERS_NON_LLM_TASK'])) as thread_pool:
            yield from thread_pool.imap_unordered(_load_one_patient, sliced_and_unsliced_paths)