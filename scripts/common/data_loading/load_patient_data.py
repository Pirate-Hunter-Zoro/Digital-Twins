import os
from typing import Iterator, Dict, Optional
from pathlib import Path
import multiprocessing
import random
from dotenv import load_dotenv
import json

from common.data_loading.fit_to_anchor import slice_and_convert_time, find_anchor_date

PROJECT_ROOT = Path(__file__).resolve().parents[3]

load_dotenv()

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
                raise ValueError(f"Patient at path {original_patient_json_file_path} does not have relevant diagnosis.")
            sliced_json = slice_and_convert_time(patient_json, anchor_date, int(os.environ['YEARS_BACK']))
            
            sliced_patient_json_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(sliced_patient_json_file_path, 'w') as f:
                # Store the sliced json now that we have created it
                json.dump(sliced_json, f, indent=4)
            return sliced_json
        
def _see_if_anchor(json_path: Path) -> Optional[str]:
    with open(json_path, 'r') as f:
        patient_dict = json.load(f)
        if find_anchor_date(patient_dict):
            return patient_dict['patient_id']
        return None

def load_patient_data() -> Iterator[Dict]:
    seed = int(os.environ['SEED'])
    num_patients = int(os.environ['NUM_PATIENTS'])
    raw_json_path = Path(os.environ['PATIENT_JSON_DIR'])
    sliced_json_path = Path(os.environ['SLICED_PATIENT_JSON_DIR'])
    artifacts_path = PROJECT_ROOT / "artifacts"
    artifacts_path.mkdir(parents=True, exist_ok=True)
    sampled_ids_path = artifacts_path / f"{num_patients}_patients/sampled_patient_ids.txt"
    sampled_ids_path.parent.mkdir(parents=True, exist_ok=True)
    all_eligible_path = artifacts_path / "all_patient_ids_with_diagnosis.txt"
    
    if not sampled_ids_path.exists():
        # We must sample and from all potential patients of interest
        relevant_ids = []
        if not all_eligible_path.exists():
            # We need to know who our valid patients with the relevant diagnosis we care about are
            all_raw_json = raw_json_path.glob("*.json")
            with multiprocessing.Pool(processes=int(os.environ['NUM_WORKERS_NON_LLM_TASK'])) as thread_pool:
                for id in thread_pool.imap_unordered(_see_if_anchor, all_raw_json):
                    if id:
                        relevant_ids.append(id)
        rnd = random.Random(seed)
        random_valid_ids = rnd.sample(relevant_ids, num_patients)
        with open(sampled_ids_path, 'w') as f:
            f.write("\n".join(random_valid_ids))
        
    with open(sampled_ids_path, 'r') as f:
        # Ignore empty lines
        cohort_ids = [line.strip() for line in f if line.strip()]
        # Create list of tuples - the first element is the patient's sliced json path, and the second is the patient's original json path
        sliced_and_unsliced_paths = [(sliced_json_path / f'patient_{id}.json', raw_json_path / f'patient_{id}.json') for id in cohort_ids]
        with multiprocessing.Pool(processes=int(os.environ['NUM_WORKERS_NON_LLM_TASK'])) as thread_pool:
            yield from thread_pool.imap_unordered(_load_one_patient, sliced_and_unsliced_paths)