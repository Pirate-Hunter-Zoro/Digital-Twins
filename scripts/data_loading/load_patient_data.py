import os
from typing import Tuple, Iterator, Dict
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
import pandas as pd
import json
import multiprocessing

from scripts.common.data_loading.fit_to_anchor import slice_and_convert_time, find_anchor_date
from scripts.common.data_loading.create_cohort import create_cohort

MED_DATE_CSV = Path(os.environ['MDD_MED_DATE_CSV_PATH'])
MED_DATE_DF = pd.read_csv(MED_DATE_CSV, escapechar='\\', low_memory=False)
MED_DATE_DF.set_index('PatientEpicId_SH', inplace=True)

COHORT_PATH = Path(os.environ['COHORT_PATH'])
if not COHORT_PATH.exists():
    create_cohort()
COHORT_DF = pd.read_csv(COHORT_PATH, escapechar='\\', low_memory=False)

SEED = int(os.environ['SEED'])
NUM_PATIENTS = int(os.environ['NUM_PATIENTS'])
YEARS_BACK = int(os.environ['YEARS_BACK'])
RAW_JSON_PATH = Path(os.environ['PATIENT_JSON_DIR'])
UNSLICED_JSON_PATH = Path(os.environ['UNSLICED_PATIENT_JSON_DIR'])
SLICED_JSON_PATH = Path(os.environ['SLICED_PATIENT_JSON_DIR'])

# Function for a worker to load one patient
def _load_one_patient(patient_args: Tuple[str, Path, Path, Path, Dict]) -> Tuple[Dict, Dict]:
    """
    Check for validity of the anchor date should it be valid return the sliced and unsliced electronic health record dictionary for the patient
    
    :param patient_args: path to the sliced data, path to the unsliced data, path to the raw data, and their anchor information
    :type patient_args: Tuple[str, Path, Path, Dict]
    :return: Resulting sliced and unsliced EHR data for said patient
    :rtype: Tuple[Dict, Dict]
    """
    sliced_path, unsliced_path, raw_path, anchor_data = patient_args
    sliced_json, unsliced_json = None, None
    if sliced_path.exists() and unsliced_path.exists():
        try:
            with open(sliced_path, 'r') as f:
                sliced_json = json.load(f)
            with open(unsliced_path, 'r') as f:
                unsliced_json = json.load(f)
            return (sliced_json, unsliced_json)
        except json.JSONDecodeError as e:
            print(f"Exception occured when reading from either {sliced_path} or {unsliced_path}... {str(e)}... will try to recreate...", flush=True)
    
    # Load the patient's raw data
    with open(raw_path, 'r') as f:
        raw_json = json.load(f)
    # Verify the anchor date and create the sliced dictionary
    verified_anchor = find_anchor_date(patient_json=raw_json, anchor_data=anchor_data)
    if verified_anchor is not None:
        os.makedirs(UNSLICED_JSON_PATH, exist_ok=True)
        os.makedirs(SLICED_JSON_PATH, exist_ok=True)
        sliced_json, unsliced_json = slice_and_convert_time(patient_dict=raw_json, anchor_date=verified_anchor[0], mdd_date=verified_anchor[1], years_back=YEARS_BACK)
        # Save the json to avoid re-computation
        with open(sliced_path, 'w') as f:
            json.dump(sliced_json, f)
        with open(unsliced_path, 'w') as f:
            json.dump(unsliced_json, f)
        return (sliced_json, unsliced_json)
    else:
        # Anchor date was within a 'washout' period
        return None
    
def load_patient_data() -> Iterator[Tuple[Dict, Dict]]:
    """
    Returns the sliced and unsliced information for all patients
    
    :return: For all sampled patients, return their sliced and unsliced health record going back from their anchor date
    :rtype: Iterator[Tuple[Dict, Dict]]
    """
    # Find the intersection of the medication dates
    cohort_ids = COHORT_DF['PatientEpicId_SH']
    patients_with_anchor = MED_DATE_DF.loc[MED_DATE_DF.index.isin(cohort_ids)]
    print(f"Found {len(patients_with_anchor)} patients with anchor dates in the cohort of {len(cohort_ids)} patients.", flush=True)
    
    # In multiprocessing, each worker will need arguments to do its job
    worker_args = []
    for patient_id, patient_info in patients_with_anchor.iterrows():
        anchor_data = patient_info.to_dict()
        
        worker_args.append((
            SLICED_JSON_PATH / f"patient_{patient_id}.json",
            UNSLICED_JSON_PATH / f"patient_{patient_id}.json",
            RAW_JSON_PATH / f"patient_{patient_id}.json",
            anchor_data
        ))
        
    with multiprocessing.Pool(processes=int(os.environ['NUM_WORKERS_NON_LLM_TASK'])) as pool:
        for json_pair in pool.imap_unordered(func=_load_one_patient, iterable=worker_args):
            if json_pair is not None:
                yield json_pair