import os
from typing import Tuple, Iterator, Dict
from pathlib import Path
import pandas as pd
import json
import multiprocessing
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from scripts.data_loading.fit_to_anchor import slice_and_convert_time
from scripts.data_loading.create_cohort import create_cohort

MED_DATE_CSV = Path(os.environ['MDD_MED_DATE_CSV_PATH'])
MED_DATE_DF = pd.read_csv(MED_DATE_CSV, escapechar='\\', low_memory=False)
MED_DATE_DF.set_index('PatientEpicId_SH', inplace=True)

COHORT_PATH = Path(os.environ['COHORT_PATH'])
if not COHORT_PATH.exists():
    create_cohort()
COHORT_DF = pd.read_csv(COHORT_PATH, escapechar='\\', low_memory=False)

RAW_JSON_PATH = Path(os.environ['PATIENT_JSON_DIR'])
UNSLICED_JSON_PATH = Path(os.environ['UNSLICED_PATIENT_JSON_DIR'])
SLICED_JSON_PATH = Path(os.environ['SLICED_PATIENT_JSON_DIR'])

# Function for a worker to load one patient
def _load_one_patient(patient_args: Tuple[Path, Path, Dict]) -> Dict:
    """
    Return sliced json of the patient
    """
    sliced_path, raw_path, anchor_data = patient_args
    sliced_json = None
    if sliced_path.exists() and int(os.environ['SCRUB_PATIENT_JSON']) == 0:
        try:
            with open(sliced_path, 'r') as f:
                sliced_json = json.load(f)
                return sliced_json
        except json.JSONDecodeError as e:
            print(f"Exception occured when reading from {sliced_path}... {str(e)}... will try to recreate...", flush=True)
    
    # Load the patient's raw data
    with open(raw_path, 'r') as f:
        raw_json = json.load(f)
        return slice_and_convert_time(patient_dict=raw_json, anchor_date=datetime.strptime(anchor_data.get('MedStartInstant'), '%Y-%m-%d'))
    
def load_patient_data() -> Iterator[Dict]:
    """
    Returns the sliced JSON for all patients
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
            SLICED_JSON_PATH / f"{patient_id}.json",
            RAW_JSON_PATH / f"{patient_id}.json",
            anchor_data
        ))
        
    with multiprocessing.Pool(processes=int(os.environ['NUM_WORKERS_NON_LLM_TASK'])) as pool:
        for sliced_json in pool.imap_unordered(func=_load_one_patient, iterable=worker_args):
            if sliced_json is not None:
                yield sliced_json