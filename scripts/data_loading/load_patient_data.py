import os
from typing import Tuple, Iterator, Dict, Optional
from pathlib import Path
import pandas as pd
import json
import multiprocessing
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from scripts.data_loading.fit_to_anchor import slice_and_convert_time
from scripts.data_loading.create_cohort import create_cohort

# Information on all patients assigned an antidepressant
MED_DATE_CSV = Path(os.environ['MDD_MED_DATE_CSV_PATH'])
MED_DATE_DF = pd.read_csv(MED_DATE_CSV, escapechar='\\', low_memory=False)
MED_DATE_DF.set_index('PatientEpicId_SH', inplace=True)

# Create list of MDD patients
COHORT_PATH = Path(os.environ['COHORT_PATH'])
if not COHORT_PATH.exists():
    create_cohort()
COHORT_DF = pd.read_csv(COHORT_PATH, escapechar='\\', low_memory=False)

# Location of patient jsons
RAW_JSON_PATH = Path(os.environ['PATIENT_JSON_DIR'])
SLICED_JSON_PATH = Path(os.environ['SLICED_PATIENT_JSON_DIR'])

# Function for a worker to load one patient
def _load_one_patient(patient_args: Tuple[Path, Path, Dict]) -> Dict:
    """Return sliced json of the patient if they fit the window criteria, or a json explaining why they did not

    Args:
        patient_args (Tuple[Path, Path, Dict]): sliced path, raw path, anchor data

    Returns:
        Dict: Resulting sliced json or failure reason
    """
    sliced_path, raw_path, anchor_data = patient_args
    failure_path = sliced_path.with_suffix(".rejected")
    sliced_json = None
    if int(os.environ['SCRUB_PATIENT_JSON']) == 0:
        if sliced_path.exists():
            try:
                with open(sliced_path, 'r') as f:
                    sliced_json = json.load(f)
                    return sliced_json
            except json.JSONDecodeError as e:
                print(f"Exception occured when reading from {sliced_path}... {str(e)}... will try to recreate...", flush=True)
        elif failure_path.exists():
            # Patient was rejected due to inadequate history
            try:
                with open(failure_path, 'r') as f:
                    failure_json = json.load(f)
                    return failure_json
            except json.JSONDecodeError as e:
                print(f"Exception occured when reading from {failure_path}... {str(e)}... will try to recreate...", flush=True)
    
    # Load the patient's raw data
    with open(raw_path, 'r') as f:
        raw_json = json.load(f)
        sliced_json = slice_and_convert_time(patient_dict=raw_json, anchor_date=datetime.strptime(anchor_data.get('MedStartInstant'), '%Y-%m-%d'))
        if 'reason' not in sliced_json.keys():
            # Record the json
            with open(sliced_path, 'w') as f:
                json.dump(sliced_json, f, indent=4)
        else:
            # Record the failure
            with open(failure_path, 'w') as f:
                # Just create the dummy file to mark the failure
                json.dump(sliced_json, f, indent=4)
        return sliced_json
    
def load_patient_data() -> Iterator[Dict]:
    """
    Returns the sliced JSON for all patients
    """
    # Make sliced directory if it has not yet been made
    os.makedirs(SLICED_JSON_PATH, exist_ok=True)
    
    # Find the intersection of the medication dates
    cohort_ids = COHORT_DF['PatientEpicId_SH']
    patients_with_anchor = MED_DATE_DF.loc[MED_DATE_DF.index.isin(cohort_ids)]
    patients_with_anchor = patients_with_anchor.sort_values(by='MedStartInstant', ascending=True)
    earliest_mask = ~patients_with_anchor.index.duplicated(keep='first')
    patients_with_anchor = patients_with_anchor[earliest_mask]
    print(f"Found {len(patients_with_anchor)} patients with anchor dates in the cohort of {len(cohort_ids)} patients (pre-filter upper bound; YEARS_BACK / YEARS_AHEAD slicing will reject those without adequate history or follow-up).", flush=True)
    
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
            if sliced_json != None and 'reason' not in sliced_json.keys():
                yield sliced_json