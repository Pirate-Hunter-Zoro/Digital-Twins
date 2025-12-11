import os
from typing import Tuple, Iterator, Dict
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
import pandas as pd

from common.data_loading.fit_to_anchor import slice_and_convert_time, find_anchor_date

MED_DATE_CSV = Path(os.environ['MDD_MED_DATE_CSV_PATH'])
MED_DATE_DF = pd.read_csv(MED_DATE_CSV)
MED_DATE_DF.set_index('PatientEpicId_SH')

COHORT_PATH = Path(os.environ['COHORT_PATH'])
SEED = int(os.environ['SEED'])
NUM_PATIENTS = int(os.environ['NUM_PATIENTS'])
RAW_JSON_PATH = Path(os.environ['PATIENT_JSON_DIR'])
SLICED_JSON_PATH = Path(os.environ['SLICED_PATIENT_JSON_DIR'])
    
def load_patient_data() -> Iterator[Tuple[Dict, Dict]]:
    
    pass