from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import copy
import sys
import pandas as pd
from pathlib import Path
import os

MED_INTERVAL_TOLERANCE = 1 # If one patient stops a medication and then this many days later restarts it, just shove it into one interval
WASHOUT = 180
DEBUG = False
MED_DATE_CSV = Path(os.environ['MDD_MED_DATE_CSV_PATH'])
MED_DATE_DF = pd.read_csv(MED_DATE_CSV)

def find_anchor_date(patient_json: dict) -> datetime:
    """
    Find the anchor date for a patient based on their medication history.
    The anchor date is defined as the first date they start an MDD-related medication.
    """
    patient_id = patient_json['patient_id']
    

def slice_and_convert_time(patient_dict: Dict, anchor_date: datetime, mdd_date: datetime, years_back: int) -> Tuple[Dict, Dict]:
    """
    Remove all irrelevant history taking place after the anchor date and recast the remaining events in time relative to the anchor.
    Also add the anchor date as a new field.
    """
    start_date = min(mdd_date, anchor_date - timedelta(days=years_back * 365))
    processed_sliced_patient = {
        'patient_id':patient_dict['patient_id'],
        'demographics':patient_dict['demographics'],
        'anchor_date':anchor_date.strftime('%Y-%m-%d'),
        'active_medications':[],
        'encounters':[]
    }
    processed_patient = copy.deepcopy(processed_sliced_patient)
    
    return (processed_sliced_patient, processed_patient)