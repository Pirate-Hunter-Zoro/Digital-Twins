from typing import Dict, Tuple, Optional
from datetime import datetime, timedelta
import copy
import pandas as pd
from pathlib import Path
import os

MED_INTERVAL_TOLERANCE = 1 # If one patient stops a medication and then this many days later restarts it, just shove it into one interval
WASHOUT = 180
DEBUG = False
MED_DATE_CSV = Path(os.environ['MDD_MED_DATE_CSV_PATH'])
MED_DATE_DF = pd.read_csv(MED_DATE_CSV)

def find_anchor_date(patient_json: Dict, anchor_data: Optional[pd.Series]) -> Optional[Tuple[datetime, datetime]]:
    """
    Find the anchor date of a patient given their history and the med date dataframe row that pertains to them
    
    :param patient_json: All patient's electronic health records
    :type patient_json: Dict
    :param anchor_data: Information on the 
    :type anchor_data: Optional[pd.Series]
    :return: Corresponding MDD diagnosis and preceding medications dates
    :rtype: Tuple[datetime, datetime] | None
    """
    if anchor_data is None:
        # The patient never even had any post-MDD medications
        return None
    
    candidate_date = datetime.strptime(anchor_data['MedStartInstant'], '%Y-%m-%d')
    mdd_date = datetime.strptime(anchor_data['first_depression_dx_date'], '%Y-%m-%d')
    target_ingredient = anchor_data['MedSimpleGenericName']
    for encounter in patient_json['encounters']:
        for med in encounter['medications']:
            if (target_ingredient.lower() in med['MedSimpleGenericName'].lower()) or (target_ingredient.lower() in med['MedName'].lower()):
                # See if this falls within the washout date
                med_start_date = datetime.strptime(med['MedStartInstant'], '%Y-%m-%d')
                if (med_start_date > candidate_date - timedelta(days=WASHOUT)) and (med_start_date < candidate_date):
                    # The date found for the first post-MDD antidepressant will not suffice as the patient had the same ingredient within the previous washout period
                    return None
    return (candidate_date, mdd_date)
    

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
    
    unique_sliced_meds = {}
    unique_unsliced_meds = {}
    for encounter in patient_dict['encounters']:
        encounter_copy = {'details' : encounter['details'], 'procedures' : [], 'diagnoses' : encounter['diagnoses']}
        info = encounter_copy['details']
        encounter_start_date = datetime.strptime(info['start_visit'], '%Y-%m-%d')
        encounter_end_date = min(anchor_date, datetime.strptime(info['end_visit'], '%Y-%m-%d'))
        if encounter_start_date > anchor_date:
            # Happens in the future - does not even belong in the unsliced dict
            continue
        
        start_offset = -(anchor_date - encounter_start_date).days
        end_offset = -(anchor_date - encounter_end_date).days
        info['start_visit'] = start_offset
        info['end_visit'] = end_offset
        for procedure in encounter['procedures']:
            procedure_copy = copy.deepcopy(procedure)
            procedure_start = datetime.strptime(procedure_copy['ProcedureStartInstant'], '%Y-%m-%d')
            procedure_end = min(anchor_date, datetime.strptime(procedure_copy['ProcedureEndInstant'], '%Y-%m-%d'))
            procedure_copy['ProcedureStartInstant'] = -(anchor_date-procedure_start).days
            procedure_copy['ProcedureEndInstant'] = -(anchor_date-procedure_end).days
            encounter_copy['procedures'].append(procedure_copy)
        
        # Append the encounter into the encounter list of the unsliced dict
        processed_patient['encounters'].append(encounter_copy)
        if encounter_start_date >= start_date:
            # Append the encounter into the encounter list of the sliced dict - since it occurred within the time window
            processed_sliced_patient['encounters'].append(encounter_copy)
            
        # Handle the active medications - note that the same medication MAY be listed multiple times due to overlap which means we should collapse such intervals
        med_intervals = {}
        sliced_med_intervals = {}
        for med in encounter['medications']:
            med_start_date = datetime.strptime(med['MedStartInstant'], '%Y-%m-%d')
            if med_start_date > anchor_date:
                # Med started in the future - not interested
                continue
            med_end_date = min(anchor_date, datetime.strptime(med['MedEndInstant'], '%Y-%m-%d'))
            key = (
                med["MedName"],
                med["MedSimpleGenericName"],
                med["MedStrength"],
                med["MedForm"],
                med["MedRoute"],
                med["MedFrequency"],
            )
            med_copy = copy.deepcopy(med)
            med_start_num = -(anchor_date-med_start_date).days
            med_end_num = -(anchor_date-med_end_date).days
            # DEFINITELY include this medication interval to the unsliced patient json
            if key not in med_intervals.keys():
                med_intervals[key] = []
            med_intervals[key].append([med_start_num, med_end_num])
            
            if med_end_date > 
    
    return (processed_sliced_patient, processed_patient)