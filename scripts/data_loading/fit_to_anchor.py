from typing import Dict, Tuple, Optional
from datetime import datetime, timedelta
import copy
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
import os

# If someone we have never seen before comes into the hospital that leads to MDD and antidepressant ASAP
# Then they could have zero length - technically should be one day
# Look at that patient's JSON and see if this is the case
# Take days calculation and add 1

# Super long history possible explanation: Age 20 - MDD diagnosis, no antidepressant; 10 years later, come in and get antidepressant for unrelated reason

MED_OVERLAP_TOLERANCE = 1 # If one patient stops a medication and then this many days later restarts it, just shove it into one interval
DEBUG = False
YEARS_BACK = int(os.environ['YEARS_BACK'])
YEARS_FORWARD = int(os.environ['YEARS_FORWARD'])

def find_anchor_date(anchor_data: Optional[pd.Series]) -> Optional[Tuple[datetime, datetime]]:
    """
    Find the anchor date of a patient given their history and the med date dataframe row that pertains to them.
    """
    if anchor_data is None:
        return None
    
    # 1. Validate Dates
    start_date_str = anchor_data.get('MedStartInstant')
    mdd_date_str = anchor_data.get('first_depression_dx_date')
    
    if not isinstance(start_date_str, str) or not isinstance(mdd_date_str, str):
        return None
        
    candidate_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    mdd_date = datetime.strptime(mdd_date_str, '%Y-%m-%d')
                    
    return (candidate_date, mdd_date)

def merge_and_add(med_intervals: dict[any, list[list[int]]], patient_json: dict):
    """
    Given interval of occurences for a bunch of unique medications, for each medication merge the intervals and append all unique intervals of occurence to the patient json
    
    :param med_intervals: Medications' intervals of occurrences
    :type med_intervals: dict[any, list[list[int]]]
    :param patient_json: Record to update post-interval merging
    :type patient_json: dict
    """
    for med_key, dates in med_intervals.items():
        dates.sort(key=lambda x : x[0]) # Sort this medications active intervals by start time
        merged_intervals = [dates[0]]
        for date in dates[1:]:
            if merged_intervals[-1][1] >= date[0] - MED_OVERLAP_TOLERANCE: # Merge this interval with the last merged interval
                merged_intervals[-1][1] = max(merged_intervals[-1][1], date[1]) # New ending time for this med is potentially greater
            else:
                merged_intervals.append(date)
        # Now all disjoint intervals count for unique instances of this particular medication
        for interval in merged_intervals:
            patient_json['active_medications'].append(
                {
                    "MedName" : med_key[0],
                    "MedSimpleGenericName" : med_key[1],
                    "MedStrength" : med_key[2],
                    "MedForm" : med_key[3],
                    "MedRoute" : med_key[4],
                    "MedFrequency" : med_key[5],
                    "MedStartInstant" : interval[0],
                    "MedEndInstant" : interval[1]
                }
            )

def slice_and_convert_time(patient_dict: Dict, anchor_date: datetime, mdd_date: datetime) -> Tuple[Dict, Dict]:
    start_date = min(mdd_date, anchor_date - timedelta(days=YEARS_BACK * 365))
    # For the unsliced json
    forward_limit = anchor_date + timedelta(days=YEARS_FORWARD * 365)
    
    processed_sliced_patient = {
        'patient_id': patient_dict['patient_id'],
        'demographics': patient_dict['demographics'],
        'anchor_date': anchor_date.strftime('%Y-%m-%d'),
        'active_medications': [],
        'encounters': []
    }
    processed_patient = copy.deepcopy(processed_sliced_patient)
    
    med_intervals = {}
    sliced_med_intervals = {}
    
    earliest_unsliced_encounter_date = anchor_date
    earliest_sliced_encounter_date = anchor_date
    for encounter in patient_dict['encounters']:
        sliced_encounter = {'details': encounter['details'], 'procedures': [], 'diagnoses': encounter['diagnoses'], 'vitals': encounter['vitals'],}
        info = sliced_encounter['details']
        
        enc_start_str = info.get('start_visit')
        if not isinstance(enc_start_str, str):
            continue # Skip bad encounters
            
        encounter_start_date = datetime.strptime(enc_start_str, '%Y-%m-%d')
        if encounter_start_date < earliest_unsliced_encounter_date:
            earliest_unsliced_encounter_date = encounter_start_date
        if encounter_start_date < earliest_sliced_encounter_date and encounter_start_date >= start_date:
            earliest_sliced_encounter_date = encounter_start_date
        
        enc_end_str = info.get('end_visit')
        if isinstance(enc_end_str, str):
            encounter_end_date = datetime.strptime(enc_end_str, '%Y-%m-%d')
        else:
            encounter_end_date = encounter_start_date

        if encounter_start_date > forward_limit:
            continue
        
        start_offset = -(anchor_date - encounter_start_date).days
        end_offset = -(anchor_date - encounter_end_date).days
        info['start_visit'] = start_offset
        info['end_visit'] = min(0, end_offset)
        
        # Unsliced encounter
        unsliced_encounter = copy.deepcopy(sliced_encounter)
        unsliced_encounter['details']['end_visit'] = end_offset # Don't clip the ending in the unsliced dict
        
        for procedure in encounter['procedures']:
            procedure_copy = copy.deepcopy(procedure)
            proc_start_str = procedure_copy.get('ProcedureStartInstant')
            if isinstance(proc_start_str, str):
                procedure_start = datetime.strptime(proc_start_str, '%Y-%m-%d')
                procedure_copy['ProcedureStartInstant'] = -(anchor_date - procedure_start).days
                
                # Handle End Date
                proc_end_str = procedure_copy.get('ProcedureEndInstant')
                if isinstance(proc_end_str, str):
                    procedure_end = datetime.strptime(proc_end_str, '%Y-%m-%d')
                    procedure_copy['ProcedureEndInstant'] = -(anchor_date - procedure_end).days
                else:
                    procedure_copy['ProcedureEndInstant'] = procedure_copy['ProcedureStartInstant']
                
                if procedure_start <= anchor_date:
                    sliced_procedure = copy.deepcopy(procedure_copy)
                    sliced_procedure['ProcedureEndInstant'] = min(0, sliced_procedure['ProcedureEndInstant'])
                    sliced_encounter['procedures'].append(sliced_procedure)
                
                # Append to unsliced encounter regardless
                unsliced_encounter['procedures'].append(procedure_copy)
        
        processed_patient['encounters'].append(unsliced_encounter)
        if encounter_start_date >= start_date:
            processed_sliced_patient['encounters'].append(sliced_encounter)
            
        for med in encounter['medications']:
            # --- Verify Medication Dates ---
            med_start_str = med.get('MedStartInstant')
            if not isinstance(med_start_str, str):
                continue
            
            med_start_date = datetime.strptime(med_start_str, '%Y-%m-%d')
            if med_start_date > forward_limit:
                continue
                
            med_end_date_str = med.get('MedEndInstant')
            if isinstance(med_end_date_str, str):
                med_end_date = datetime.strptime(med_end_date_str, '%Y-%m-%d')
            else:
                med_end_date = forward_limit
                
            key = (
                med.get("MedName"),
                med.get("MedSimpleGenericName"),
                med.get("MedStrength"),
                med.get("MedForm"),
                med.get("MedRoute"),
                med.get("MedFrequency"),
            )
            
            med_start_num = -(anchor_date - med_start_date).days
            med_end_num = -(anchor_date - med_end_date).days
            
            if key not in med_intervals: med_intervals[key] = []
            med_intervals[key].append([med_start_num, med_end_num])
            
            if med_end_date > start_date and med_start_date <= anchor_date:
                if key not in sliced_med_intervals: sliced_med_intervals[key] = []
                sliced_med_intervals[key].append([med_start_num, min(0, med_end_num)])
        
    merge_and_add(med_intervals, processed_patient)
    merge_and_add(sliced_med_intervals, processed_sliced_patient)
    
    # Sort
    processed_patient['encounters'].sort(key=lambda x: -x['details']['start_visit'])
    processed_patient['active_medications'].sort(key=lambda x: -x['MedStartInstant'])
    processed_sliced_patient['encounters'].sort(key=lambda x: -x['details']['start_visit'])
    processed_sliced_patient['active_medications'].sort(key=lambda x: -x['MedStartInstant'])
    
    # Total chronological length
    timespan_unsliced = (anchor_date - earliest_unsliced_encounter_date).days + 1
    timespan_sliced = (anchor_date - earliest_sliced_encounter_date).days + 1
    processed_sliced_patient['days_of_history'] = timespan_sliced
    processed_patient['days_of_history'] = timespan_unsliced
    
    return (processed_sliced_patient, processed_patient)

if __name__=="__main__":
    from pathlib import Path
    import json
    json_path = Path("/media/studies/ehr_study/analysis/mferguson/sliced_patient_json/")
    ids = []
    empty_ids = []
    record_every = 1000
    done = 0
    for json_file in json_path.glob("*.json"):
        id = json_file.stem[8:]
        with open(json_file, 'r') as f:
            contents = f.read()
            if len(contents) == 0:
                empty_ids.append(id)
            else:
                patient_json = json.loads(contents)
                if patient_json['days_of_history'] == 1:
                    ids.append(id)
        done += 1
        if done % record_every == 0:
            print(f"Scanned {done} patient json files...", flush=True)
    with open(Path("test_data/length_1_ids.txt"), 'w') as f:
        f.write("\n".join(ids))
    with open(Path("test_data/empty_ids.txt"), 'w') as f:
        f.write("\n".join(empty_ids))