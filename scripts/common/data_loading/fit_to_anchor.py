import os
import json
from typing import List, Dict, Optional
import re
from datetime import datetime, timedelta
import copy
import pandas as pd
from dotenv import load_dotenv
import sys

load_dotenv()

ICD_CODE_PATTERN = re.compile(r"(F32|F33|296.2|296.3)")
MED_INTERVAL_TOLERNACE = 1 # Consider the same medication if all fields are the same and the ending of the first is only this many days before the start of the second

def find_anchor_date(patient_dict: Dict) -> Optional[datetime]:
    """
    Helper function to find the first instance of one of the ICD_CODE_PATTERN diagnoses, and return its date
    """
    anchor_date = None
    for encounter in patient_dict['encounters']:
        for diagnosis in encounter['diagnoses']:
            # This diagnosis has potentially multiple code values attached to it
            for code in diagnosis['codes']:
                icd_code = code['code']
                match = ICD_CODE_PATTERN.match(icd_code)
                if match:
                    date_str = encounter['details']['start_visit']
                    current_date = datetime.strptime(date_str, '%Y-%m-%d')
                    # See if this date broke the record for earliest occurence of MDD diagnosis
                    if anchor_date == None or current_date < anchor_date: # Note - datetime objects are comparable
                        anchor_date = current_date
    return anchor_date

def deep_clean_dict(original_dict: Dict):
    """
    Helper method to remove all 'null', 'None', and 'NaN' values in place
    """
    # Must create initial shallow copy because we cannot delete keys while iterating over original dictionary
    for key, value in original_dict.copy().items(): # Note that we do not need to deep copy, because each inner dictionary will get a recursive call and have 'copy()' called on it
        if isinstance(value, Dict):
            deep_clean_dict(value) # In place, so that takes care of it
        elif isinstance(value, List):
            for v in value:
                if isinstance(v, Dict):
                    deep_clean_dict(v) # Again in place
        elif pd.isna(value):
            del original_dict[key]

def slice_and_convert_time(patient_dict: Dict, anchor_date: datetime, years_back: int) -> Dict:
    """
    Remove all irrelevant history taking place after the anchor date and recast the remaining events in time relative to the anchor.
    Also add the anchor date as a new field.
    """
    start_date = anchor_date - timedelta(days=years_back * 365)
    processed_patient = {
        'patient_id':patient_dict['patient_id'],
        'demographics':patient_dict['demographics'],
        'anchor_date':anchor_date.strftime('%Y-%m-%d'),
        'active_medications':[],
        'encounters':[]
    }
    
    # Find all the encounters that land in our time window and grav their procedures and diagnoses
    # In my experience, oftentimes medications get repeated
    unique_medications = {}
    for encounter in patient_dict['encounters']:
        encounter_start_date_str = encounter['details']['start_visit']
        encounter_start_date = datetime.strptime(encounter_start_date_str, '%Y-%m-%d')
        if not (encounter_start_date > anchor_date or encounter_start_date < start_date):
            new_encounter = copy.deepcopy(encounter)
            del new_encounter['details']['start_visit']
            new_encounter['details']['start_visit'] = -(anchor_date - encounter_start_date).days
            del new_encounter['details']['end_visit']
            encounter_end_date_str = encounter['details']['end_visit']
            encounter_end_date = datetime.strptime(encounter_end_date_str, '%Y-%m-%d')
            new_encounter['details']['end_visit'] = -(anchor_date - encounter_end_date).days
            
            # Fix dates on procedures
            for proc in new_encounter['procedures']:
                if 'ProcedureStartInstant' in proc.keys():
                    if proc['ProcedureStartInstant'] is not None:
                        proc_start_date = datetime.strptime(proc['ProcedureStartInstant'], '%Y-%m-%d')
                        proc['proc_start_instant'] = -(anchor_date - proc_start_date).days
                    del proc['ProcedureStartInstant']
                if 'ProcedureEndInstant' in proc.keys():
                    if proc['ProcedureEndInstant'] is not None:
                        proc_end_date = datetime.strptime(proc['ProcedureEndInstant'], '%Y-%m-%d')
                        proc['proc_end_instant'] = -(anchor_date - proc_end_date).days
                    del proc['ProcedureEndInstant']
            
            del new_encounter['medications']
            
            deep_clean_dict(new_encounter)
            processed_patient['encounters'].append(new_encounter)
        
        # For each of those encounters, we still want to see any medications active during our window
        for med in encounter['medications']:
            med_key = (
                med["MedName"],
                med["MedStrength"],
                med["MedForm"],
                med["MedRoute"],
                med["MedFrequency"],
            )
            
            # The medication end may not be present or the medication may still be active
            if 'MedEndInstant' in med.keys() and med['MedEndInstant']:
                med_end_date = datetime.strptime(med['MedEndInstant'], '%Y-%m-%d')
            else:
                # Still ongoing
                med_end_date = datetime.max
                
            med_start_date = datetime.strptime(med['MedStartInstant'], '%Y-%m-%d')
            if med_end_date >= start_date and med_start_date <= anchor_date:
                # This medication was active during our time window of relevance
                start_instant = -(anchor_date - med_start_date).days
                
                if med_end_date == datetime.max:
                    # Still active
                    end_instant = sys.maxsize
                else:
                    end_instant = -(anchor_date - med_end_date).days
                
                if not med_key in unique_medications.keys():
                    unique_medications[med_key] = [[start_instant, end_instant]]
                else:
                    unique_medications[med_key].append([start_instant, end_instant])
        
    # Now for each medication, go through the time intervals and merge them
    for med_key, intervals in unique_medications.items():
        intervals.sort(key=lambda x:x[0]) # sort my starting time
        merged = []
        for interval in intervals:
            if len(merged) == 0 or merged[-1][1] + MED_INTERVAL_TOLERNACE < interval[0]:
                # New interval
                merged.append(interval)
            else:
                # Merge this interval
                merged[-1][1] = max(merged[-1][1], interval[1])
        unique_medications[med_key] = merged
    
    # Now that all the time frames for each medication have been merged PER medication, that defines our unique medications that we care about
    for med_key, disjoint_intervals in unique_medications.items():
        for interval in disjoint_intervals:
            processed_patient['active_medications'].append({
                'med_start_instant': interval[0],
                'med_end_instant': interval[1] if interval[1] < sys.maxsize else "ongoing",
                'med_name': med_key[0],
                'med_strength': med_key[1],
                'med_form': med_key[2],
                'med_route': med_key[3],
                'med_frequency': med_key[4]
            })
    
    # Now sort the encounters in reverse chronological order
    processed_patient['encounters'].sort(key=lambda x: x['details']['start_visit'], reverse=True)
    # Sort medications in reverse chronological order by starting date (since ending date may not be present)
    processed_patient['active_medications'].sort(key=lambda x: x['med_start_instant'], reverse=True)
    return processed_patient
    
if __name__ == "__main__":
    # Dry run test
    YEARS_BACK = 2
    from pathlib import Path
    test_file = Path(os.environ['PATIENT_JSON_DIR']) / "patient_700E678B71CAB827718AE59CC697A7C2.json"
    with open(test_file, 'r') as f_orig:
        patient_dict = json.load(f_orig)
        anchor_date = find_anchor_date(patient_dict)
        if anchor_date != None:
            print(f"Found anchor date: {anchor_date}")
            sliced_dict = slice_and_convert_time(patient_dict, anchor_date, YEARS_BACK)
            new_file = Path("test_data/patient_700E678B71CAB827718AE59CC697A7C2.json")
            with open(new_file, 'w') as f_new:
                json.dump(sliced_dict, f_new, indent=4)
        else:
            print(f"No anchor date found...")