from typing import Dict, Optional
from datetime import datetime, timedelta
import copy
import os

from dotenv import load_dotenv
load_dotenv()

from scripts.data_loading.diagnoses_definitions import get_mdd_components

MED_OVERLAP_TOLERANCE = 1 # If one patient stops a medication and then this many days later restarts it, just shove it into one interval
YEARS_BACK = int(os.environ['YEARS_BACK'])

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

def slice_and_convert_time(patient_dict: Dict, anchor_date: datetime) -> Optional[Dict]:
    """Verify that the patient has an MDD diagnosis prior to their anchor date, and that their history extends the required number of years back or more, and if so return their sliced record

    Args:
        patient_dict (Dict): Raw json of the patient
        anchor_date (datetime): Information on the antidepressant anchor date of the patient

    Returns:
        Optional[Dict]: Sliced record of the patient or null if they do not meet the window criteria
    """
    # First loop through every encounter and look for an MDD diagnosis which precedes or occurs at the same time as the anchor date
    mdd_prereq = False
    earliest_sliced_encounter_date = anchor_date
    latest_mdd_date = None
    latest_mdd_recurrence = None
    latest_mdd_severity = None
    for encounter in patient_dict['encounters']:
        info = encounter['details']
        enc_start_str = info.get('start_visit')
        if not isinstance(enc_start_str, str):
            continue # Skip bad encounters
        encounter_start_date = datetime.strptime(enc_start_str, '%Y-%m-%d')
        if encounter_start_date <= anchor_date:
            # Potential MDD preceding anchor date
            if encounter_start_date < earliest_sliced_encounter_date:
                # Update record for earliest encounter date
                earliest_sliced_encounter_date = encounter_start_date
            # Still need to check diagnoses for codes
            for diagnosis in encounter['diagnoses']:
                for code_dict in diagnosis['codes']:
                    mdd_rec, mdd_sev = get_mdd_components(code_dict['code'])
                    if mdd_rec != None:
                        mdd_prereq = True
                        if latest_mdd_date == None or latest_mdd_date < encounter_start_date:
                            latest_mdd_date = encounter_start_date
                            latest_mdd_recurrence, latest_mdd_severity = mdd_rec, mdd_sev
                            
                            
    # If no MDD diagnosis occurs before anchor date
    if not mdd_prereq:
        return None
    mdd_to_anchor_days = (anchor_date - latest_mdd_date).days
    
    # Define cutoff date
    cutoff_date = anchor_date - timedelta(int(os.environ['YEARS_BACK'])*365)
    if earliest_sliced_encounter_date > cutoff_date:
        # Not enough patient history
        return None
    
    # Now process the patient json
    processed_sliced_patient = {
        'patient_id': patient_dict['patient_id'],
        'demographics': patient_dict['demographics'],
        'anchor_date': anchor_date.strftime('%Y-%m-%d'),
        'mdd_to_anchor_days': mdd_to_anchor_days,
        'mdd_recurrence': latest_mdd_recurrence,
        'mdd_severity': latest_mdd_severity,
        'active_medications': [],
        'encounters': []
    }
    
    sliced_med_intervals = {}
    
    for encounter in patient_dict['encounters']:
        sliced_encounter = {'details': encounter['details'], 'procedures': [], 'diagnoses': encounter['diagnoses'], 'vitals': encounter['vitals'],}
        info = sliced_encounter['details']
        
        enc_start_str = info.get('start_visit')
        if not isinstance(enc_start_str, str):
            continue # Skip bad encounters
            
        encounter_start_date = datetime.strptime(enc_start_str, '%Y-%m-%d')
        
        enc_end_str = info.get('end_visit')
        if isinstance(enc_end_str, str):
            encounter_end_date = datetime.strptime(enc_end_str, '%Y-%m-%d')
        else:
            encounter_end_date = encounter_start_date

        if encounter_end_date < cutoff_date:
            # Ancient history
            continue
        if encounter_start_date > anchor_date:
            # Future
            continue
        
        encounter_start_date = max(encounter_start_date, cutoff_date)
        encounter_end_date = min(encounter_end_date, anchor_date)
        
        start_offset = -(anchor_date - encounter_start_date).days
        end_offset = -(anchor_date - encounter_end_date).days
        info['start_visit'] = start_offset
        info['end_visit'] = min(0, end_offset)
        
        for procedure in encounter['procedures']:
            procedure_copy = copy.deepcopy(procedure)
            proc_start_str = procedure_copy.get('ProcedureStartInstant')
            if isinstance(proc_start_str, str):
                procedure_start = datetime.strptime(proc_start_str, '%Y-%m-%d')
                if procedure_start > anchor_date:
                    # Future
                    continue
                
                # Handle End Date
                proc_end_str = procedure_copy.get('ProcedureEndInstant')
                if isinstance(proc_end_str, str):
                    procedure_end = datetime.strptime(proc_end_str, '%Y-%m-%d')
                else:
                    procedure_end = procedure_start
                    
                if procedure_end < cutoff_date:
                    # Ancient history
                    continue
                
                # Truncate to window
                procedure_start = max(procedure_start, cutoff_date)
                procedure_end = min(procedure_end, anchor_date)
                
                procedure_copy['ProcedureStartInstant'] = -(anchor_date - procedure_start).days
                procedure_copy['ProcedureEndInstant'] = -(anchor_date - procedure_end).days
            
                sliced_encounter['procedures'].append(procedure_copy)
                
        if encounter_start_date <= anchor_date:
            processed_sliced_patient['encounters'].append(sliced_encounter)
            
        for med in encounter['medications']:
            # --- Verify Medication Dates ---
            med_start_str = med.get('MedStartInstant')
            if not isinstance(med_start_str, str):
                continue
            
            med_start_date = datetime.strptime(med_start_str, '%Y-%m-%d')
            if med_start_date > anchor_date:
                # Started in future
                continue
                
            med_end_date_str = med.get('MedEndInstant')
            if isinstance(med_end_date_str, str):
                med_end_date = datetime.strptime(med_end_date_str, '%Y-%m-%d')
            else:
                med_end_date = anchor_date
            if med_end_date < cutoff_date:
                # Not active during window
                continue
            
            med_start_date = max(med_start_date, cutoff_date)
            med_end_date = min(med_end_date, anchor_date)
                
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
            
            if key not in sliced_med_intervals: sliced_med_intervals[key] = []
            sliced_med_intervals[key].append([med_start_num, min(0, med_end_num)])
        
    merge_and_add(sliced_med_intervals, processed_sliced_patient)
    
    # Sort
    processed_sliced_patient['encounters'].sort(key=lambda x: -x['details']['start_visit'])
    processed_sliced_patient['active_medications'].sort(key=lambda x: -x['MedStartInstant'])
    
    # Total chronological length
    timespan_sliced = (anchor_date - earliest_sliced_encounter_date).days + 1
    # We still want to record the actual length of history observed
    processed_sliced_patient['days_of_history'] = timespan_sliced
    # We also want the number of visits within the time window
    processed_sliced_patient['num_encounters'] = len(processed_sliced_patient['encounters'])
    
    return processed_sliced_patient

if __name__=="__main__":
    from pathlib import Path
    json_path = Path(os.environ['SLICED_PATIENT_JSON_DIR'])
    ids = []
    record_every = 1000
    done = 0
    for json_file in json_path.glob("*.json"):
        ids.append(json_file.stem)
        done += 1
        if done % record_every == 0:
            print(f"Scanned {done} patient json files...", flush=True)
    with open(Path("test_data/valid_ids.txt"), 'w') as f:
        f.write("\n".join(ids))