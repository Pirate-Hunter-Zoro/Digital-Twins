from typing import Dict, Tuple, Optional
from datetime import datetime, timedelta
import copy
import pandas as pd

MED_OVERLAP_TOLERANCE = 1 # If one patient stops a medication and then this many days later restarts it, just shove it into one interval
WASHOUT = 180 # If a medication occurence is a candidate 'anchor date', it fails if another medication with the same ingredient occurred less than or equal to this many days before it
DEBUG = False

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
    
    med_intervals = {}
    sliced_med_intervals = {}
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
        for med in encounter['medications']:
            med_start_date = datetime.strptime(med['MedStartInstant'], '%Y-%m-%d')
            if med_start_date > anchor_date:
                # Med started in the future - not interested
                continue
            med_end_date_str = med.get('MedEndInstant')
            if med_end_date_str:
                med_end_date = min(anchor_date, datetime.strptime(med_end_date_str, '%Y-%m-%d'))
            else: # Ongoing
                med_end_date = anchor_date
            key = (
                med["MedName"],
                med["MedSimpleGenericName"],
                med["MedStrength"],
                med["MedForm"],
                med["MedRoute"],
                med["MedFrequency"],
            )
            med_start_num = -(anchor_date-med_start_date).days
            med_end_num = -(anchor_date-med_end_date).days
            # DEFINITELY include this medication interval to the unsliced patient json
            if key not in med_intervals.keys():
                med_intervals[key] = []
            med_intervals[key].append([med_start_num, med_end_num])
            
            if med_end_date > start_date:
                # This was an active medication within our window for the slicing
                if key not in sliced_med_intervals.keys():
                    sliced_med_intervals[key] = []
                sliced_med_intervals[key].append([med_start_num, med_end_num])
        
    # Now merge the intervals for both the sliced and unsliced versions of our active medications
    merge_and_add(med_intervals, processed_patient)
    merge_and_add(sliced_med_intervals, processed_sliced_patient)
    
    # Sort both medications and encounters by reverse chronological order
    processed_patient['encounters'].sort(key = lambda encounter: -encounter['details']['start_visit'])
    processed_patient['active_medications'].sort(key = lambda med: -med['MedStartInstant'])
    processed_sliced_patient['encounters'].sort(key = lambda encounter: -encounter['details']['start_visit'])
    processed_sliced_patient['active_medications'].sort(key = lambda med: -med['MedStartInstant'])
    
    return (processed_sliced_patient, processed_patient)