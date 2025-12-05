from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import copy
import sys
import pandas as pd

from scripts.common.data_loading.diagnoses_definitions import get_diagnosis_arm, MDD
from scripts.common.data_loading.med_definitions import ALL_ARM_INGREDIENTS

MED_INTERVAL_TOLERANCE = 1 # If one patient stops a medication and then this many days later restarts it, just shove it into one interval
WASHOUT = 180
DEBUG = False

def find_anchor_date(patient_dict: Dict) -> Optional[Tuple[datetime, datetime]]:
    """
    Helper function to find the first instance of one of the ALL_ARM_INGREDIENTS medications that corresponds to an MDD diagnosis, and return both dates
    """
    # First find the patient's first diagnosis of MDD
    mdd_dates = []
    for encounter in patient_dict['encounters']:
        diagnosis_date = datetime.strptime(encounter['details']['start_visit'], '%Y-%m-%d')
        for diagnosis in encounter['diagnoses']:
            for code in diagnosis['codes']:
                if get_diagnosis_arm(code['code']) == MDD:
                    mdd_dates.append(diagnosis_date)
    # Sort mdd diagnoses chronologically
    mdd_dates.sort()
    
    # Each MDD diagnosis may have medications attached to it
    med_dates_map = {mdd_date : [] for mdd_date in mdd_dates}
    
    all_meds = {}
    if DEBUG:
        print(f"Patient {patient_dict['patient_id']} has MDD diagnosis dates: {[date.strftime('%Y-%m-%d') for date in mdd_dates]}")
    for encounter in patient_dict['encounters']:
        for medication in encounter['medications']:
            # Grab the med's name and see if it matches
            for arm_ingredient in ALL_ARM_INGREDIENTS:
                if arm_ingredient in medication['MedName'].lower():
                    start_date = datetime.strptime(medication['MedStartInstant'], '%Y-%m-%d')
                    if arm_ingredient not in all_meds.keys():
                        all_meds[arm_ingredient] = []
                    all_meds[arm_ingredient].append(start_date)
                    # Find the window of mdd_dates that this is in the middle of
                    left = 0
                    right = len(mdd_dates)
                    while left < right:
                        mid = (left + right) // 2
                        if mdd_dates[mid] > start_date:
                            # Later MDD diagnosis than this start date
                            right = mid
                        elif mdd_dates[mid] < start_date and (mid >= len(mdd_dates) - 1 or mdd_dates[mid+1] < start_date):
                            # Earlier MDD diagnosis than this start date, BUT not the latest possible earlier MDD diagnosis
                            left = mid+1
                        else:
                            med_dates_map[mdd_dates[mid]].append((start_date, arm_ingredient))
                            break
                        
    # For each arm ingredient, sort the occurences
    for dates in all_meds.values():
        dates.sort()
            
    # We want to find the earliest instance of an MDD diagnosis which also has an associated arm medication - and return the date of the earliest said arm medication
    anchor_date = None
    respective_mdd_date = None
    if DEBUG:
        print(f"Patient {patient_dict['patient_id']} has medication dates map:")
        for mdd_date in med_dates_map.keys():
            print(f"  MDD date {mdd_date.strftime('%Y-%m-%d')}: {[date.strftime('%Y-%m-%d') + ' (' + ingredient + ')' for date, ingredient in med_dates_map[mdd_date]]}")
    for mdd_date in mdd_dates:
        if len(med_dates_map[mdd_date]) > 0:
            # Linear search for the earliest med date
            earliest = None
            for date, ingredient in med_dates_map[mdd_date]:
                other_ingredient_dates = all_meds[ingredient]
                # Ensure this occurence of the ingredient was not preceded by another occurence of this ingredient within 90 days
                left = 0
                right = len(other_ingredient_dates)
                # In this binary search, we WILL find the exact date
                while left < right:
                    mid = (left + right) // 2
                    if other_ingredient_dates[mid] == date:
                        # Found the date
                        if mid == 0 or ((date - other_ingredient_dates[mid-1]).days > WASHOUT):
                            # Valid - this is a standalone occurence of the ingredient
                            if earliest is None or date < earliest:
                                earliest = date
                        break
                    elif other_ingredient_dates[mid] < date:
                        # Look right
                        left = mid+1
                    else:
                        # Look left
                        right = mid
            anchor_date = earliest
            respective_mdd_date = mdd_date
            break
        
    return (anchor_date, respective_mdd_date)

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

def merge_intervals(unique_meds: dict):
    for med_key, intervals in unique_meds.items():
        intervals.sort(key=lambda x:x[0]) # sort my starting time
        merged = []
        for interval in intervals:
            if len(merged) == 0 or merged[-1][1] + MED_INTERVAL_TOLERANCE < interval[0]:
                # New interval
                merged.append(interval)
            else:
                # Merge this interval
                merged[-1][1] = max(merged[-1][1], interval[1])
        unique_meds[med_key] = merged

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
    
    # Find all the encounters that land in our time window and grav their procedures and diagnoses
    # In my experience, oftentimes medications get repeated
    unique_medications_sliced = {}
    unique_medications = {}
    for encounter in patient_dict['encounters']:
        encounter_start_date_str = encounter['details']['start_visit']
        encounter_start_date = datetime.strptime(encounter_start_date_str, '%Y-%m-%d')
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
        if encounter_start_date < anchor_date:
            # Not in the future
            processed_patient['encounters'].append(new_encounter)
            if encounter_start_date >= start_date:
                # Also in the time frame for the sliced patient json
                processed_sliced_patient['encounters'].append(new_encounter)
        
        # For each of those encounters, we still want to see any medications active during our window
        for med in encounter['medications']:
            med_key = (
                med["MedName"],
                med["MedSimpleGenericName"].lower() if isinstance(med["MedSimpleGenericName"], str) else None,
                med["MedStrength"],
                med["MedForm"],
                med["MedRoute"],
                med["MedFrequency"],
            )
            
            # The medication end may not be present or the medication may still be active
            if 'MedEndInstant' in med.keys() and isinstance(med['MedEndInstant'], str):
                med_end_date = datetime.strptime(med['MedEndInstant'], '%Y-%m-%d')
            else:
                # Still ongoing
                med_end_date = datetime.max
                
            med_start_date = datetime.strptime(med['MedStartInstant'], '%Y-%m-%d')
            start_instant = -(anchor_date - med_start_date).days
                
            if med_end_date == datetime.max:
                # Still active
                end_instant = sys.maxsize
            else:
                end_instant = -(anchor_date - med_end_date).days
            
            if med_start_date <= anchor_date:
                # Not started in the future
                if not med_key in unique_medications.keys():
                    unique_medications[med_key] = [[start_instant, end_instant]]
                else:
                    unique_medications[med_key].append([start_instant, end_instant])
                
                if med_end_date >= start_date and med_start_date <= anchor_date:
                    # This medication was active during our sliced time window of relevance
                    if not med_key in unique_medications_sliced.keys():
                        unique_medications_sliced[med_key] = [[start_instant, end_instant]]
                    else:
                        unique_medications_sliced[med_key].append([start_instant, end_instant])
        
    # Now for each medication, go through the time intervals and merge them
    merge_intervals(unique_medications)
    merge_intervals(unique_medications_sliced)
    
    # Now that all the time frames for each medication have been merged PER medication, that defines our unique medications that we care about
    # Note that if a medication ends after our anchor date, as far as we are concernred, that's "ongoing" if we're only looking up to the anchor date
    for med_key, disjoint_intervals in unique_medications_sliced.items():
        for interval in disjoint_intervals:
            processed_sliced_patient['active_medications'].append({
                'med_start_instant': interval[0],
                'med_end_instant': interval[1] if interval[1] < 0 else "ongoing",
                'med_name': med_key[0],
                'med_simple_generic_name': med_key[1],
                'med_strength': med_key[2],
                'med_form': med_key[3],
                'med_route': med_key[4],
                'med_frequency': med_key[5]
            })
    # Also the unsliced patient data
    for med_key, disjoint_intervals in unique_medications.items():
        for interval in disjoint_intervals:
            processed_patient['active_medications'].append({
                'med_start_instant': interval[0],
                'med_end_instant': interval[1] if interval[1] < 0 else "ongoing",
                'med_name': med_key[0],
                'med_simple_generic_name': med_key[1],
                'med_strength': med_key[2],
                'med_form': med_key[3],
                'med_route': med_key[4],
                'med_frequency': med_key[5]
            })
    
    # Now sort the encounters in reverse chronological order
    processed_sliced_patient['encounters'].sort(key=lambda x: x['details']['start_visit'], reverse=True)
    processed_patient['encounters'].sort(key=lambda x: x['details']['start_visit'], reverse=True)
    # Sort medications in reverse chronological order by starting date (since ending date may not be present)
    processed_sliced_patient['active_medications'].sort(key=lambda x: x['med_start_instant'], reverse=True)
    processed_patient['active_medications'].sort(key=lambda x: x['med_start_instant'], reverse=True)
    return (processed_sliced_patient, processed_patient)