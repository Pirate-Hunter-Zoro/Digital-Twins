from typing import Dict
import os
from pathlib import Path
import random
import json
import numpy as np

from dotenv import load_dotenv
load_dotenv()

from scripts.data_loading.diagnoses_definitions import get_mdd_description, PSYCH_ARMS, MEDICAL_ARMS, SAFETY_ARMS
from scripts.data_loading.med_definitions import ALL_ARMS
from scripts.data_loading.features import (
    psych_comorbidity, 
    medical_comorbidity, 
    suicidality_flag,
    prior_adequate_trials,
    benzo_days,
    augmentation_flag,
    polypharmacy,
    nsaid_burden,
    psych_utilization,
    hypnotic_burden,
    somatic_treatment_flag,
    psychotherapy_count,
    safety_comorbidity,
    sud_specifics,
    get_sdoh,
    get_vitals_average,
)

YEARS_BACK = int(os.environ['YEARS_BACK'])

ATTRIBUTE_INDICES = {
    'mdd_to_anchor_days' : 0,
    'num_encounters': 1,
    'days_of_history': 2,
    'AgeInYears': 3,
    'bmi': 4,
    'bps': 5,
    'bp_dias': 6,
    'suicide_flag': 7,
    'somatic_flag': 8,
    'augmentation_occured': 9,
    'benzo_days_coverage': 10,
    'psychotherapy_count': 11,
    'polypharmacy_count': 12,
    'nsaid_count': 13,
    'hypnotics_burden': 14,
    'in_patient_days': 15,
    'num_emergency': 16,
}
for key in sorted(PSYCH_ARMS):
    ATTRIBUTE_INDICES[f"psych_{key}"] = len(ATTRIBUTE_INDICES)
for key in sorted(MEDICAL_ARMS):
    ATTRIBUTE_INDICES[f"medical_{key}"] = len(ATTRIBUTE_INDICES)
for key in sorted(SAFETY_ARMS):
    ATTRIBUTE_INDICES[f"safety_{key}"] = len(ATTRIBUTE_INDICES)
for key in sorted(ALL_ARMS):
    ATTRIBUTE_INDICES[f"trials_{key}"] = len(ATTRIBUTE_INDICES)

CATEGORICAL_FIELDS = ["Sex", "PreferredLanguage", "SexualOrientation", "MaritalStatus", "Religion", "SmokingStatus", "Race_Ethnicity"]
KNOWN_CATEGORIES = {k: set() for k in CATEGORICAL_FIELDS}
sliced_json_dir = Path(os.environ['SLICED_PATIENT_JSON_DIR'])
for sliced_json_file in sliced_json_dir.glob("*.json"):
    with open(sliced_json_file, 'r') as f:
        demographics = json.load(f)['demographics']
        for key in CATEGORICAL_FIELDS:
            val = demographics.get(key, None)
            if val != None and not (isinstance(val, float) and np.isnan(val)) and not (val not in KNOWN_CATEGORIES[key]):
                KNOWN_CATEGORIES[key].add(val)
                # TODO
                ATTRIBUTE_INDICES[f""] = len(ATTRIBUTE_INDICES)

def get_bool_int(val: bool) -> int:
    if val:
        return 1
    return 0

def generate_deterministic_vector(sliced_json: Dict) -> np.array:
    """Parse the sliced patient json to generate a deterministic vector to represent the patient

    Args:
        sliced_json (Dict): Anchor date going back a certain number of years
    Returns:
        np.array: Resulting vector for patient
    """
    # First check for pre-existence
    narrative_save_path = Path(os.environ['DETERMINISTIC_VECTORS_DIR']) / f"{sliced_json['patient_id']}.npy"
    if narrative_save_path.exists() and int(os.environ['SCRUB_DETERMINISTIC_VECTORS']) == 0:
        return (sliced_json['patient_id'], sliced_json['days_of_history'])
    
    demographics_of_interests = [
        "Sex",
        "PreferredLanguage",
        "AgeInYears",
        "SexualOrientation",
        "MaritalStatus",
        "Religion",
        "SmokingStatus",
        "Race_Ethnicity"
    ]
    
    psych_comorbidity_dict = psych_comorbidity(sliced_json)
    
    medical_comorbidity_dict = medical_comorbidity(sliced_json)
    
    # Various flags
    suicide_flag = suicidality_flag(sliced_json)
    somatic_flag = somatic_treatment_flag(sliced_json)
    
    augmentation_occured = augmentation_flag(sliced_json)
    
    # Treatment counts 
    adequate_trials_count = prior_adequate_trials(sliced_json)
    benzo_days_coverage = benzo_days(sliced_json)
    psychotherapy_treament_count = psychotherapy_count(sliced_json)
    
    # Burden
    hypnotics_burden_set = hypnotic_burden(sliced_json)
    safety_comorbidity_dict = safety_comorbidity(sliced_json)
    distinct_ingredients = polypharmacy(sliced_json)
    distinct_nsaid_ingredients = nsaid_burden(sliced_json) # Which have duration at least 7 days
    
    # Utilization
    in_patient_days, num_emergency = psych_utilization(sliced_json, YEARS_BACK)
    
    # Substance abuse
    sud_names_dict = sud_specifics(sliced_json)
    
    # Vitals
    vitals = get_vitals_average(sliced_json)
    bmi = vitals['bmi'] if vitals['bmi'] != "Missing" else np.nan
    bps = vitals['bp_sys'] if vitals['bp_sys'] != "Missing" else np.nan
    bp_dias = vitals['bp_dias'] if vitals['bp_dias'] != "Missing" else np.nan
    
    # Form the vector
    patient_vector = [
        sliced_json['mdd_to_anchor_days'],
        sliced_json['num_encounters'],
        sliced_json['days_of_history'],
        float(sliced_json['demographics']['AgeInYears']),
        bmi,
        bps,
        bp_dias,
        get_bool_int(suicide_flag),
        get_bool_int(somatic_flag),
        get_bool_int(augmentation_occured),
        benzo_days_coverage,
        psychotherapy_treament_count,
        len(distinct_ingredients),
        len(distinct_nsaid_ingredients),
        len(hypnotics_burden_set),
        in_patient_days,
        num_emergency
    ]

    # Now handle the comorbidities
    for key in sorted(psych_comorbidity_dict.keys()):
        patient_vector.append(int(psych_comorbidity_dict[key]))
    for key in sorted(medical_comorbidity_dict.keys()):
        patient_vector.append(int(medical_comorbidity_dict[key]))
    for key in sorted(safety_comorbidity_dict.keys()):
        patient_vector.append(int(safety_comorbidity_dict[key]))
    for key in sorted(adequate_trials_count.keys()):
        patient_vector.append(adequate_trials_count[key])

if __name__=="__main__":
    # Take a sample of produced narratives and put them in the local test_data directory
    # TODO - sample random narratives and their respective vectors
    pass