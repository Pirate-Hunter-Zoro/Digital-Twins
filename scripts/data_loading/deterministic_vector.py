from typing import Dict
import os
from pathlib import Path
import json
import numpy as np

from dotenv import load_dotenv
load_dotenv()

from scripts.data_loading.diagnoses_definitions import PSYCH_ARMS, MEDICAL_ARMS, SAFETY_ARMS, SDOH_MAP
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
    get_vitals_average,
    get_sdoh
)

YEARS_BACK = int(os.environ['YEARS_BACK'])

ANALYSIS_DIR = Path(os.environ['ANALYSIS_DIR'])
PATIENT_ATTRIBUTES_PATH = ANALYSIS_DIR / "patient_attributes.json"

ATTRIBUTE_INDICES = {}
CATEGORICAL_FIELDS = ["Sex", "PreferredLanguage", "MaritalStatus", "Religion", "SmokingStatus", "Race_Ethnicity"]
SUD_SUBSTANCES = sorted(["Alcohol", "Opioid", "Cannabis", "Sedative/Hypnotic", "Cocaine", "Other Stimulant", "Hallucinogen", "Nicotine", "Inhalant", "Other Substance"])
MDD_RECURRENCES = sorted(["Single Episode", "Recurrent", "Dysthymia"])                                                                 
MDD_SEVERITIES = sorted(["Mild", "Moderate", "Severe", "Psychotic", "Remission", "Unspecified"]) 
SDOH_CATEGORIES = sorted(SDOH_MAP.values())
KNOWN_CATEGORIES = {k: set() for k in CATEGORICAL_FIELDS}

LANGUAGE_MAP = {
    "English":                                "English Only",
    "Spanish":                                "Spanish",
    "German":                                 "Other Indo-European",
    "Russian":                                "Other Indo-European",
    "Farsi; Persian":                         "Other Indo-European",
    "Greek":                                  "Other Indo-European",
    "Swedish":                                "Other Indo-European",
    "Turkish":                                "Other Indo-European",
    "Urdu Pakistan":                          "Other Indo-European",
    "Burmese":                                "Asian and Pacific Island",
    "Chinese, Cantonese (inc Toishanese)":    "Asian and Pacific Island",
    "Hmong":                                  "Asian and Pacific Island",
    "Korean":                                 "Asian and Pacific Island",
    "Mandarin":                               "Asian and Pacific Island",
    "Thai":                                   "Asian and Pacific Island",
    "Vietnamese":                             "Asian and Pacific Island",
    "Sign Language":                          "Other",
    "Other":                                  "Other",
}

MARITAL_MAP = {
    "Married":            "Now Married",
    "Significant Other":  "Now Married",
    "Single":             "Never Married",
    "Other":              "Never Married",
    "Divorced":           "Divorced",
    "Legally Separated":  "Separated",
    "Widowed":            "Widowed",
}

SMOKING_MAP = {
    "Every Day":                             "Current Smoker",
    "Some Days":                             "Current Smoker",
    "Heavy Smoker":                          "Current Smoker",
    "Light Smoker":                          "Current Smoker",
    "Smoker, Current Status Unknown":        "Current Smoker",
    "Former":                                "Former Smoker",
    "Never":                                 "Never Smoker",
    "Passive Smoke Exposure - Never Smoker": "Never Smoker",
}

RELIGION_MAP = {
    "Baptist":                        "Protestant",
    "Methodist":                      "Protestant",
    "Lutheran":                       "Protestant",
    "Presbyterian":                   "Protestant",
    "Episcopalian":                   "Protestant",
    "Pentecostal":                    "Protestant",
    "Assembly of God":                "Protestant",
    "Church Of Christ":               "Protestant",
    "Church Of God":                  "Protestant",
    "Free Will Baptist":              "Protestant",
    "Gospel":                         "Protestant",
    "Nazarene":                       "Protestant",
    "Non-Denominational":             "Protestant",
    "Seventh Day Adventist":          "Protestant",
    "Reorganized Latter Day Saints":  "Protestant",
    "Latter Day Saints":              "Protestant",
    "Anglican":                       "Protestant",
    "United Church Of Christ":        "Protestant",
    "Independent":                    "Protestant",
    "Catholic":                       "Catholic",
    "Orthodox":                       "Orthodox",
    "Jewish":                         "Non-Christian",
    "Muslim":                         "Non-Christian",
    "Buddhist":                       "Non-Christian",
    "Hindu":                          "Non-Christian",
    "Scientologist":                  "Non-Christian",
    "Christian":                      "Protestant",
    "Protestant":                     "Protestant",
    "Jehovah's Witness":              "Other/Unknown",
    "Other":                          "Other/Unknown",
    "Privacy Requested":              "Other/Unknown",
    "Unitarian":                      "Other/Unknown",
}

CATEGORICAL_MAPS = {
    "PreferredLanguage": LANGUAGE_MAP,
    "MaritalStatus":     MARITAL_MAP,
    "SmokingStatus":     SMOKING_MAP,
    "Religion":          RELIGION_MAP,
}

def initialize_attribute_indices():
    global ATTRIBUTE_INDICES
    global KNOWN_CATEGORIES
    # We need a function to write to a json recording what components are in each deterministic patient vector component
    if (not PATIENT_ATTRIBUTES_PATH.exists()) or (int(os.environ['SCRUB_DETERMINISTIC_VECTORS']) == 1 and ATTRIBUTE_INDICES == {}):
        # Not already done
        ATTRIBUTE_INDICES = {
            'mdd_to_anchor_days' : 0,
            'mdd_within_window': 1,
            'num_encounters': 2,
            'days_of_history': 3,
            'AgeInYears': 4,
            'bmi': 5,
            'bps': 6,
            'bp_dias': 7,
            'suicide_flag': 8,
            'somatic_flag': 9,
            'augmentation_occured': 10,
            'benzo_days_coverage': 11,
            'psychotherapy_count': 12,
            'polypharmacy_count': 13,
            'nsaid_count': 14,
            'hypnotics_burden': 15,
            'in_patient_days': 16,
            'num_emergency': 17,
        }
        for key in sorted(PSYCH_ARMS):
            ATTRIBUTE_INDICES[f"psych_{key}"] = len(ATTRIBUTE_INDICES)
        for key in sorted(MEDICAL_ARMS):
            ATTRIBUTE_INDICES[f"medical_{key}"] = len(ATTRIBUTE_INDICES)
        for key in sorted(SAFETY_ARMS):
            ATTRIBUTE_INDICES[f"safety_{key}"] = len(ATTRIBUTE_INDICES)
        for key in sorted(ALL_ARMS):
            ATTRIBUTE_INDICES[f"trials_{key}"] = len(ATTRIBUTE_INDICES)

        # One hot encodings for categorical fields
        sliced_json_dir = Path(os.environ['SLICED_PATIENT_JSON_DIR'])
        for sliced_json_file in sliced_json_dir.glob("*.json"):
            # Find every possible value for every field
            with open(sliced_json_file, 'r') as f:
                demographics = json.load(f)['demographics']
                for key in CATEGORICAL_FIELDS:
                    raw_val = demographics.get(key, None)
                    
                    if raw_val != None and not (isinstance(raw_val, float) and np.isnan(raw_val)):
                        if key in CATEGORICAL_MAPS.keys():
                            # We don't want the raw value - we want the value this maps to
                            value_map = CATEGORICAL_MAPS[key]
                            value = value_map.get(raw_val, None)
                            if value != None:  
                                KNOWN_CATEGORIES[key].add(value)
                        elif raw_val not in KNOWN_CATEGORIES[key]:
                            KNOWN_CATEGORIES[key].add(raw_val)
        # Indices of each category value
        for key in CATEGORICAL_FIELDS:
            for raw_val in sorted(KNOWN_CATEGORIES[key]):
                ATTRIBUTE_INDICES[f"{key}_{raw_val}"] = len(ATTRIBUTE_INDICES) 
               
        for substance in SUD_SUBSTANCES:
            ATTRIBUTE_INDICES[f"sud_{substance}"] = len(ATTRIBUTE_INDICES)
        
        for category in SDOH_CATEGORIES:
            ATTRIBUTE_INDICES[f"sdoh_{category}"] = len(ATTRIBUTE_INDICES)
        
        for recurrence in MDD_RECURRENCES:
            ATTRIBUTE_INDICES[f"mdd_rec_{recurrence}"] = len(ATTRIBUTE_INDICES)
            
        for severity in MDD_SEVERITIES:
            ATTRIBUTE_INDICES[f"mdd_sev_{severity}"] = len(ATTRIBUTE_INDICES)
                
        # Cache this record
        with open(PATIENT_ATTRIBUTES_PATH, 'w') as f:
            json.dump(ATTRIBUTE_INDICES, f, indent=4)
    elif ATTRIBUTE_INDICES == {}:
        with open(PATIENT_ATTRIBUTES_PATH, 'r') as f:
            ATTRIBUTE_INDICES = json.load(f)
            # Still need to grab the categorical fields
            for key in CATEGORICAL_FIELDS:
                # Grab everything preceding the '{key}_'
                attribute_val_keys = [label[len(key)+1:] for label in ATTRIBUTE_INDICES if label.startswith(f"{key}_")]
                KNOWN_CATEGORIES[key] = set(attribute_val_keys)

def get_bool_int(val: bool) -> int:
    if val:
        return 1
    return 0

def generate_deterministic_vector(sliced_json: Dict):
    """Parse the sliced patient json to generate a deterministic vector to represent the patient

    Args:
        sliced_json (Dict): Anchor date going back a certain number of years
    Returns:
        np.array: Resulting vector for patient
    """
    # First check for pre-existence
    initialize_attribute_indices()
    vector_save_path = Path(os.environ['DETERMINISTIC_VECTORS_DIR']) / f"{sliced_json['patient_id']}.npy"
    if vector_save_path.exists() and int(os.environ['SCRUB_DETERMINISTIC_VECTORS']) == 0:
        return np.load(vector_save_path, allow_pickle=True)
    
    psych_comorbidity_dict = psych_comorbidity(sliced_json)
    
    medical_comorbidity_dict = medical_comorbidity(sliced_json)
    
    patient_sdoh_flags = get_sdoh(sliced_json)
    
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
    
    # MDD recurrence and severity
    patient_mdd_rec, patient_mdd_sev = sliced_json['mdd_recurrence'], sliced_json['mdd_severity']
    
    # Vitals
    vitals = get_vitals_average(sliced_json)
    bmi = vitals['bmi'] if vitals['bmi'] != "Missing" else np.nan
    bps = vitals['bp_sys'] if vitals['bp_sys'] != "Missing" else np.nan
    bp_dias = vitals['bp_dias'] if vitals['bp_dias'] != "Missing" else np.nan
    
    # Form the vector
    patient_vector = [
        sliced_json['mdd_to_anchor_days'],
        get_bool_int(sliced_json['mdd_to_anchor_days'] <= 365*YEARS_BACK),
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
        
    # One-hot encoding demographics
    for field in CATEGORICAL_FIELDS:
        patient_val = sliced_json['demographics'].get(field, None)
        if field in CATEGORICAL_MAPS.keys():
            # Mapped value one hot encoding - change the patient's value to whatever value it maps to
            mapped_vals = CATEGORICAL_MAPS[field]
            patient_val = mapped_vals.get(patient_val, None)
        for val in sorted(KNOWN_CATEGORIES[field]):
            patient_vector.append(get_bool_int(patient_val == val))
            
    # One-hot encoding for SUD substances
    for substance in SUD_SUBSTANCES:
        if sud_names_dict.get(substance, False):
            patient_vector.append(1)
        else:
            patient_vector.append(0)
    
    # One-hot encoding for SDOH_CATEGORIES
    for sdoh_flag in SDOH_CATEGORIES:
        patient_vector.append(get_bool_int(sdoh_flag in patient_sdoh_flags))
    
    # One-hot encoding for MDD recurrence and severity 
    for rec in MDD_RECURRENCES:
        patient_vector.append(get_bool_int(rec == patient_mdd_rec))
    for sev in MDD_SEVERITIES:
        patient_vector.append(get_bool_int(sev == patient_mdd_sev))
    
    # Convert to numpy, save, and return
    patient_vector = np.array(patient_vector, dtype=np.float32)
    os.makedirs(vector_save_path.parent, exist_ok=True)
    np.save(vector_save_path, patient_vector)