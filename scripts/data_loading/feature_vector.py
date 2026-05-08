from typing import Dict
import os
from pathlib import Path
import json
import numpy as np
import pandas as pd

from dotenv import load_dotenv
load_dotenv()

from scripts.data_loading.diagnoses_definitions import PSYCH_ARMS, MEDICAL_ARMS, SAFETY_ARMS, SDOH_MAP
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
CATEGORICAL_LEVELS_PATH = ANALYSIS_DIR / "categorical_levels.json"

CATEGORICAL_FIELDS = ["Sex", "PreferredLanguage", "MaritalStatus", "Religion", "SmokingStatus", "Race_Ethnicity"]
SUD_SUBSTANCES = sorted(["Alcohol", "Opioid", "Cannabis", "Sedative/Hypnotic", "Cocaine", "Other Stimulant", "Hallucinogen", "Nicotine", "Inhalant", "Other Substance"])
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

def initialize_categorical_levels():
    global KNOWN_CATEGORIES
    # We need a function to write to a json recording what components are in each patient's feature vector
    if (not CATEGORICAL_LEVELS_PATH.exists()) or (int(os.environ['SCRUB_FEATURE_VECTORS']) == 1):
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
        # Now that KNOWN_CATEGORIES is done
        res = {
            field: sorted(list(KNOWN_CATEGORIES[field]))
            for field in KNOWN_CATEGORIES
        }
        with open(CATEGORICAL_LEVELS_PATH, 'w') as f:
            json.dump(res, f, indent=4)
    else:
        with open(CATEGORICAL_LEVELS_PATH, 'r') as f:
            res = json.load(f)
            for field in CATEGORICAL_FIELDS:
                KNOWN_CATEGORIES[field] = set(res[field])
                
def generate_feature_vector(sliced_json: Dict) -> pd.Series:
    """Parse the sliced patient json to generate a feature vector to represent the patient

    Args:
        sliced_json (Dict): Anchor date going back a certain number of years

    Returns:
        pd.Series: Resulting feature vector
    """
    # First check for pre-existence
    initialize_categorical_levels()
     
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
    row = {
        "mdd_to_anchor_days": float(sliced_json['mdd_to_anchor_days']),
        "num_encounters": float(sliced_json["num_encounters"]),
        "days_of_history": float(sliced_json["days_of_history"]),
        "AgeInYears": float(sliced_json["demographics"]["AgeInYears"]),
        "bmi": bmi,
        "bp_sys": bps,
        "bp_dias": bp_dias,
        "benzo_days_coverage": float(benzo_days_coverage),
        "psychotherapy_count": float(psychotherapy_treament_count),
        "polypharmacy_count": float(len(distinct_ingredients)),
        "nsaid_count": float(len(distinct_nsaid_ingredients)),
        "hypnotics_burden": float(len(hypnotics_burden_set)),
        "in_patient_days": float(in_patient_days),
        "num_emergency": float(num_emergency)
    }
    
    for arm in sorted(adequate_trials_count.keys()):
        row[f"trials_{arm}"] = float(adequate_trials_count[arm])

    row["mdd_within_window"] = bool(sliced_json["mdd_to_anchor_days"] <= 365 * YEARS_BACK)
    row["suicide_flag"] = bool(suicide_flag)
    row["augmentation_occured"] = bool(augmentation_occured)
    row["somatic_flag"] = bool(somatic_flag)

    # Various comorbidities
    for arm in sorted(PSYCH_ARMS):
        row[f"psych_{arm}"] = bool(psych_comorbidity_dict.get(arm, False))
    for arm in sorted(MEDICAL_ARMS):
        row[f"medical_{arm}"] = bool(medical_comorbidity_dict.get(arm, False))
    for arm in sorted(SAFETY_ARMS):
        row[f"safety_{arm}"] = bool(safety_comorbidity_dict.get(arm, False))

    # Substance abuse
    for substance in SUD_SUBSTANCES:
        row[f"sud_{substance}"] = bool(sud_names_dict.get(substance, False))
    
    # Social determinants of health
    for category in SDOH_CATEGORIES:
        row[f"sdoh_{category}"] = bool(category in patient_sdoh_flags)

    for field in CATEGORICAL_FIELDS:
        # Missing, mapped, unmapped
        raw_val = sliced_json['demographics'].get(field)
        if raw_val is None or (isinstance(raw_val, float) and np.isnan(raw_val)):
            row[field] = np.nan
        elif field in CATEGORICAL_MAPS:
            # Whatever value this patient has corresponds to a broader category
            compressed_val = CATEGORICAL_MAPS[field].get(raw_val, np.nan)
            row[field] = compressed_val
        else:
            # Unmapped value like Sex
            row[field] = raw_val

    # MDD recurrence and severity
    row['mdd_recurrence'] = sliced_json['mdd_recurrence']
    row['mdd_severity'] = sliced_json['mdd_severity']
    
    return pd.Series(row)