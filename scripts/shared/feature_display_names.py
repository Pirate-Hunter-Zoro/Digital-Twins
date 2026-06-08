import re
from typing import Iterable

RAW_TO_DISPLAY: dict[str, str] = {
    # Numeric block (passed through StandardScaler verbatim; vitals dropped at load_data_set time)
    "mdd_to_anchor_days":     "MDD history before index (days)",    # numeric
    "num_encounters":         "Encounter count",                    # numeric
    "pre_anchor_history_days": "History length (days)",             # numeric
    "AgeInYears":             "Age (years)",                        # numeric
    "benzo_days_coverage":    "Anxiolytic days before index",       # numeric
    "polypharmacy_count":     "Medications active at index",        # numeric
    "nsaid_count":            "Anti-inflammatories active at index", # numeric
    "hypnotics_burden":       "Sleep meds active at index",         # numeric
    "in_patient_days":        "Inpatient days before index",        # numeric
    "num_emergency":          "ED visits (count)",                  # numeric
    "trials_BUPROPION":       "Bupropion trials (6+ wk)",           # numeric
    "trials_MIRTAZAPINE":     "Mirtazapine trials (6+ wk)",         # numeric
    "trials_SNRI":            "SNRI trials (6+ wk)",                # numeric
    "trials_SSRI":            "SSRI trials (6+ wk)",                # numeric
    "trials_VORTIOXETINE":    "Vortioxetine trials (6+ wk)",        # numeric
    # Bool block (single-flag and multi-label indicators; passed through as int8)
    "mdd_within_window":                       "MDD in history",     # boolean
    "suicide_flag":                            "Suicidality",        # boolean
    "augmentation_occured":                    "Augmentation therapy", # boolean
    # Psychiatric comorbidities
    "psych_ADJUSTMENT_DISORDER":               "Adjustment disorder",            # boolean
    "psych_ANXIETY":                           "Anxiety disorder",               # boolean
    "psych_DYSTHYMIA":                         "Dysthymia (chronic depression)", # boolean
    "psych_INSOMNIA":                          "Insomnia",                       # boolean
    "psych_OCD":                               "OCD",                            # boolean
    "psych_PTSD":                              "PTSD",                           # boolean
    "psych_SOCIAL_ANXIETY":                    "Social anxiety disorder",        # boolean
    "psych_SUD":                               "Substance use disorder (any)",   # boolean
    # Medical comorbidities
    "medical_CHRONIC_PAIN":                    "Chronic pain",                   # boolean
    "medical_DIABETES":                        "Diabetes",                       # boolean
    "medical_HYPERLIPIDEMIA":                  "High cholesterol",               # boolean
    "medical_THYROID":                         "Thyroid disorder",               # boolean
    # Safety
    "safety_EPILEPSY":                         "Seizure disorder",          # boolean
    "safety_UNCONTROLLED_HTN":                 "Uncontrolled hypertension", # boolean
    # Substance use disorder by substance
    "sud_Alcohol":                             "Alcohol use disorder",           # boolean
    "sud_Cannabis":                            "Cannabis use disorder",          # boolean
    "sud_Cocaine":                             "Cocaine use disorder",           # boolean
    "sud_Hallucinogen":                        "Hallucinogen use disorder",      # boolean
    "sud_Inhalant":                            "Inhalant use disorder",          # boolean
    "sud_Nicotine":                            "Nicotine use disorder",          # boolean
    "sud_Opioid":                              "Opioid use disorder",            # boolean
    "sud_Other Stimulant":                     "Other stimulant use disorder",   # boolean
    "sud_Other Substance":                     "Other substance use disorder",   # boolean
    "sud_Sedative/Hypnotic":                   "Sedative/hypnotic use disorder", # boolean
    # Social determinants of health
    "sdoh_Education/Literacy":                 "Education or literacy issue",    # boolean
    "sdoh_Employment":                         "Employment issue",               # boolean
    "sdoh_Housing/Economic":                   "Housing or financial issue",     # boolean
    "sdoh_Legal/Crime/Other Psychosocial":     "Legal or criminal issue",        # boolean
    "sdoh_Occupational Exposure":              "Occupational hazard exposure",   # boolean
    "sdoh_Primary Support Group/Family":       "Family/support group issue",     # boolean
    "sdoh_Psychosocial Circumstances":         "Psychosocial circumstances",     # boolean
    "sdoh_Social Environment":                 "Social environment issue",       # boolean
    "sdoh_Upbringing":                         "Upbringing issue",               # boolean
    # Categorical one-hot block (OneHotEncoder(drop='if_binary', handle_unknown='ignore'))
    # Binary: Sex (drops reference 'Female', keeps 'Male'); other categories keep all levels.
    "Sex_Male":                                          "Sex: Male",      # one-hot
    "PreferredLanguage_Asian and Pacific Island":        "Language: Asian/Pacific Islander",  # one-hot
    "PreferredLanguage_English Only":                    "Language: English",                 # one-hot
    "PreferredLanguage_Other":                           "Language: Other",                   # one-hot
    "PreferredLanguage_Other Indo-European":             "Language: Other Indo-European",     # one-hot
    "PreferredLanguage_Spanish":                         "Language: Spanish",                 # one-hot
    "MaritalStatus_Divorced":                            "Marital status: Divorced",      # one-hot
    "MaritalStatus_Never Married":                       "Marital status: Never married", # one-hot
    "MaritalStatus_Now Married":                         "Marital status: Married",       # one-hot
    "MaritalStatus_Separated":                           "Marital status: Separated",     # one-hot
    "MaritalStatus_Widowed":                             "Marital status: Widowed",       # one-hot
    "Religion_Catholic":                                 "Religion: Catholic",            # one-hot
    "Religion_Non-Christian":                            "Religion: Non-Christian",       # one-hot
    "Religion_Orthodox":                                 "Religion: Orthodox",            # one-hot
    "Religion_Other/Unknown":                            "Religion: Other/Unknown",       # one-hot
    "Religion_Protestant":                               "Religion: Protestant",          # one-hot
    "SmokingStatus_Current Smoker":                      "Smoking: Current",              # one-hot
    "SmokingStatus_Former Smoker":                       "Smoking: Former",               # one-hot
    "SmokingStatus_Never Smoker":                        "Smoking: Never",                # one-hot
    "Race_Ethnicity_American Indian or Alaska Native":   "Race/Ethnicity: American Indian/Alaska Native",      # one-hot
    "Race_Ethnicity_Asian":                              "Race/Ethnicity: Asian",                              # one-hot
    "Race_Ethnicity_Black or African American":          "Race/Ethnicity: Black/African American",             # one-hot
    "Race_Ethnicity_Hispanic or Latino":                 "Race/Ethnicity: Hispanic/Latino",                    # one-hot
    "Race_Ethnicity_Multi-Race":                         "Race/Ethnicity: Multi-race",                         # one-hot
    "Race_Ethnicity_Native Hawaiian or Other Pacific Islander": "Race/Ethnicity: Native Hawaiian/Pacific Islander", # one-hot
    "Race_Ethnicity_White or Caucasian":                 "Race/Ethnicity: White/Caucasian",                    # one-hot
    # MDD recurrence and severity (per diagnoses_definitions.py; not in categorical_levels.json)
    # Blank recurrence level (get_mdd_components emits "") is labeled Unspecified at the display
    # boundary only — the source field is left as-is to preserve the existing embedding cache.
    "mdd_recurrence_":                                   "MDD recurrence: Unspecified",                  # one-hot
    "mdd_recurrence_Single Episode":                     "MDD recurrence: Single episode",               # one-hot
    "mdd_recurrence_Recurrent":                          "MDD recurrence: Recurrent",                    # one-hot
    "mdd_recurrence_Dysthymia":                          "MDD recurrence: Dysthymia (chronic depression)", # one-hot
    "mdd_severity_Unspecified":                          "MDD severity: Unspecified",                    # one-hot
    "mdd_severity_Mild":                                 "MDD severity: Mild",                           # one-hot
    "mdd_severity_Moderate":                             "MDD severity: Moderate",                       # one-hot
    "mdd_severity_Severe":                               "MDD severity: Severe",                         # one-hot
    "mdd_severity_Psychotic":                            "MDD severity: Psychotic",                      # one-hot
    "mdd_severity_Remission":                            "MDD severity: Remission",                      # one-hot
}

PREFIX_PATTERN = re.compile(r"^(num__|cat__|bool__)")

def humanize_feature_names(raw_names: Iterable[str]) -> list[str]:
    """Without mutating the input, return the humanized names of the raw names associated with the input

    Args:
        raw_names (Iterable[str]): Names from a dataframe

    Returns:
        list[str]: Humanized respective names
    """
    out = []
    for name in raw_names:
        stripped = re.sub(PREFIX_PATTERN, "", name)
        out.append(RAW_TO_DISPLAY.get(stripped, stripped))
    return out
