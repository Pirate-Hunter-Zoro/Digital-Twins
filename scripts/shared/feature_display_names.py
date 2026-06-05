import re
from typing import Iterable

RAW_TO_DISPLAY: dict[str, str] = {
    # Numeric block (passed through StandardScaler verbatim; vitals dropped at load_data_set time)
    "mdd_to_anchor_days":     "MDD-to-anchor gap (days) [numeric]", # TODO: Index instead of anchor "Duration of MDD history prior to index"
    "num_encounters":         "Encounter count [numeric]", # TODO: Number of encounters
    "pre_anchor_history_days": "Pre-anchor history (days) [numeric]", # TODO: Duration of encounter history
    "AgeInYears":             "Age (years) [numeric]",
    "benzo_days_coverage":    "Anti-anxiety med days [numeric]", # TODO: Anxiolytic treatment days prior to index
    "polypharmacy_count":     "Active med count at baseline [numeric]", # TODO: Number of different medications at index
    "nsaid_count":            "Anti-inflammatory pain med count (NSAID) [numeric]", # TODO: Number of different anti-inflammatory medications at index
    "hypnotics_burden":       "Sleep medication count [numeric]", # TODO: Number of different .....
    "in_patient_days":        "Psych inpatient days [numeric]", # TODO: Total number of in-patient days prior to index
    "num_emergency":          "ED psych visit count [numeric]", # TODO: Total number of ED visits
    "trials_BUPROPION":       "Bupropion: full-length (6wk) course count [numeric]", # TODO: Number of Bupropion trials lasting at least six weeks or more
    "trials_MIRTAZAPINE":     "Mirtazapine: full-length (6wk) course count [numeric]", # TODO: Same
    "trials_SNRI":            "SNRI: full-length (6wk) course count [numeric]", # TODO: Same
    "trials_SSRI":            "SSRI: full-length (6wk) course count [numeric]", # TODO: Same
    "trials_VORTIOXETINE":    "Vortioxetine: full-length (6wk) course count [numeric]", # TODO: Same
    # Bool block (single-flag and multi-label indicators; passed through as int8)
    "mdd_within_window":                       "MDD diagnosed in history [boolean]",
    "suicide_flag":                            "Suicidality flagged [boolean]",
    "augmentation_occured":                    "Augmentation therapy used [boolean]",
    # Psychiatric comorbidities
    "psych_ADJUSTMENT_DISORDER":               "Adjustment disorder [boolean]",
    "psych_ANXIETY":                           "Anxiety disorder [boolean]",
    "psych_DYSTHYMIA":                         "Dysthymia (chronic depression) [boolean]",
    "psych_INSOMNIA":                          "Insomnia [boolean]",
    "psych_OCD":                               "OCD [boolean]",
    "psych_PTSD":                              "PTSD [boolean]",
    "psych_SOCIAL_ANXIETY":                    "Social anxiety disorder [boolean]",
    "psych_SUD":                               "Substance use disorder (any) [boolean]",
    # Medical comorbidities
    "medical_CHRONIC_PAIN":                    "Chronic pain [boolean]",
    "medical_DIABETES":                        "Diabetes [boolean]",
    "medical_HYPERLIPIDEMIA":                  "High cholesterol [boolean]",
    "medical_THYROID":                         "Thyroid disorder [boolean]",
    # Safety
    "safety_EPILEPSY":                         "Epilepsy [boolean]", # TODO: Comorbid seizure disorder
    "safety_UNCONTROLLED_HTN":                 "Uncontrolled high blood pressure [boolean]", # TODO: Comorbid uncontrolled hypertension
    # Substance use disorder by substance
    "sud_Alcohol":                             "Alcohol use disorder [boolean]",
    "sud_Cannabis":                            "Cannabis use disorder [boolean]",
    "sud_Cocaine":                             "Cocaine use disorder [boolean]",
    "sud_Hallucinogen":                        "Hallucinogen use disorder [boolean]",
    "sud_Inhalant":                            "Inhalant use disorder [boolean]",
    "sud_Nicotine":                            "Nicotine use disorder [boolean]",
    "sud_Opioid":                              "Opioid use disorder [boolean]",
    "sud_Other Stimulant":                     "Other stimulant use disorder [boolean]",
    "sud_Other Substance":                     "Other substance use disorder [boolean]",
    "sud_Sedative/Hypnotic":                   "Sedative/hypnotic use disorder [boolean]",
    # Social determinants of health         # TODO - how are these issues defined? Find out from Katie or Elizabeth
    "sdoh_Education/Literacy":                 "Education or literacy issue [boolean]", 
    "sdoh_Employment":                         "Employment issue [boolean]",
    "sdoh_Housing/Economic":                   "Housing or financial issue [boolean]",
    "sdoh_Legal/Crime/Other Psychosocial":     "Legal or criminal issue [boolean]",
    "sdoh_Occupational Exposure":              "Occupational hazard exposure [boolean]",
    "sdoh_Primary Support Group/Family":       "Family/support group issue [boolean]",
    "sdoh_Psychosocial Circumstances":         "Psychosocial circumstances [boolean]",
    "sdoh_Social Environment":                 "Social environment issue [boolean]",
    "sdoh_Upbringing":                         "Upbringing-related issue [boolean]",
    # Categorical one-hot block (OneHotEncoder(drop='if_binary', handle_unknown='ignore'))
    # Binary: Sex (drops reference 'Female', keeps 'Male'); other categories keep all levels.
    # TODO - try to make this less wordy - succinctly communicate category and which one
    # TODO - Find out about marital status - e.g. you could be married, but widowed prior - are multiple one-hot encodings possible?
    # TODO - Get rid of "one-hot"
    "Sex_Male":                                          "Sex: Male [one-hot]",
    "Sex_Female":                                        "Sex: Female [one-hot]",
    "PreferredLanguage_Asian and Pacific Island":        "Preferred language: Asian/Pacific Islander [one-hot]",
    "PreferredLanguage_English Only":                    "Preferred language: English [one-hot]",
    "PreferredLanguage_Other":                           "Preferred language: Other [one-hot]",
    "PreferredLanguage_Other Indo-European":             "Preferred language: Other Indo-European [one-hot]",
    "PreferredLanguage_Spanish":                         "Preferred language: Spanish [one-hot]",
    "MaritalStatus_Divorced":                            "Marital status: Divorced [one-hot]",
    "MaritalStatus_Never Married":                       "Marital status: Never married [one-hot]",
    "MaritalStatus_Now Married":                         "Marital status: Married [one-hot]",
    "MaritalStatus_Separated":                           "Marital status: Separated [one-hot]",
    "MaritalStatus_Widowed":                             "Marital status: Widowed [one-hot]",
    "Religion_Catholic":                                 "Religion: Catholic [one-hot]",
    "Religion_Non-Christian":                            "Religion: Non-Christian [one-hot]",
    "Religion_Orthodox":                                 "Religion: Orthodox [one-hot]",
    "Religion_Other/Unknown":                            "Religion: Other/Unknown [one-hot]",
    "Religion_Protestant":                               "Religion: Protestant [one-hot]",
    "SmokingStatus_Current Smoker":                      "Smoking: Current [one-hot]",
    "SmokingStatus_Former Smoker":                       "Smoking: Former [one-hot]",
    "SmokingStatus_Never Smoker":                        "Smoking: Never [one-hot]",
    "Race_Ethnicity_American Indian or Alaska Native":   "Race/Ethnicity: American Indian/Alaska Native [one-hot]",
    "Race_Ethnicity_Asian":                              "Race/Ethnicity: Asian [one-hot]",
    "Race_Ethnicity_Black or African American":          "Race/Ethnicity: Black/African American [one-hot]",
    "Race_Ethnicity_Hispanic or Latino":                 "Race/Ethnicity: Hispanic/Latino [one-hot]",
    "Race_Ethnicity_Multi-Race":                         "Race/Ethnicity: Multi-race [one-hot]",
    "Race_Ethnicity_Native Hawaiian or Other Pacific Islander": "Race/Ethnicity: Native Hawaiian/Pacific Islander [one-hot]",
    "Race_Ethnicity_White or Caucasian":                 "Race/Ethnicity: White/Caucasian [one-hot]",
    # MDD recurrence and severity (per diagnoses_definitions.py; not in categorical_levels.json)
    "mdd_recurrence_Single Episode":                     "MDD recurrence: Single episode [one-hot]",
    "mdd_recurrence_Recurrent":                          "MDD recurrence: Recurrent [one-hot]",
    "mdd_recurrence_Dysthymia":                          "MDD recurrence: Dysthymia (chronic depression) [one-hot]",
    "mdd_severity_Unspecified":                          "MDD severity: Unspecified [one-hot]",
    "mdd_severity_Mild":                                 "MDD severity: Mild [one-hot]",
    "mdd_severity_Moderate":                             "MDD severity: Moderate [one-hot]",
    "mdd_severity_Severe":                               "MDD severity: Severe [one-hot]",
    "mdd_severity_Psychotic":                            "MDD severity: Psychotic [one-hot]",
    "mdd_severity_Remission":                            "MDD severity: Remission [one-hot]",
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