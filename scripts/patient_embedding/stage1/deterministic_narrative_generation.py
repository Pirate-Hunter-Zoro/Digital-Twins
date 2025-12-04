from typing import Dict
import json

from scripts.common.data_loading.fit_to_anchor import find_anchor_date, slice_and_convert_time
from scripts.common.data_loading.med_definitions import SSRI, BUPROPION
from scripts.common.data_loading.diagnoses_definitions import PTSD, ANXIETY
from scripts.common.data_loading.features import (
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
    sud_specifics
)

def get_bool_str(val: bool) -> str:
    if val:
        return "Present"
    return "Absent"

def generate_narrative(sliced_json: Dict, unsliced_json: Dict) -> str:
    demographics_of_interests = [
        "DepressionDiagnosis",
        "BipolarDiagnosis",
        "SchizophreniaDiagnosis",
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
    medical_comorbidity_dict = medical_comorbidity(unsliced_json) # We care about this EVER happening within our anchor
    
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
    in_patient_days_1_yr, num_emergency_1_yr = psych_utilization(sliced_json, 1)
    in_patient_days_3_yr, num_emergency_3_yr = psych_utilization(sliced_json, 3)
    
    # Substance abuse
    sud_names_dict = sud_specifics(sliced_json)
    
    # Form the narrative
    
    # Header
    condition = "MDD"
    baseline_window = "-730...0"
    HEADER = f"### COHORT & INDEX\nCondition: {condition} | Index date: {sliced_json['anchor_date']} | Baseline window: {baseline_window} days\n"
    
    # Demographics
    demographics = [f'{demographic}: {sliced_json["demographics"].get(demographic, "Missing")}' for demographic in demographics_of_interests]
    DEMOGRAPHICS = f"### DEMOGRAPHICS\n{' | '.join(demographics)}\n"
    
    # Psych history
    psych_comorbidities = [f'{psych_arm}: {get_bool_str(psych_comorbidity_dict[psych_arm])}' for psych_arm in psych_comorbidity_dict.keys()]
    PSYCH_HISTORY = f"### PSYCH HISTORY\n{' | '.join(psych_comorbidities)}\n"
    
    # Medical comorbidity
    med_comorbidities = [f'{med_arm}: {get_bool_str(medical_comorbidity_dict[med_arm])}' for med_arm in medical_comorbidity_dict.keys()]
    MED_HISTORY = f"### MEDICAL COMORBIDITY\n{' | '.join(med_comorbidities)}\n"
    
    # Treatment exposure
    TREAT_EXPOSURE = f"### TREATMENT EXPOSURE\nPrior adequate AD trials: {' | '.join([f'{arm}: {adequate_trials_count[arm]}' for arm in adequate_trials_count.keys()]) if len(adequate_trials_count) > 0 else 'Absent'}\n\
Benzodiazepine days (90d): {benzo_days_coverage}\n\
Hypnotics: {' | '.join([hypnotic for hypnotic in hypnotics_burden_set]) if len(hypnotics_burden_set) > 0 else 'Absent'}\n\
Augmentation: {get_bool_str(augmentation_occured)}\n\
Somatic treatments: {get_bool_str(somatic_flag)} | Psychotherapy visits (12m): {psychotherapy_treament_count if psychotherapy_treament_count > 0 else 'Absent'}\n"
        
    # Medication burden
    MED_BURDEN = f"### MEDICATION BURDEN\nActive meds at baseline: {len(distinct_ingredients)} ({', '.join([ingredient for ingredient in distinct_ingredients]) if len(distinct_ingredients) > 0 else 'Absent'})\n\
NSAID burden: {len(distinct_nsaid_ingredients)} ({', '.join([ingredient for ingredient in distinct_nsaid_ingredients]) if len(distinct_nsaid_ingredients) > 0 else 'Absent'})\n"
    
    # Specific substance abuse types
    substances = ' | '.join([sud_name for sud_name in sorted(list(sud_names_dict.keys())) if sud_names_dict[sud_name]])
    SUBSTANCE_ABUSE = f"### SUBSTANCE ABUSE\nSubstances abused by patient: {substances if len(substances) > 0 else 'None'}\n"
    
    # Utilization
    UTILIZATION = f"### UTILIZATION\nPsych inpatient days: {in_patient_days_1_yr} (12m) / {in_patient_days_3_yr} (3y) | ED psych visits: {num_emergency_1_yr} (12m) / {num_emergency_3_yr} (3y)\n"
        
    # Safety
    SAFETY = f"### SAFETY\n{' | '.join([f'{safety_arm}: {get_bool_str(safety_comorbidity_dict[safety_arm])}' for safety_arm in safety_comorbidity_dict.keys()])}\n"
    
    # Suicide
    SUICIDE = f"### SUICIDE FLAG (3y)\n{suicide_flag}\n"
    
    # Machine flag
    MACHINE_FLAG = f"AGE={sliced_json['demographics']['AgeInYears']};\
SEX={'F' if sliced_json['demographics']['Sex']=='Female' else 'M'};\
ANXIETY={'Y' if psych_comorbidity_dict[ANXIETY] else 'N'};\
PTSD={'Y' if psych_comorbidity_dict[PTSD] else 'N'};\
SUD={'Y' if len(substances) > 0 else 'N'};\
SUICIDALITY12M={'Y' if suicide_flag else 'N'};\
NSAID_CT={len(distinct_nsaid_ingredients)};\
POLYPHARM_CT={len(distinct_ingredients)};\
PRIOR_AD=SSRI:{adequate_trials_count.get(SSRI, 0)},BUP:{adequate_trials_count.get(BUPROPION, 0)};"
                    
    return "\n".join([HEADER, DEMOGRAPHICS, PSYCH_HISTORY, MED_HISTORY, TREAT_EXPOSURE, MED_BURDEN, SUBSTANCE_ABUSE, UTILIZATION, SAFETY, SUICIDE, MACHINE_FLAG])

if __name__ == "__main__":
    # Dry run test
    YEARS_BACK = 2
    from pathlib import Path
    test_files = list(Path("/media/studies/ehr_study/analysis/mferguson/patient_json/").glob("*.json"))
    anchor_found = False
    for i, test_file in enumerate(test_files):
        id = test_file.stem
        with open(test_file, 'r') as f_orig:
            patient_dict = json.load(f_orig)
            anchor_date, mdd_date = find_anchor_date(patient_dict)
            if anchor_date != None:
                print(f"Found anchor date: {anchor_date}")
                sliced_dict, unsliced_dict = slice_and_convert_time(patient_dict, anchor_date, mdd_date, YEARS_BACK)
                new_file = Path(f"test_data/sliced_{id}.json")
                unsliced_file = Path(f"test_data/unsliced_{id}.json")
                with open(new_file, 'w') as f_new:
                    json.dump(sliced_dict, f_new, indent=4)
                with open(unsliced_file, 'w') as f_unsliced:
                    json.dump(unsliced_dict, f_unsliced, indent=4)
                anchor_found = True
                
                # Now generate the narrative
                narrative_file = Path(f"test_data/narrative_{id}.md")
                with open(narrative_file, 'w') as f_narrative:
                    f_narrative.write(generate_narrative(sliced_dict, unsliced_dict))
                break
        if (i+1) % 1000 == 0:
            print(f"Searched for an anchor in {i+1} patients so far...")
    
    if not anchor_found:
        print("No anchor date found in any patients...")