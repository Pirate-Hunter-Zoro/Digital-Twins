from typing import Dict
import os
from pathlib import Path
import random

from dotenv import load_dotenv
load_dotenv()

from scripts.data_loading.diagnoses_definitions import get_mdd_description
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

def get_bool_str(val: bool) -> str:
    if val:
        return "Present"
    return "Absent"

def generate_deterministic_narrative(sliced_json: Dict) -> tuple[str, int]:
    """Parse the sliced patient json to generate a deterministic markdown file output

    Args:
        sliced_json (Dict): Anchor date going back a certain number of years
    Returns:
        tuple[str, int]: Patient id and chronologic length of the patient
    """
    # First check for pre-existence
    narrative_save_path = Path(os.environ['DETERMINISTIC_NARRATIVES_DIR']) / f"{sliced_json['patient_id']}.md"
    if narrative_save_path.exists() and int(os.environ['SCRUB_NARRATIVES']) == 0:
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
    in_patient_days_1_yr, num_emergency_1_yr = psych_utilization(sliced_json, 1)
    in_patient_days_3_yr, num_emergency_3_yr = psych_utilization(sliced_json, 3)
    
    # Substance abuse
    sud_names_dict = sud_specifics(sliced_json)
    
    # Form the narrative
    
    # Header
    condition = "MDD" # Default
    # Search for specifics
    found_mdd = False
    for encounter in sliced_json['encounters']:
        for diagnosis in encounter['diagnoses']:
            for code_dict in diagnosis['codes']:
                code = code_dict['code']
                description = get_mdd_description(code)
                if description != None:
                    condition += f" ({description})"
                    found_mdd = True
                    break
            if found_mdd:
                break 
        if found_mdd:
            break
                
    HEADER = f"### COHORT & INDEX\nCondition: {condition} | Index date: {sliced_json['anchor_date']} | Baseline window: {-365*int(os.environ['YEARS_BACK'])}...0 days\n"
    
    # Demographics
    demographics = [f'{demographic}: {sliced_json["demographics"].get(demographic, "Missing")}' for demographic in demographics_of_interests]
    sdoh_categories = get_sdoh(patient_dict=sliced_json)
    DEMOGRAPHICS = f"### SOCIODEMOGRAPHICS / ACCESS\n{' | '.join(demographics)}\nSDOH: {' | '.join(sdoh_categories)}\n"
    
    # Vitals
    vitals_avg_dict = get_vitals_average(sliced_json)
    
    # Handle BMI formatting
    bmi_val = vitals_avg_dict['bmi']
    bmi_str = f"{bmi_val:.1f}" if isinstance(bmi_val, (int, float)) else "Missing"
    
    # Handle BP formatting
    sys_val = vitals_avg_dict['bp_sys']
    dia_val = vitals_avg_dict['bp_dias']
    
    if isinstance(sys_val, (int, float)) and isinstance(dia_val, (int, float)):
        bp_str = f"{sys_val:.0f}/{dia_val:.0f}"
    else:
        bp_str = "Missing"

    VITALS = f"### PHYSICAL HEALTH\nBMI: {bmi_str} | BP (mean): {bp_str}\n"
    
    # Psych history
    substances = ' | '.join([sud_name for sud_name in sorted(list(sud_names_dict.keys())) if sud_names_dict[sud_name]])
    psych_comorbidities = [f'{psych_arm}: {get_bool_str(psych_comorbidity_dict[psych_arm])}' for psych_arm in psych_comorbidity_dict.keys()]
    PSYCH_HISTORY = f"### PSYCH HISTORY\n{' | '.join(psych_comorbidities)}\nSUICIDE FLAG (12m): {get_bool_str(suicide_flag)}\nSUBSTANCE ABUSE: {substances if len(substances) > 0 else 'None'}\n"
    
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
     
    # Utilization
    UTILIZATION = f"### UTILIZATION\nPsych inpatient days: {in_patient_days_1_yr} (12m) / {in_patient_days_3_yr} (3y) | ED psych visits: {num_emergency_1_yr} (12m) / {num_emergency_3_yr} (3y)\n"
        
    # Safety
    SAFETY = f"### SAFETY\n{' | '.join([f'{safety_arm}: {get_bool_str(safety_comorbidity_dict[safety_arm])}' for safety_arm in safety_comorbidity_dict.keys()])}\n"
    
    result = "\n".join([HEADER, DEMOGRAPHICS, VITALS, PSYCH_HISTORY, MED_HISTORY, TREAT_EXPOSURE, MED_BURDEN, UTILIZATION, SAFETY])
    # Record the narrative
    os.makedirs(narrative_save_path.parent, exist_ok=True)
    with open(narrative_save_path, 'w') as f:
        f.write(result)
    return (sliced_json['patient_id'], sliced_json['days_of_history'])


if __name__=="__main__":
    # Take a sample of produced narratives and put them in the local test_data directory
    all_narratives = list(Path(os.environ['DETERMINISTIC_NARRATIVES_DIR']).glob("*.md"))
    for narrative in random.sample(all_narratives, 10):
        with open(narrative, 'r') as f:
            content = f.read()
            new_file = Path("test_data") / narrative.name
            with open(new_file, 'w') as nf:
                nf.write(content)