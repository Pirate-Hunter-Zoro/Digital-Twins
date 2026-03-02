from typing import Dict, Set, Tuple

from scripts.data_loading.diagnoses_definitions import PSYCH_ARMS, MEDICAL_ARMS, get_diagnosis_arm, SUICIDE_ARMS, SAFETY_ARMS, get_sud_substance, SDOH_MAP
from scripts.data_loading.med_definitions import NSAID_INGREDIENTS, BENZODIAZEPINE_INGREDIENTS, get_med_arm, ALL_ARMS, AUGMENTATION_INGREDIENTS, ALL_ARM_INGREDIENTS, HYPNOTICS_INGREDIENTS, MASTER_INGREDIENTS_MAP
from scripts.data_loading.procedure_definitions import ECT_KEYWORDS, TMS_KEYWORDS

def comorbidity(patient_dict: Dict, arm_set: Set[str]) -> Dict[str, bool]:
    """
    Determine all of the specified comorbidity flags of this patient
    """
    comorbidity = {code : False for code in arm_set}
    for encounter in patient_dict['encounters']:
        for diagnosis in encounter['diagnoses']:
            for diagnosis_code in diagnosis['codes']:
                arm = get_diagnosis_arm(diagnosis_code['code'])
                if arm != None and arm in comorbidity.keys():
                    comorbidity[arm] = True
    
    return comorbidity

def psych_comorbidity(patient_dict: Dict) -> Dict[str, bool]:
    """
    Return if a diagnosis of each psychological arm is present in the patient data
    """
    return comorbidity(patient_dict, PSYCH_ARMS)

def medical_comorbidity(patient_dict: Dict) -> Dict[str, bool]:
    """
    Return if a diagnosis of each medical arm is present in the patient data
    """
    return comorbidity(patient_dict, MEDICAL_ARMS)

def safety_comorbidity(patient_dict: Dict) -> Dict[str, bool]:
    """
    Return if a diagnosis of each safety arm is present in the patient data
    """
    return comorbidity(patient_dict, SAFETY_ARMS)

def suicidality_flag(patient_dict: Dict) -> bool:
    """
    Analyze the history window searching for specific suicide codes
    """
    for encounter in patient_dict['encounters']:
        for diagnosis in encounter['diagnoses']:
            codes = diagnosis['codes']
            for code_info in codes:
                if suicidal(code_info['code']):
                    return True
    return False

def suicidal(diagnosis_code: str) -> bool:
    """
    Determine if diagnosis code is suicidal
    """
    return get_diagnosis_arm(diagnosis_code) in SUICIDE_ARMS

def burden(patient_dict: dict, ingredients: set[str]) -> set[str]:
    """
    Helper method to count the number of included ingredients that have duration at least 7 days
    """
    distinct_ingredients = set()
    for med in patient_dict['active_medications']:
        name = med['MedName']
        for ingredient in ingredients:
            if ingredient in name.lower():
                med_end = med['MedEndInstant']
                med_start = med['MedStartInstant']
                
                if med_end - med_start >= 7:
                    # at least a week duration
                    distinct_ingredients.add(ingredient)
                break
    return distinct_ingredients

def nsaid_burden(patient_dict: dict) -> set[str]:
    """
    Count distinct NSAID ingredients with duration at least 7 days
    """
    return burden(patient_dict, NSAID_INGREDIENTS)

def hypnotic_burden(patient_dict: dict) -> set[str]:
    """
    Count distinct hypnotic ingredients with duration at least 7 days
    """
    return burden(patient_dict, HYPNOTICS_INGREDIENTS)

def polypharmacy(patient_dict: dict) -> set[str]:
    """
    Count distinct active ingredients at t=0 (active means start at or before 0 days and end on or after 0 days or ongoing)
    """
    distinct_active_meds = set()
    for med in patient_dict['active_medications']:
        name = med['MedName']
        for ingredient in MASTER_INGREDIENTS_MAP.keys():
            if ingredient in name.lower():
                start = med['MedStartInstant']
                end = med['MedEndInstant']
                # Determine if active
                if start <= 0 and (end == "ongoing" or end >= 0):
                    distinct_active_meds.add(ingredient)
    return distinct_active_meds

def benzo_days(patient_dict: dict) -> int:
    """
    Sum of total days covered by benzos in the sliced window
    """
    intervals = []
    for med in patient_dict['active_medications']:
        for ingredient in BENZODIAZEPINE_INGREDIENTS:
            if ingredient in med['MedName'].lower():
                end = min(0, med['MedEndInstant']) if med['MedEndInstant'] != 'ongoing' else 0
                start = med['MedStartInstant']
                intervals.append([start, end])
                break
    # Sort intervals by starting date
    intervals.sort(key=lambda x:x[0])
    merged = []
    for interval in intervals:
        if len(merged) == 0 or merged[-1][1] < interval[0]:
            # New interval
            merged.append(interval)
        else:
            # Merge this interval
            merged[-1][1] = max(merged[-1][1], interval[1])
    return sum([interval[1] - interval[0] for interval in merged])

def prior_adequate_trials(patient_dict: dict) -> Dict[str, int]:
    """
    Prior adequate antidepressant trials (24 months): for each class, did the patient have at least 6 weeks at a therapeutic dose?
    """
    result = {MedName: 0 for MedName in ALL_ARMS}
    for med in patient_dict['active_medications']:
        arm = get_med_arm(med['MedName'])
        if arm != None:
            end = med['MedEndInstant']
            end =  min(0, end) if end != 'ongoing' else 0
            start = med['MedStartInstant']
            if end - start >= 42:
                # The med in the current arm has been ongoing for adequate time
                result[arm] += 1
    return result

def psych_utilization(patient_dict: dict, years: int) -> Tuple[int, int]:
    """
    Count total number of 'in_patient' days and total number of emergency visits
    """
    start = -years*365
    in_patient_days = 0
    emergency_visits = 0
    for encounter in patient_dict['encounters']:
        if encounter['details']['start_visit'] < 0 and encounter['details']['end_visit'] > start:
            # In relevant time window
            if encounter['details']['patient_class'] == 'Inpatient':
                # Increment the in_patient_days
                end = min(0, encounter['details']['end_visit'])
                effective_start = max(start, encounter['details']['start_visit'])
                in_patient_days += max(end - effective_start, 0)
            elif encounter['details']['patient_class'] == 'Emergency':
                emergency_visits += 1
            
    return (in_patient_days, emergency_visits)

def augmentation_flag(patient_dict: dict) -> bool:
    """
    Check for temporal overlap between any drug in ALL_ARM_INGREDIENTS and any drug in LITHIUM | ANTIPSYCHOTICS.
    If they overlap for at least 14 days, return True
    """
    antidepressant_intervals = []
    augmenting_agent_intervals = []
    for med in patient_dict['active_medications']:
        name = med['MedName']
        for ingredient in ALL_ARM_INGREDIENTS:
            if ingredient in name.lower():
                # This is an antidepressant
                start = min(0, med['MedStartInstant'])
                end = min(0, med['MedEndInstant']) if med['MedEndInstant'] != "ongoing" else 0
                if end - start >= 14:
                    antidepressant_intervals.append([start, end])
                break
        for ingredient in AUGMENTATION_INGREDIENTS:
            if ingredient in name.lower():
                # This is an augmenting agent
                start = min(0, med['MedStartInstant'])
                end = min(0, med['MedEndInstant']) if med['MedEndInstant'] != "ongoing" else 0
                if end - start >= 14:
                    augmenting_agent_intervals.append([start, end])
                break
    # Now examine overlap between the intervals
    antidepressant_intervals.sort(key=lambda x:x[0])
    ad_pointer = 0
    augmenting_agent_intervals.sort(key=lambda x:x[0])
    aug_pointer = 0
    while ad_pointer < len(antidepressant_intervals) and aug_pointer < len(augmenting_agent_intervals):
        ad_interval = antidepressant_intervals[ad_pointer]
        aug_interval = augmenting_agent_intervals[aug_pointer]
        # check overlap between these two
        latest_start = max(ad_interval[0], aug_interval[0])
        earliest_end = min(ad_interval[1], aug_interval[1])
        if earliest_end - latest_start >= 14:
            return True
        # otherwise, adjust our pointers - discard the interval with the earlier end, because it has no hope of overlapping more with any future interval from the other list of intervals
        if ad_interval[1] >= aug_interval[1]:
            aug_pointer += 1
        else:
            ad_pointer += 1
    return False

def somatic_treatment_flag(patient_dict: dict) -> bool:
    """
    Flag for if a patient has received any somatic treatment during their history window
    """
    for encounter in patient_dict['encounters']:
        for procedure in encounter['procedures']:
            description = str(procedure['Procedure_Description']).upper()
            for keyword in ECT_KEYWORDS:
                if keyword in description:
                    return True
            for keyword in TMS_KEYWORDS:
                if keyword in description:
                    return True
    return False

def psychotherapy_count(patient_dict: dict) -> int:
    """
    Count number of instances of procedures with 'PSYCHOTHERAPY' in their description
    """
    count = 0
    for encounter in patient_dict['encounters']:
        for procedure in encounter['procedures']:
            if "PSYCHOTHERAPY" in str(procedure['Procedure_Description']).upper():
                count += 1
    return count

def sud_specifics(patient_dict: dict) -> dict[str, bool]:
    """
    Determine exact ingredients of substances abused by patient
    """
    result = {}
    for encounter in patient_dict['encounters']:
        for diagnosis in encounter['diagnoses']:
            for code_dict in diagnosis['codes']:
                code = code_dict['code']
                substance = get_sud_substance(code=code)
                if substance != "Other Substance":
                    result[substance] = True
    return result

def get_sdoh(patient_dict: dict) -> set[str]:
    """
    Retrieve available sociodemographic flags from patient
    """
    result = set()
    for encounter in patient_dict['encounters']:
        for diagnosis in encounter['diagnoses']:
            for code_dict in diagnosis['codes']:
                code = code_dict['code'].split('.')[0] # Only care about prefix
                if code in SDOH_MAP.keys():
                    result.add(SDOH_MAP[code])
    if len(result) == 0:
        result.add("None Recorded")
    return result

def get_vitals_average(patient_dict: dict) -> dict:
    """
    Determine average vital readings over various encounters
    """
    systolic_values = [] # blood pressure
    diastolic_values = [] # blood pressure
    bmi_values = []
    for encounter in patient_dict['encounters']:
        for vital in encounter['vitals']:
            sys_bp = vital['SystolicBloodPressure']
            dias_bp = vital['DiastolicBloodPressure']
            bmi = vital['BodyMassIndex']
            if isinstance(sys_bp, (int, float)) and sys_bp > 0:
                systolic_values.append(sys_bp)
            if isinstance(dias_bp, (int, float)) and dias_bp > 0:
                diastolic_values.append(dias_bp)
            if isinstance(bmi, (int, float)) and bmi > 0:
                bmi_values.append(bmi)
    return {
        'bmi': sum(bmi_values) / len(bmi_values) if len(bmi_values) > 0 else "Missing", 
        'bp_sys': sum(systolic_values) / len(systolic_values) if len(systolic_values) > 0 else "Missing",
        'bp_dias': sum(diastolic_values) / len(diastolic_values) if len(diastolic_values) > 0 else "Missing"
    }