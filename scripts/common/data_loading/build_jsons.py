import pandas as pd
from pathlib import Path
import os
from dotenv import load_dotenv
import json
import multiprocessing
import re

load_dotenv()
PERSON_CSV_PATH = Path(os.environ['PERSON_CSV_PATH'])
ENCOUNTER_CSV_PATH = Path(os.environ['ENCOUNTER_CSV_PATH'])
DIAGNOSIS_CSV_PATH = Path(os.environ['DIAGNOSIS_CSV_PATH'])
MEDICATION_CSV_PATH = Path(os.environ['MEDICATION_CSV_PATH'])
PROCEDURE_CSV_PATH = Path(os.environ['PROCEDURE_CSV_PATH'])
PATIENT_JSON_DIR = Path(os.environ['PATIENT_JSON_DIR'])

CHECKPOINT = 1

DIAGNOSIS_CODE_PATTERN = re.compile(r"Diagnosis_(\d+)_Code")

IMPORTANT_MEDICATION_FIELDS = [
    "MedName",
    "MedSimpleGenericName",
    "MedStrength",
    "MedForm",
    "MedRoute",
    "MedFrequency",
    "MedStartInstant",
    "MedEndInstant"
]

IMPORTANT_PROCEDURE_FIELDS = [
    "Procedure_Category",
    "Procedure_Description",
    "ProcedureStartInstant",
    "ProcedureEndInstant"
]

SCRUB = True
    
def clean_encounter(encounter: dict) -> dict:
    # Omit useless bloated information and grab only the details that matter
    details = {
        "patient_type": encounter['details']['PatientType'],
        "patient_class": encounter['details']['PatientClass'],
        "start_visit": encounter['details']['StartVisit'],
        "end_visit": encounter['details']['EndVisit'],
    }
    
    # Clean the diagnoses by getting rid of all 'null' values
    cleaned_diagnoses = []
    for diagnosis in encounter['diagnoses']:
        trimmed_diagnosis = {
            'name': diagnosis['Diagnosis_Name'],
            'is_primary': diagnosis['IsPrimary'],
            'status': diagnosis['DiagnosisStatus'],
            'codes': []
        }
        for key, value in diagnosis.items():
            match = DIAGNOSIS_CODE_PATTERN.match(key)
            if match and pd.notna(value):
                # Find its index, and grab the respective Vocab and Description
                index = match.group(1) # This grabs the index because of the way we compiled the pattern above with the \d parentheses
                vocab = diagnosis[f'Diagnosis_{index}_Vocab']
                description = diagnosis[f'Diagnosis_{index}_Description']
                trimmed_diagnosis['codes'].append({
                    "code": value,
                    "vocab": vocab,
                    "description": description,
                })
        cleaned_diagnoses.append(trimmed_diagnosis)
        
    # Clean the medications by only grabbing certain fields
    cleaned_medications = []
    for medication in encounter['medications']:
        trimmed_medication = {}
        for key in IMPORTANT_MEDICATION_FIELDS:
            trimmed_medication[key] = medication[key]
        cleaned_medications.append(trimmed_medication)
        
    # Clean the procedures again by only grabbing certain fields
    cleaned_procedures = []
    for procedure in encounter['procedures']:
        trimmed_procedure = {}
        for key in IMPORTANT_PROCEDURE_FIELDS:
            trimmed_procedure[key] = procedure[key]
        cleaned_procedures.append(trimmed_procedure)

    # Put all cleaned information together
    return {
        "details": details,
        "diagnoses": cleaned_diagnoses,
        "medications": cleaned_medications,
        "procedures": cleaned_procedures
    }
    

def process_patient(patient_id: str, raw: bool=False) -> bool:
    """
    Load all of the patients demographics and encounter data and save their json
    """
    output_file_path = PATIENT_JSON_DIR / f'patient_{patient_id}.json'
    if output_file_path.exists() and not SCRUB:
        # Cleaned patient json already exists - return value to indicate we did NOT have to process this patient
        return False
        
    global people_df
    global encounters_df
    global diagnoses_df
    global medications_df
    global procedures_df
    patient_dict = {'patient_id':patient_id}
    patient_demographics = people_df.loc[patient_id].to_dict()
    patient_dict['demographics'] = patient_demographics
    
    # Grab all events related to this patient from the other dataframes
    try:
        patient_encounters = encounters_df.loc[patient_id]
        if isinstance(patient_encounters, pd.Series):
            # There was only one encounter present
            patient_encounters = patient_encounters.to_frame().T
        patient_encounters = patient_encounters.to_dict('records')
    except KeyError:
        # The patient had no encounters
        patient_encounters = []
    try:
        patient_diagnoses = diagnoses_df.loc[patient_id]
        if isinstance(patient_diagnoses, pd.Series):
            patient_diagnoses = patient_diagnoses.to_frame().T
        patient_diagnoses = patient_diagnoses.to_dict('records')
    except KeyError:
        patient_diagnoses = []
    try:
        patient_medications = medications_df.loc[patient_id]
        if isinstance(patient_medications, pd.Series):
            patient_medications = patient_medications.to_frame().T
        patient_medications = patient_medications.to_dict('records')
    except KeyError:
        patient_medications = []
    try:
        patient_procedures = procedures_df.loc[patient_id]
        if isinstance(patient_procedures, pd.Series):
            patient_procedures = patient_procedures.to_frame().T
        patient_procedures = patient_procedures.to_dict('records')
    except KeyError:
        patient_procedures = []
        
    # Now we construct the json based on all the patient's information
    patient_dict['encounters'] = []
    
    # For each encounter, extract the relevant medications, diagnoses, and procedures
    for encounter in patient_encounters:
        # Which medications, diagnoses, and procedures are part of this encounter?
        encounter_id = encounter['EncounterId_SH']
        relevant_meds = [med for med in patient_medications if med['EncounterId_SH'] == encounter_id]
        relevant_diagnoses = [diagnosis for diagnosis in patient_diagnoses if diagnosis['EncounterId_SH'] == encounter_id]
        relevant_procedures = [procedure for procedure in patient_procedures if procedure['EncounterId_SH'] == encounter_id]
        raw_encounter_dict = {
            'details': encounter,
            'medications': relevant_meds,
            'diagnoses': relevant_diagnoses,
            'procedures': relevant_procedures
        }
        patient_dict['encounters'].append((clean_encounter(raw_encounter_dict) if not raw else raw_encounter_dict))
    
    # Save the entire patient dict
    with open(output_file_path, 'w') as f:
        json.dump(patient_dict, f, indent=4)
        
    # Need a return value to signal that the patient was a newly processed patient
    return True

def load_all_patient_json():
    global people_df
    global encounters_df
    global diagnoses_df
    global medications_df
    global procedures_df
    
    existing_patient_json = list(PATIENT_JSON_DIR.glob("*.json"))
    patient_ids_with_visits = people_df.index
    print(f"Found {len(patient_ids_with_visits)-(len(existing_patient_json) if not SCRUB else 0)} new patients to create json for...", flush=True)
    
    # Build json for each patient
    PATIENT_JSON_DIR.mkdir(parents=True, exist_ok=True)
    with multiprocessing.Pool(processes=int(os.environ['NUM_WORKERS_NON_LLM_TASK'])) as thread_pool:
        # Apply the process_patient function to every single relevant patient ID
        num_completed = 0
        for newly_created in thread_pool.imap_unordered(process_patient, patient_ids_with_visits):
            # imap_unordered returns an iterator as we get results
            if newly_created:
                num_completed += 1
                if num_completed % CHECKPOINT == 0:
                    print(f"Completed {num_completed} patient jsons...")

def count_file_lines(p: Path) -> int:
    """
    Docstring for count_file_lines
    
    :param p: Input file path
    :type p: Path
    :return: Number of lines in file
    :rtype: int
    """
    try:
        with open(p, 'r') as f:
            return sum(1 for _ in f)
    except Exception:
        return 0

DRY_RUN = True
if __name__ == "__main__":
    global people_df
    global encounters_df
    global diagnoses_df
    global medications_df
    global procedures_df
    
    # TODO - handle commas within quotes - those should not be counted
    
    print("Started patient json creation...", flush=True)
    
    total_people_lines = count_file_lines(PERSON_CSV_PATH)  
    people_df = pd.read_csv(PERSON_CSV_PATH, dtype={23: str}, on_bad_lines='skip', low_memory=False) # PatientStatus is a string
    print(f"Percentage of people lines read successfully: {len(people_df)/(total_people_lines if total_people_lines > 0 else 1)*100:.2f}%", flush=True)
    # Make person id consistent with the other .csv files
    people_df.rename(columns={'person_id': 'PatientEpicId_SH'}, inplace=True)
    
    total_encounter_lines = count_file_lines(ENCOUNTER_CSV_PATH)
    encounters_df = pd.read_csv(ENCOUNTER_CSV_PATH, on_bad_lines='skip', low_memory=False)
    print(f"Percentage of encounter lines read successfully: {len(encounters_df)/(total_encounter_lines if total_encounter_lines > 0 else 1)*100:.2f}%", flush=True)
    
    total_diagnosis_lines = count_file_lines(DIAGNOSIS_CSV_PATH)
    diagnoses_df = pd.read_csv(DIAGNOSIS_CSV_PATH, on_bad_lines='skip', low_memory=False)
    print(f"Percentage of diagnosis lines read successfully: {len(diagnoses_df)/(total_diagnosis_lines if total_diagnosis_lines > 0 else 1)*100:.2f}%", flush=True)
    
    total_medication_lines = count_file_lines(MEDICATION_CSV_PATH)
    medications_df = pd.read_csv(MEDICATION_CSV_PATH, on_bad_lines='skip', low_memory=False)
    print(f"Percentage of medication lines read successfully: {len(medications_df)/(total_medication_lines if total_medication_lines > 0 else 1)*100:.2f}%", flush=True)
    
    total_procedure_lines = count_file_lines(PROCEDURE_CSV_PATH)
    procedures_df = pd.read_csv(PROCEDURE_CSV_PATH, on_bad_lines='skip', low_memory=False)
    print(f"Percentage of procedure lines read successfully: {len(procedures_df)/(total_procedure_lines if total_procedure_lines > 0 else 1)*100:.2f}%", flush=True)
    
    dfs = [people_df, encounters_df, diagnoses_df, medications_df, procedures_df]
    for df in dfs:
        df.set_index('PatientEpicId_SH', inplace=True)
        df.sort_index(inplace=True)
    
    if DRY_RUN:
        patient_id = "FF1E56AE15FB21D74ABB78D4DA026C5E"
        raw_json_path = Path(f"test_data/raw_{patient_id}.json")
        os.makedirs(raw_json_path.parent, exist_ok=True)
        cleaned_json_path = Path(f"test_data/cleaned_{patient_id}.json")
        
        output_file_path = PATIENT_JSON_DIR / f'patient_{patient_id}.json'
        os.makedirs(output_file_path.parent, exist_ok=True)
        process_patient(patient_id=patient_id, raw=True)
        with open(output_file_path, 'r') as f:
            raw_json = json.load(f)
        process_patient(patient_id=patient_id, raw=False)
        with open(output_file_path, 'r') as f:
            cleaned_json = json.load(f)
        
        with open(raw_json_path, 'w') as f:
            json.dump(raw_json, f, indent=4)
        with open(cleaned_json_path, 'w') as f:
            json.dump(cleaned_json, f, indent=4)
        
    else:
        load_all_patient_json()