import pandas as pd
import sqlite3
from pathlib import Path
import os
from dotenv import load_dotenv
import json
import multiprocessing
import re

load_dotenv()
db_path = Path(os.environ['DB_PATH'])
person_csv_path = Path(os.environ['PERSON_CSV_PATH'])
output_json_dir = Path(os.environ['PATIENT_JSON_DIR'])

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
    "CPT_Procedure_Category",
    "CPT_Procedure_Description",
    "ProcedureStartInstant",
    "ProcedureEndInstant"
]

def init_worker(db_path: Path):
    global connection
    # Connect to the database for other patient data
    connection = sqlite3.connect(db_path)
    
def clean_encounter(encounter: dict) -> dict:
    # Omit useless bloated information and grab only the details that matter
    details = {
        "patient_type": encounter['details']['PatientType'],
        "patient_class": encounter['details']['PatientClass'],
        "start_visit": encounter['details']['StartVisit'],
        "end_visit": encounter['details']['EndVisit'],
        "group_type": encounter['details']['GroupType'],
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
            if match and value != None:
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
    

def process_patient(patient_id: str) -> any:
    """
    Load all of the patients demographics and encounter data and save their json
    """
    output_file_path = output_json_dir / f'patient_{patient_id}.json'
    if output_file_path.exists():
        # Cleaned patient json already exists - return value to indicate we did NOT have to process this patient
        return False
        
    global person_df
    patient_dict = {'patient_id':patient_id}
    patient_demographics = person_df.loc[patient_id].to_dict()
    patient_dict['demographics'] = patient_demographics
    # Grab all the encounters of this patient (each with an encounter ID that we can use to get its other information)
    patient_dict['encounters'] = []
    patient_encounters = pd.read_sql_query('SELECT * FROM Encounter_Table WHERE PatientEpicId_SH = ?', connection, params=(patient_id,))
    patient_diagnoses = pd.read_sql_query('SELECT * FROM Diagnosis_Table WHERE PatientEpicId_SH = ?', connection, params=(patient_id,))
    patient_medications = pd.read_sql_query('SELECT * FROM Medication_Table WHERE PatientEpicId_SH = ?', connection, params=(patient_id,))
    patient_procedures = pd.read_sql_query('SELECT * FROM Procedure_Table WHERE PatientEpicId_SH = ?', connection, params=(patient_id,))
    for _, encounter_row in patient_encounters.iterrows():
        encounter_id = encounter_row['EncounterId_SH']
        encounter_dict = {'details':encounter_row.to_dict()}
        # Now grab the diagnoses, medications, and procedures associated with this encounter
        encounter_diagnoses = patient_diagnoses[patient_diagnoses['EncounterId_SH'] == encounter_id]
        encounter_medications = patient_medications[patient_medications['EncounterId_SH'] == encounter_id]
        encounter_procedures = patient_procedures[patient_procedures['EncounterId_SH'] == encounter_id]
        # Those three things are data frames - store them as dictionaries
        encounter_dict['diagnoses'] = encounter_diagnoses.to_dict('records') 
        # Note - the 'records' argument is telling pandas that this dataframe shall become a list of dictionaries - each row->dictionary being one diagnosis
        encounter_dict['medications'] = encounter_medications.to_dict('records')
        # Each dict in the resulting list of dictionaries is one medication
        encounter_dict['procedures'] = encounter_procedures.to_dict('records')
        # Each dict in the resulting list of dictionaries is one procedure
        patient_dict['encounters'].append(clean_encounter(encounter_dict))
    # Save the entire patient dict
    with open(output_file_path, 'w') as f:
        json.dump(patient_dict, f, indent=4)
        
    # Need a return value to signal that the patient was a newly processed patient
    return True

dry_run = False

if __name__ == "__main__":
    
    if dry_run:
        target_json = Path("test_data/patient_FF1E56AE15FB21D74ABB78D4DA026C5E.json")
        with open(target_json, 'r') as f:
            raw_dict = json.load(f)
            for i, encounter in enumerate(raw_dict['encounters']):
                raw_dict['encounters'][i] = clean_encounter(encounter)
            output_file = Path("test_data/cleaned_patient_FF1E56AE15FB21D74ABB78D4DA026C5E.json")
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(raw_dict, f, indent=4)
        
    else:
        print("Started patient json creation...", flush=True)
        # Connect to the database for other patient data
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()
        cursor.execute('SELECT DISTINCT PatientEpicId_SH FROM Encounter_Table;')
        patient_ids_with_visits = cursor.fetchall()
        # Turn from tuples to strings
        patient_ids_with_visits = [patient_tuple[0] for patient_tuple in patient_ids_with_visits]
        
        person_df = pd.read_csv(person_csv_path, dtype={23: str})
        # Make person id consistent with the .db file
        person_df.rename(columns={'person_id': 'PatientEpicId_SH'}, inplace=True)
        person_df.set_index('PatientEpicId_SH', inplace=True)
        
        connection.close()
        existing_patient_json = list(output_json_dir.glob("*.json"))
        print(f"Found {len(patient_ids_with_visits)-len(existing_patient_json)} new patients to create json for...", flush=True)
        
        # Build json for each patient
        output_json_dir.mkdir(parents=True, exist_ok=True)
        with multiprocessing.Pool(processes=int(os.environ['NUM_WORKERS_NON_LLM_TASK']), initializer=init_worker, initargs=(db_path,)) as thread_pool:
            # Apply the process_patient function to every single relevant patient ID
            num_completed = 0
            for newly_created in thread_pool.imap_unordered(process_patient, patient_ids_with_visits):
                # imap_unordered returns an iterator as we get results
                if newly_created:
                    num_completed += 1
                    if num_completed % CHECKPOINT == 0:
                        print(f"Completed {num_completed} patient jsons...")