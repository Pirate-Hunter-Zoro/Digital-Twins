import pandas as pd
import sqlite3
import os

# --- Re-use your path definitions ---
base_data_path = "/mnt/dell_storage/studies/ehr_study/data-EHR-prepped/25_04_17-v1/PrepData-2025_04_17/"
output_json_dir = "/home/librad.laureateinstitute.org/mferguson/Digital-Twins/real_data/" # Your requested output directory
db_file = os.path.join(output_json_dir, "ehr_data.db")

# --- Re-load your sampled person_df (if running this as a separate script) ---data-EHR-prepped/25_04_17-v1/PrepData-2025_04_17/"
person_df = pd.read_csv(os.path.join(base_data_path, "Person_Table-25_04_17-v1.csv"), low_memory=False)
TARGET_NUM_PATIENTS = 5000 # Make sure this matches your config
if TARGET_NUM_PATIENTS < len(person_df):
    person_df = person_df.sample(n=TARGET_NUM_PATIENTS, random_state=42).copy()
person_df['person_id'] = person_df['person_id'].astype(str)
print(f"Loaded and sampled person_df for check: {person_df.shape}")


# --- Connect to SQLite and get all unique PatientEpicId_SH from Encounter_Table ---
print("\nPerforming ID overlap check...")
try:
    with sqlite3.connect(db_file) as conn:
        # Get all unique PatientEpicId_SH from Encounter_Table
        query_unique_encounter_ids = "SELECT DISTINCT PatientEpicId_SH FROM Encounter_Table;"
        unique_encounter_ids_df = pd.read_sql_query(query_unique_encounter_ids, conn)
        
        # Convert to set for fast lookup and ensure string type
        unique_encounter_ids_in_db = set(unique_encounter_ids_df['PatientEpicId_SH'].astype(str).tolist())
        
        print(f"Total unique PatientEpicId_SH in Encounter_Table in DB: {len(unique_encounter_ids_in_db)}")

        # Get the patient_ids from your sampled person_df
        sampled_person_ids = set(person_df['person_id'].tolist())
        print(f"Total patient_ids in your sampled Person_Table: {len(sampled_person_ids)}")

        # Find the intersection (overlap)
        overlapping_ids = sampled_person_ids.intersection(unique_encounter_ids_in_db)

        print(f"\nNumber of sampled patients with matching encounters in DB: {len(overlapping_ids)}")

        if len(overlapping_ids) > 0:
            print("Example overlapping IDs (first 5):")
            for i, oid in enumerate(list(overlapping_ids)):
                if i >= 5: break
                print(f"- {oid}")
        else:
            print("NO OVERLAP FOUND! None of your 5000 sampled patients have matching encounters in the database.")
            print("This suggests the sampling drew patients who do not have associated encounters.")


except Exception as e:
    print(f"An error occurred during ID overlap check: {e}")