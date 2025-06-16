import pandas as pd
import os
import sqlite3
import json

# Define paths
base_data_path = "/mnt/dell_storage/studies/ehr_study/data-EHR-prepped/25_04_17-v1/PrepData-2025_04_17"
output_json_dir = "./real_data/"
os.makedirs(output_json_dir, exist_ok=True)

# Define the SQLite database file path
db_file = os.path.join(output_json_dir, "ehr_data.db") # The database will be created here

# --- Function to load CSV to SQLite in chunks ---
def load_csv_to_sqlite(csv_filepath, table_name, db_connection, chunk_size=100000, **kwargs):
    print(f"Loading {table_name} from {csv_filepath} into SQLite...")
    chunk_count = 0
    # Use 'iterator=True' to get a TextFileReader object, which allows chunking
    for chunk in pd.read_csv(csv_filepath, chunksize=chunk_size, **kwargs):
        chunk.to_sql(table_name, db_connection, if_exists='append', index=False)
        chunk_count += 1
        print(f"  Processed {table_name} chunk {chunk_count}")
    print(f"Finished loading {table_name}.")

# --- Step 1.1: Connect to SQLite Database ---
# This will create the database file if it doesn't exist, or connect to it if it does
conn = sqlite3.connect(db_file)
print(f"Connected to SQLite database: {db_file}")

# --- Step 1.2: Load each large CSV into a SQLite table ---
# We'll use the 'on_bad_lines' for Diagnosis and others as we found before
csv_files_to_load = {
    "Encounter_Table": "Encounter_Table-25_04_17-v1.csv",
    "Diagnosis_Table": "Diagnosis_Table-25_04_17-v1.csv",
    "Medication_Table": "Medication_Table-25_04_17-v1.csv",
    "Procedure_Table": "Procedure_Table-25_04_17-v1.csv"
}

# Common kwargs for read_csv (apply to all if suitable)
common_read_csv_kwargs = {
    'low_memory': False,
    'on_bad_lines': 'skip' # Use 'skip' for potentially malformed lines
}

for table_name, filename in csv_files_to_load.items():
    filepath = os.path.join(base_data_path, filename)
    try:
        # Pass specific dtypes here if you want to enforce them during SQLite loading
        # For example, ensure IDs are strings to match `person_id` in Person_Table
        specific_dtypes = {}
        if table_name == "Encounter_Table":
            specific_dtypes = {
                'PatientEpicId_SH': str,
                'EncounterId_SH': str,
                'PatientType': str, # For SQLite, easier as str, can convert to category later if needed in Pandas
                'PatientClass': str,
                'DepartmentKey': str, # Assuming these might have non-int values
                'PlaceOfServiceKey': str,
                'GroupType': str
            }
        elif table_name == "Diagnosis_Table":
            specific_dtypes = {
                'EncounterId_SH': str,
                'Diagnosis_1_Code': str # Ensure diagnosis codes are strings
                # Add other diagnosis columns here as str
            }
        elif table_name == "Medication_Table":
             specific_dtypes = {
                'EncounterId_SH': str,
                # Add other medication columns here as str
            }
        elif table_name == "Procedure_Table":
             specific_dtypes = {
                'EncounterId_SH': str,
                # Add other procedure columns here as str
            }

        load_csv_to_sqlite(filepath, table_name, conn, dtype=specific_dtypes, **common_read_csv_kwargs)
    except FileNotFoundError:
        print(f"File not found: {filepath}. Skipping {table_name}.")
    except Exception as e:
        print(f"Error loading {table_name}: {e}")

# Don't forget to close the connection when done with loading
conn.close()
print("All large tables loaded into SQLite. Database connection closed.")


# --- Step 1.3: Load Person_Table (still into DataFrame, as it's smaller) ---
print("\nLoading Person_Table into memory...")
try:
    person_df = pd.read_csv(os.path.join(base_data_path, "Person_Table-25_04_17-v1.csv"), low_memory=False)
    person_df['person_id'] = person_df['person_id'].astype(str) # Ensure consistent type
    print(f"Loaded Person_Table. Shape: {person_df.shape}")
except Exception as e:
    print(f"Error loading Person_Table: {e}")
    exit()

# Now, the next part of the script will involve opening the SQLite connection again
# and querying it iteratively for each patient. That's Step 2!

# --- Step 2: Iterate through Persons and Build JSON from SQLite ---
print("\nStarting patient-centric JSON conversion from SQLite...")
processed_patients_count = 0

# Reconnect to the database for querying
conn_query = sqlite3.connect(db_file)
cursor = conn_query.cursor() # Get a cursor object to execute SQL commands

# For potential date parsing:
# Function to safely parse dates (can be used later if needed)
def parse_date_col(df, col_name):
    if col_name in df.columns:
        df[col_name] = pd.to_datetime(df[col_name], errors='coerce')
    return df

# Start the main loop to process each patient
for index, person_row in person_df.iterrows():
    patient_id = person_row['person_id'] # This is already a string due to earlier conversion

    # Initialize patient's JSON structure
    patient_data = {
        "patient_id": patient_id,
        "demographics": person_row.drop('person_id').to_dict(), # Demographics directly from person_df
        "encounters": []
    }

    # --- Query for Patient's Encounters ---
    # Convert 'PatientEpicId_SH' to string in SQL query for safety
    cursor.execute(f"SELECT * FROM Encounter_Table WHERE PatientEpicId_SH = '{patient_id}'")
    patient_encounters_raw = cursor.fetchall() # Get all rows for this patient

    # Get column names from cursor description (assuming order is preserved from .to_sql)
    # This assumes the order of columns in SQLite table matches the order used when to_sql was called.
    # A safer way would be to get the column names directly from the first fetched row's table schema.
    # For now, let's assume the names were inferred by pandas.to_sql correctly.
    encounter_cols = [description[0] for description in cursor.description]

    if patient_encounters_raw: # Check if the patient has any encounters
        for raw_encounter_data in patient_encounters_raw:
            # Convert tuple to dictionary using column names
            encounter_dict = dict(zip(encounter_cols, raw_encounter_data))

            # Pop the PatientEpicId_SH as it's redundant at this level
            encounter_id = encounter_dict.pop('EncounterId_SH', None)
            encounter_details = {k: v for k, v in encounter_dict.items() if k != 'PatientEpicId_SH'} # Remove patient ID from details

            current_encounter_data = {
                "encounter_id": encounter_id,
                "details": encounter_details,
                "diagnoses": [],
                "medications": [],
                "procedures": []
            }

            # --- Query for Encounter's Diagnoses ---
            if encounter_id: # Only query if encounter_id is valid
                cursor.execute(f"SELECT * FROM Diagnosis_Table WHERE EncounterId_SH = '{encounter_id}'")
                diagnoses_raw = cursor.fetchall()
                diagnosis_cols = [description[0] for description in cursor.description] # Get cols for diagnosis

                for raw_diagnosis_data in diagnoses_raw:
                    diag_dict = dict(zip(diagnosis_cols, raw_diagnosis_data))
                    diag_dict.pop('EncounterId_SH', None) # Remove redundant encounter ID
                    current_encounter_data["diagnoses"].append(diag_dict)

            # --- Query for Encounter's Medications ---
            if encounter_id:
                cursor.execute(f"SELECT * FROM Medication_Table WHERE EncounterId_SH = '{encounter_id}'")
                medications_raw = cursor.fetchall()
                medication_cols = [description[0] for description in cursor.description]

                for raw_medication_data in medications_raw:
                    med_dict = dict(zip(medication_cols, raw_medication_data))
                    med_dict.pop('EncounterId_SH', None)
                    current_encounter_data["medications"].append(med_dict)

            # --- Query for Encounter's Procedures ---
            if encounter_id:
                cursor.execute(f"SELECT * FROM Procedure_Table WHERE EncounterId_SH = '{encounter_id}'")
                procedures_raw = cursor.fetchall()
                procedure_cols = [description[0] for description in cursor.description]

                for raw_procedure_data in procedures_raw:
                    proc_dict = dict(zip(procedure_cols, raw_procedure_data))
                    proc_dict.pop('EncounterId_SH', None)
                    current_encounter_data["procedures"].append(proc_dict)

            patient_data["encounters"].append(current_encounter_data)

    # --- Write JSON to File ---
    output_filename = os.path.join(output_json_dir, f"patient_{patient_id}.json")
    with open(output_filename, 'w') as f:
        # Use default=str to handle non-JSON serializable types (like NaT if any slipped through)
        json.dump(patient_data, f, indent=4, default=str)

    processed_patients_count += 1
    if processed_patients_count % 1000 == 0:
        print(f"Processed {processed_patients_count} patients. Example output for patient {patient_id}...")
        # Optional: Print a snippet of a generated JSON file to verify its structure
        # if processed_patients_count == 1000:
        #    with open(output_filename, 'r') as f_read:
        #        print(f_read.read()[:1000] + "\n...")

# Close the database connection after all patients are processed
conn_query.close()
print(f"\nFinished processing all {processed_patients_count} patients. JSON files saved to: {output_json_dir}")