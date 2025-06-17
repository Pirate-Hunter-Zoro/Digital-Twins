import pandas as pd
import os
import sqlite3
import json
import numpy as np
import multiprocessing
import math

# --- Define Paths ---
base_data_path = "/mnt/dell_storage/studies/ehr_study/data-EHR-prepped/25_04_17-v1/PrepData-2025_04_17/" # Corrected base_data_path
output_json_dir = "/home/librad.laureateinstitute.org/mferguson/Digital-Twins/real_data/" # Your requested output directory
os.makedirs(output_json_dir, exist_ok=True)

db_file = os.path.join(output_json_dir, "ehr_data.db")

# Define the log file for processed patient IDs
PROCESSED_IDS_LOG = os.path.join(output_json_dir, "processed_patient_ids.log")


# --- Helper function to load previously processed IDs ---
def load_processed_ids(log_filepath):
    processed_ids = set()
    if os.path.exists(log_filepath):
        with open(log_filepath, 'r') as f:
            for line in f:
                processed_ids.add(line.strip())
    return processed_ids

# --- DEFINITION OF load_csv_to_sqlite FUNCTION (MUST BE PRESENT!) ---
def load_csv_to_sqlite(csv_filepath, table_name, db_connection, chunk_size=100000, **kwargs):
    print(f"Loading {table_name} from {csv_filepath} into SQLite...")
    chunk_count = 0
    for chunk in pd.read_csv(csv_filepath, chunksize=chunk_size, **kwargs):
        chunk.to_sql(table_name, db_connection, if_exists='append', index=False)
        chunk_count += 1
        print(f"  Processed {table_name} chunk {chunk_count}")
    print(f"Finished loading {table_name}.")


# --- PHASE 1: DATABASE POPULATION (RUN THIS ONCE, THEN COMMENT OUT) ---
"""
print("\n--- PHASE 1: Populating SQLite Database from CSVs ---")
conn_populate = sqlite3.connect(db_file)
print(f"Connected to SQLite database for population: {db_file}")

csv_files_to_load = {
    "Encounter_Table": "Encounter_Table-25_04_17-v1.csv",
    "Diagnosis_Table": "Diagnosis_Table-25_04_17-v1.csv",
    "Medication_Table": "Medication_Table-25_04_17-v1.csv",
    "Procedure_Table": "Procedure_Table-25_04_17-v1.csv"
}

common_read_csv_kwargs = {
    'low_memory': False,
    'on_bad_lines': 'skip'
}

for table_name, filename in csv_files_to_load.items():
    filepath = os.path.join(base_data_path, filename)
    try:
        specific_dtypes = {}
        if table_name == "Encounter_Table":
            specific_dtypes = {'PatientEpicId_SH': str, 'EncounterId_SH': str, 'PatientType': str, 'PatientClass': str, 'DepartmentKey': str, 'PlaceOfServiceKey': str, 'GroupType': str}
        elif table_name == "Diagnosis_Table":
            specific_dtypes = {'EncounterId_SH': str, 'Diagnosis_1_Code': str}
        elif table_name == "Medication_Table":
             specific_dtypes = {'EncounterId_SH': str}
        elif table_name == "Procedure_Table":
             specific_dtypes = {'EncounterId_SH': str}

        load_csv_to_sqlite(filepath, table_name, conn_populate, dtype=specific_dtypes, **common_read_csv_kwargs)
    except FileNotFoundError:
        print(f"File not found: {filepath}. Skipping {table_name}.")
    except Exception as e:
        print(f"Error loading {table_name}: {e}")

conn_populate.close()
print("PHASE 1: SQLite database population complete. Connection closed.")
"""


# --- PHASE 2: INDEXING (RUN THIS AFTER POPULATION, THEN COMMENT OUT) ---
# Load Person_Table (still into DataFrame, as it's smaller)
print("\nLoading Person_Table into memory...")
try:
    person_df = pd.read_csv(os.path.join(base_data_path, "Person_Table-25_04_17-v1.csv"), low_memory=False)
    person_df['person_id'] = person_df['person_id'].astype(str)
    print(f"Loaded Person_Table. Shape: {person_df.shape}")
except Exception as e:
    print(f"Error loading Person_Table: {e}")
    exit()

print("\n--- PHASE 2: Creating indexes on SQLite tables... This will take some time. ---")
conn_index = sqlite3.connect(db_file)
cursor_index = conn_index.cursor()

try:
    cursor_index.execute("CREATE INDEX IF NOT EXISTS idx_encounter_patient ON Encounter_Table (PatientEpicId_SH);")
    print("  Index 'idx_encounter_patient' created on Encounter_Table.")
    cursor_index.execute("CREATE INDEX IF NOT EXISTS idx_diagnosis_encounter ON Diagnosis_Table (EncounterId_SH);")
    print("  Index 'idx_diagnosis_encounter' created on Diagnosis_Table.")
    cursor_index.execute("CREATE INDEX IF NOT EXISTS idx_medication_encounter ON Medication_Table (EncounterId_SH);")
    print("  Index 'idx_medication_encounter' created on Medication_Table.")
    cursor_index.execute("CREATE INDEX IF NOT EXISTS idx_procedure_encounter ON Procedure_Table (EncounterId_SH);")
    print("  Index 'idx_procedure_encounter' created on Procedure_Table.")
    conn_index.commit()
    print("PHASE 2: All necessary indexes created and committed!")
except sqlite3.Error as e:
    print(f"An error occurred during indexing: {e}")
finally:
    conn_index.close()


# --- PHASE 3: JSON CONVERSION (OPTIMIZED WITH MULTI-PROCESSING AND RESUMPTION!) ---
# This function will be executed by each parallel process
def process_patient_chunk(person_df_chunk, db_file, output_json_dir, initial_processed_ids_set):
    """
    Function to be run by each process, processing a chunk of patients.
    Takes a DataFrame chunk, DB file path, output directory, and a set of already processed IDs.
    Each process writes its results to a temporary JSON part file and returns IDs it successfully processed.
    """
    processed_count_in_chunk = 0
    ids_processed_by_this_chunk = [] # To return IDs processed by this specific chunk

    # Each process needs its own database connection
    with sqlite3.connect(db_file) as conn_query:
        # Generate a unique filename for this chunk's output
        # Using PID for uniqueness and a random int to avoid conflicts if PIDs are reused quickly
        chunk_output_file = os.path.join(output_json_dir, f"temp_chunk_part_{os.getpid()}_{np.random.randint(10000)}.json")

        with open(chunk_output_file, 'w') as outfile:
            outfile.write('[\n') # Start the JSON array for this chunk
            is_first_patient_in_chunk = True

            for _, person_row in person_df_chunk.iterrows():
                patient_id = person_row['person_id']

                # --- RESUMPTION LOGIC (Check if already processed before doing heavy work) ---
                if patient_id in initial_processed_ids_set:
                    # print(f"Process {os.getpid()}: Skipping already processed patient {patient_id}")
                    continue # Skip this patient as it's already processed

                # --- (rest of patient data processing, same as before) ---
                demographics_dict = person_row.drop('person_id').replace({np.nan: None, pd.NA: None}).to_dict()

                patient_data = {
                    "patient_id": patient_id,
                    "demographics": demographics_dict,
                    "encounters": []
                }

                encounters_query = f"SELECT * FROM Encounter_Table WHERE PatientEpicId_SH = '{patient_id}'"
                patient_encounters_df = pd.read_sql_query(encounters_query, conn_query)

                if not patient_encounters_df.empty:
                    patient_encounters_df['EncounterId_SH'] = patient_encounters_df['EncounterId_SH'].astype(str)
                    patient_encounters_df = patient_encounters_df.replace({np.nan: None, pd.NA: None})

                    patient_encounter_ids = patient_encounters_df['EncounterId_SH'].tolist()

                    if patient_encounter_ids:
                        # Protect against SQL IN clause limits if patient_encounter_ids is huge
                        ids_str = "','".join(patient_encounter_ids)

                        diagnoses_query = f"SELECT * FROM Diagnosis_Table WHERE EncounterId_SH IN ('{ids_str}')"
                        patient_diagnoses_df = pd.read_sql_query(diagnoses_query, conn_query)
                        if not patient_diagnoses_df.empty:
                            patient_diagnoses_df['EncounterId_SH'] = patient_diagnoses_df['EncounterId_SH'].astype(str)
                            patient_diagnoses_df = patient_diagnoses_df.replace({np.nan: None, pd.NA: None})

                        medications_query = f"SELECT * FROM Medication_Table WHERE EncounterId_SH IN ('{ids_str}')"
                        patient_medications_df = pd.read_sql_query(medications_query, conn_query)
                        if not patient_medications_df.empty:
                            patient_medications_df['EncounterId_SH'] = patient_medications_df['EncounterId_SH'].astype(str)
                            patient_medications_df = patient_medications_df.replace({np.nan: None, pd.NA: None})

                        procedures_query = f"SELECT * FROM Procedure_Table WHERE EncounterId_SH IN ('{ids_str}')"
                        patient_procedures_df = pd.read_sql_query(procedures_query, conn_query)
                        if not patient_procedures_df.empty:
                            patient_procedures_df['EncounterId_SH'] = patient_procedures_df['EncounterId_SH'].astype(str)
                            patient_procedures_df = patient_procedures_df.replace({np.nan: None, pd.NA: None})

                        for enc_index, encounter_row in patient_encounters_df.iterrows():
                            encounter_id = encounter_row['EncounterId_SH']
                            encounter_details = encounter_row.drop(['PatientEpicId_SH', 'EncounterId_SH']).to_dict()

                            current_encounter_data = {
                                "encounter_id": encounter_id,
                                "details": encounter_details,
                                "diagnoses": [],
                                "medications": [],
                                "procedures": []
                            }

                            current_encounter_data["diagnoses"] = patient_diagnoses_df[
                                patient_diagnoses_df['EncounterId_SH'] == encounter_id
                            ].drop('EncounterId_SH', axis=1).to_dict(orient='records')

                            current_encounter_data["medications"] = patient_medications_df[
                                patient_medications_df['EncounterId_SH'] == encounter_id
                            ].drop('EncounterId_SH', axis=1).to_dict(orient='records')

                            current_encounter_data["procedures"] = patient_procedures_df[
                                patient_procedures_df['EncounterId_SH'] == encounter_id
                            ].drop('EncounterId_SH', axis=1).to_dict(orient='records')

                            patient_data["encounters"].append(current_encounter_data)

                # --- Write the current patient's JSON to this chunk's temporary file ---
                if not is_first_patient_in_chunk:
                    outfile.write(',\n')
                json.dump(patient_data, outfile, indent=4)
                is_first_patient_in_chunk = False

                processed_count_in_chunk += 1
                ids_processed_by_this_chunk.append(patient_id) # Add to list of processed IDs for this chunk
                if processed_count_in_chunk % 1000 == 0:
                    print(f"Process {os.getpid()}: Processed {processed_count_in_chunk} patients in this chunk.")

            # End of if not patient_encounters_df.empty
        # End of for person_row loop

        outfile.write('\n]\n') # Close the JSON array for this chunk
    print(f"Process {os.getpid()} completed its chunk, processed {processed_count_in_chunk} patients.")
    # Return both the chunk file path and the IDs processed by this chunk
    return chunk_output_file, ids_processed_by_this_chunk


print("\n--- PHASE 3: Starting patient-centric JSON conversion (optimized with Multi-Processing!)... ---")

# --- RESUMPTION LOGIC FOR MAIN PROCESS ---
master_processed_ids_set = load_processed_ids(PROCESSED_IDS_LOG)
print(f"Loaded {len(master_processed_ids_set)} previously processed patient IDs.")

# Determine number of CPU cores to use
num_processes = multiprocessing.cpu_count()
print(f"Using {num_processes} CPU cores for parallel processing.")

# Filter person_df to only include unprocessed patients
# This is crucial for efficient resumption
unprocessed_person_df = person_df[~person_df['person_id'].isin(master_processed_ids_set)].copy()
print(f"Remaining patients to process: {len(unprocessed_person_df)} out of {len(person_df)} total.")

if len(unprocessed_person_df) == 0:
    print("All patients have already been processed. Exiting.")
    # Rebuild the final combined JSON if all processed, just in case
    # Call the final combination logic (copy-pasted from below or refactored into a function)
    # to ensure the final all_patients_combined.json is up-to-date and valid.
    # This might require some careful handling if this exact script is used for that.
    # For now, just exit.
    exit()

# Split *unprocessed* person_df into chunks for each process
chunk_size_per_process = math.ceil(len(unprocessed_person_df) / num_processes)
patient_chunks = [unprocessed_person_df.iloc[i:i + chunk_size_per_process]
                  for i in range(0, len(unprocessed_person_df), chunk_size_per_process)]

# Create a pool of processes
pool = multiprocessing.Pool(processes=num_processes)

# Map the processing function to each chunk
# Pass the *initial* master_processed_ids_set to each chunk
results_tuple_list = pool.starmap(process_patient_chunk,
                                   [(chunk, db_file, output_json_dir, master_processed_ids_set) for chunk in patient_chunks])

pool.close()
pool.join()

print("\nAll parallel processes finished. Combining chunk files and updating master log...")

# --- Process results from parallel chunks ---
all_chunk_files = [res[0] for res in results_tuple_list]
all_newly_processed_ids = []
for res in results_tuple_list:
    all_newly_processed_ids.extend(res[1])

# Append newly processed IDs to the master log file
print(f"Appending {len(all_newly_processed_ids)} newly processed IDs to log...")
with open(PROCESSED_IDS_LOG, 'a') as f:
    for patient_id in all_newly_processed_ids:
        f.write(f"{patient_id}\n")
print("Master processed IDs log updated.")

# --- Rebuild the final combined JSON from all processed IDs ---
# This step is crucial for resumption:
# After new chunks are processed, the "all_patients_combined.json" needs to contain
# ALL patients, both previously processed AND newly processed.
# The most robust way is to re-stream the JSON from the individual patients' data
# which implies having their individual JSONs or querying the DB for all.
# Since we are not saving individual patient JSONs, but are only building temp chunks for the current run,
# and want one *final* combined.json, the most robust way to ensure the final file
# is complete and includes ALL patients (previous runs + current run) is:
# 1. Load ALL patient IDs from the PROCESSED_IDS_LOG.
# 2. Iterate through person_df, but this time, for *every* patient (not just unprocessed).
# 3. Query their data from SQLite.
# 4. Stream *all* patient data to the final combined JSON.

# This implies a secondary, final streaming pass that consolidates everything.
# Alternatively, if a partial 'all_patients_combined.json' could be appended to, it's very complex.
# Overwriting the existing 'all_patients_combined.json' is simpler and safer.
# This means the *entire* person_df will be iterated again, but only querying the DB.
# This part is still single-threaded, so it will be the bottleneck for combining.

# Let's simplify this final combining step:
# Since PROCESSED_IDS_LOG now contains ALL processed IDs (current + previous runs),
# we can reload person_df (or use the one already loaded), filter by these IDs,
# and then re-generate the entire final combined JSON. This will still be slow
# for the final consolidation, but ensures correctness.

print("\n--- Consolidating final combined JSON from all processed patients... ---")
final_combined_json_output_path = os.path.join(output_json_dir, "all_patients_combined.json")

# Load ALL processed IDs again (including those from this current run)
all_final_processed_ids = load_processed_ids(PROCESSED_IDS_LOG)
# Filter the original person_df for all processed patients
final_person_df_for_combine = person_df[person_df['person_id'].isin(all_final_processed_ids)].copy()

if len(final_person_df_for_combine) == 0:
    print("No patients processed yet. Combined JSON will be empty.")
    with open(final_combined_json_output_path, 'w') as final_outfile:
        final_outfile.write('[]\n') # Write an empty JSON array
else:
    # --- Initialize processed_patients_count here! ---
    processed_patients_count = 0 # <-- ADD THIS LINE

    # Use a new connection for this final consolidation pass
    with sqlite3.connect(db_file) as conn_final_combine:
        with open(final_combined_json_output_path, 'w') as final_outfile:
            final_outfile.write('[\n')
            is_first_patient_in_final_combine = True

            # Loop through ALL the patients that have *ever* been processed
            for index, person_row in final_person_df_for_combine.iterrows():
                patient_id = person_row['person_id']

                # (This is a simplified copy of the JSON generation logic from process_patient_chunk)
                demographics_dict = person_row.drop('person_id').replace({np.nan: None, pd.NA: None}).to_dict()
                patient_data = {
                    "patient_id": patient_id,
                    "demographics": demographics_dict,
                    "encounters": []
                }

                encounters_query = f"SELECT * FROM Encounter_Table WHERE PatientEpicId_SH = '{patient_id}'"
                patient_encounters_df = pd.read_sql_query(encounters_query, conn_final_combine)

                if not patient_encounters_df.empty:
                    patient_encounters_df['EncounterId_SH'] = patient_encounters_df['EncounterId_SH'].astype(str)
                    patient_encounters_df = patient_encounters_df.replace({np.nan: None, pd.NA: None})
                    patient_encounter_ids = patient_encounters_df['EncounterId_SH'].tolist()

                    if patient_encounter_ids:
                        ids_str = "','".join(patient_encounter_ids)

                        diagnoses_query = f"SELECT * FROM Diagnosis_Table WHERE EncounterId_SH IN ('{ids_str}')"
                        patient_diagnoses_df = pd.read_sql_query(diagnoses_query, conn_final_combine)
                        if not patient_diagnoses_df.empty:
                            patient_diagnoses_df['EncounterId_SH'] = patient_diagnoses_df['EncounterId_SH'].astype(str)
                            patient_diagnoses_df = patient_diagnoses_df.replace({np.nan: None, pd.NA: None})

                        medications_query = f"SELECT * FROM Medication_Table WHERE EncounterId_SH IN ('{ids_str}')"
                        patient_medications_df = pd.read_sql_query(medications_query, conn_final_combine)
                        if not patient_medications_df.empty:
                            patient_medications_df['EncounterId_SH'] = patient_medications_df['EncounterId_SH'].astype(str)
                            patient_medications_df = patient_medications_df.replace({np.nan: None, pd.NA: None})

                        procedures_query = f"SELECT * FROM Procedure_Table WHERE EncounterId_SH IN ('{ids_str}')"
                        patient_procedures_df = pd.read_sql_query(procedures_query, conn_final_combine)
                        if not patient_procedures_df.empty:
                            patient_procedures_df['EncounterId_SH'] = patient_procedures_df['EncounterId_SH'].astype(str)
                            patient_procedures_df = patient_procedures_df.replace({np.nan: None, pd.NA: None})

                        for enc_index, encounter_row in patient_encounters_df.iterrows():
                            encounter_id = encounter_row['EncounterId_SH']
                            encounter_details = encounter_row.drop(['PatientEpicId_SH', 'EncounterId_SH']).to_dict()

                            current_encounter_data = {
                                "encounter_id": encounter_id,
                                "details": encounter_details,
                                "diagnoses": [],
                                "medications": [],
                                "procedures": []
                            }

                            current_encounter_data["diagnoses"] = patient_diagnoses_df[
                                patient_diagnoses_df['EncounterId_SH'] == encounter_id
                            ].drop('EncounterId_SH', axis=1).to_dict(orient='records')

                            current_encounter_data["medications"] = patient_medications_df[
                                patient_medications_df['EncounterId_SH'] == encounter_id
                            ].drop('EncounterId_SH', axis=1).to_dict(orient='records')

                            current_encounter_data["procedures"] = patient_procedures_df[
                                patient_procedures_df['EncounterId_SH'] == encounter_id
                            ].drop('EncounterId_SH', axis=1).to_dict(orient='records')

                            patient_data["encounters"].append(current_encounter_data)

                if not is_first_patient_in_final_combine:
                    final_outfile.write(',\n')
                json.dump(patient_data, final_outfile, indent=4)
                is_first_patient_in_final_combine = False

                processed_patients_count += 1 # This line is fine now

                # Print consolidation progress
                if processed_patients_count % 1000 == 0:
                    print(f"Consolidating combined JSON: Processed {processed_patients_count} patients.")

            final_outfile.write('\n]\n')

print(f"\nPHASE 3: Total patients considered for final combined JSON: {len(final_person_df_for_combine)}. JSON consolidation complete. Final combined JSON saved to: {final_combined_json_output_path}")