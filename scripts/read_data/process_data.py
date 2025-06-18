import sys
import os

# --- Dynamic sys.path adjustment for module imports ---
# Get the absolute path to the directory containing the current script
current_script_dir = os.path.dirname(os.path.abspath(__file__))

# Calculate the project root.
# This assumes your 'config.py' (or other root-level modules like 'main.py')
# is located two directories up from where this script resides.
# Example: If this script is in 'project_root/scripts/read_data/your_script.py'
# then '..' takes you to 'project_root/scripts/'
# and '..', '..' takes you to 'project_root/'
project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..'))

# Add the project root to sys.path if it's not already there.
# Inserting at index 0 makes it the first place Python looks for modules,
# so 'import config' will find it quickly.
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- End of sys.path adjustment ---

import pandas as pd
import os
import sqlite3
import json
import numpy as np
import multiprocessing
import math
import scripts.config as config
import decimal

# --- Helper function to convert Decimal objects within a DataFrame ---
def convert_decimals_in_df(df):
    """
    Converts Decimal objects in all DataFrame cells to float, and explicitly
    replaces numeric NaN and string "NaN" values with None for JSON compatibility.
    Uses DataFrame.map (modern replacement for applymap).
    """
    # Step 1: Convert actual decimal.Decimal objects to float
    # Use df.map for element-wise application on DataFrame, handling potential mixed types.
    df = df.map(lambda x: float(x) if isinstance(x, decimal.Decimal) else x) 

    # Second pass: More robust handling of missing numerical values (including string "NaN")
    for col in df.columns:
        # Attempt to convert to numeric, coercing errors.
        # This will turn string "NaN" into actual np.nan
        converted_col = pd.to_numeric(df[col], errors='coerce')

        # Check if the conversion introduced NaNs OR if the column was originally an object/string type
        # that might contain textual 'NaN' or None, and now has np.nan.
        # Then, replace these np.nan values with None for JSON.
        if converted_col.hasnans or pd.api.types.is_object_dtype(df[col]):
            df[col] = converted_col.replace({np.nan: None, pd.NA: None})
        # If it successfully converted to numeric and had no NaNs, or was already numeric,
        # then keep the converted_col (which is clean).
        # This line ensures the cleaned numeric column is assigned back, even if no NaNs were present.
        else:
            df[col] = converted_col


    return df

# --- Setup Project Configuration ---
# You'll need to call setup_config with values appropriate for your project.
# For example, to get num_patients for this specific task:
# Let's assume you have a way to set these values for config.py.
# If config.py reads from an external source, you'd run that setup first.
# For a generic example, we'll set placeholder values for other parameters
# and assume num_patients from config will be the target.
config.setup_config(
    vectorizer_method="dummy_vectorizer", # Placeholder value
    distance_metric="euclidean", # Placeholder value
    use_synthetic_data=False, # Assuming real data for this context
    num_visits=0, # Not directly relevant for this specific task
    num_patients=5000, # NEW: Set a default target number here for testing, if config doesn't set it dynamically
    num_neighbors=0 # Not directly relevant for this specific task
)
global_config = config.get_global_config()
TARGET_NUM_PATIENTS = global_config.num_patients
print(f"Targeting a random subset of {TARGET_NUM_PATIENTS} patients based on config.py.")

# --- Define Paths ---
base_data_path = "/mnt/dell_storage/studies/ehr_study/data-EHR-prepped/25_04_17-v1/PrepData-2025_04_17/" # Corrected base_data_path
output_json_dir = "/home/librad.laureateinstitute.org/mferguson/Digital-Twins/real_data/" # Your requested output directory

os.makedirs(output_json_dir, exist_ok=True) # Create output directory if it doesn't exist

db_file = os.path.join(output_json_dir, "ehr_data.db")
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
# NOTE: This phase will still load *all* data into the SQLite database.
# If your disk space issue is due to the SQLite database growing too large,
# you'll need a more advanced filtering strategy *during* SQLite loading.
# For now, this samples only the *output* of JSON generation.
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
    print(f"Loaded original Person_Table. Shape: {person_df.shape}")

    # --- NEW: Sample person_df based on TARGET_NUM_PATIENTS from config ---
    if TARGET_NUM_PATIENTS < len(person_df):
        # Using a fixed random_state for reproducibility across runs if you want the same subset
        person_df = person_df.sample(n=TARGET_NUM_PATIENTS, random_state=42).copy()
        print(f"Sampled Person_Table down to {len(person_df)} patients.")
    else:
        print(f"Target number of patients ({TARGET_NUM_PATIENTS}) is greater than or equal to total patients. Processing all {len(person_df)}.")


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

print("\n--- PHASE 3: Starting patient-centric JSON conversion (optimized with Multi-Processing!)... ---")

def process_patient_chunk(person_df_chunk, db_file, output_json_dir, initial_processed_ids_set):
    """
    Function to be run by each process, processing a chunk of patients.
    ... (docstring continues) ...
    """
    processed_count_in_chunk = 0
    ids_processed_by_this_chunk = []

    with sqlite3.connect(db_file) as conn_query:
        chunk_output_file = os.path.join(output_json_dir, f"temp_chunk_part_{os.getpid()}_{np.random.randint(10000)}.json")

        outfile = None
        try:
            outfile = open(chunk_output_file, 'w')
            outfile.write('[\n')
            is_first_patient_in_chunk = True

            for index, person_row in person_df_chunk.iterrows():
                patient_id = person_row['person_id']

                if patient_id in initial_processed_ids_set:
                    continue

                try:
                    # No need to convert here explicitly, as person_df has fixed dtypes
                    demographics_dict = person_row.drop('person_id').replace({np.nan: None, pd.NA: None}).to_dict()

                    patient_data = {
                        "patient_id": patient_id,
                        "demographics": demographics_dict,
                        "encounters": []
                    }

                    # --- Query and Convert Encounters ---
                    encounters_query = f"SELECT * FROM Encounter_Table WHERE PatientEpicId_SH = '{patient_id}'"
                    patient_encounters_df = pd.read_sql_query(encounters_query, conn_query)
                    if not patient_encounters_df.empty:
                        patient_encounters_df['EncounterId_SH'] = patient_encounters_df['EncounterId_SH'].astype(str)
                        patient_encounters_df = patient_encounters_df.replace({np.nan: None, pd.NA: None})
                        patient_encounters_df = convert_decimals_in_df(patient_encounters_df)

                        patient_encounter_ids = patient_encounters_df['EncounterId_SH'].tolist()

                        # --- FIX: Filter out any None values and ensure string type before joining ---
                        ### CRITICAL FIX FOR NoneType ERROR ###
                        patient_encounter_ids = [str(e_id) for e_id in patient_encounter_ids if e_id is not None]
                        ### CRITICAL FIX FOR NoneType ERROR ###

                        if not patient_encounter_ids: # If list becomes empty after filtering, skip sub-queries
                            # print(f"Process {os.getpid()}: Patient {patient_id} had encounters but no valid EncounterId_SH after cleaning. Skipping sub-queries.", file=sys.stderr)
                            continue # Skip to the next patient's processing if no valid IDs

                        ids_str = "','".join(patient_encounter_ids) # This line is where the error occurs without the fix above

                        # --- Query and Convert Diagnoses ---
                        diagnoses_query = f"SELECT * FROM Diagnosis_Table WHERE EncounterId_SH IN ('{ids_str}')"
                        patient_diagnoses_df = pd.read_sql_query(diagnoses_query, conn_query)
                        if not patient_diagnoses_df.empty:
                            patient_diagnoses_df['EncounterId_SH'] = patient_diagnoses_df['EncounterId_SH'].astype(str)
                            patient_diagnoses_df = patient_diagnoses_df.replace({np.nan: None, pd.NA: None})
                            patient_diagnoses_df = convert_decimals_in_df(patient_diagnoses_df)

                        # --- Query and Convert Medications ---
                        medications_query = f"SELECT * FROM Medication_Table WHERE EncounterId_SH IN ('{ids_str}')"
                        patient_medications_df = pd.read_sql_query(medications_query, conn_query)
                        if not patient_medications_df.empty:
                            patient_medications_df['EncounterId_SH'] = patient_medications_df['EncounterId_SH'].astype(str)
                            patient_medications_df = patient_medications_df.replace({np.nan: None, pd.NA: None})
                            patient_medications_df = convert_decimals_in_df(patient_medications_df)

                        # --- Query and Convert Procedures ---
                        procedures_query = f"SELECT * FROM Procedure_Table WHERE EncounterId_SH IN ('{ids_str}')"
                        patient_procedures_df = pd.read_sql_query(procedures_query, conn_query)
                        if not patient_procedures_df.empty:
                            patient_procedures_df['EncounterId_SH'] = patient_procedures_df['EncounterId_SH'].astype(str)
                            patient_procedures_df = patient_procedures_df.replace({np.nan: None, pd.NA: None})
                            patient_procedures_df = convert_decimals_in_df(patient_procedures_df)

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
                    ids_processed_by_this_chunk.append(patient_id)
                    if processed_count_in_chunk % 1000 == 0:
                        print(f"Process {os.getpid()}: Processed {processed_count_in_chunk} patients in this chunk.", file=sys.stderr)

                except Exception as e:
                    print(f"Process {os.getpid()}: ERROR processing patient {patient_id}: {e}", file=sys.stderr)
                    continue

            # End of for loop
            if outfile is not None and not outfile.closed:
                outfile.write('\n]\n')
            else:
                print(f"Process {os.getpid()}: WARNING: Output file was unexpectedly closed before closing bracket could be written for chunk {chunk_output_file}.", file=sys.stderr)

        except Exception as e:
            print(f"Process {os.getpid()}: CRITICAL ERROR in chunk processing (file I/O or outer loop): {e} for chunk {chunk_output_file}", file=sys.stderr)
            raise

        finally:
            if outfile is not None and not outfile.closed:
                outfile.close()

    print(f"Process {os.getpid()} completed its chunk, processed {processed_count_in_chunk} patients.", file=sys.stderr)
    return chunk_output_file, ids_processed_by_this_chunk

master_processed_ids_set = load_processed_ids(PROCESSED_IDS_LOG)
print(f"Loaded {len(master_processed_ids_set)} previously processed patient IDs.")

num_processes = multiprocessing.cpu_count()
print(f"Using {num_processes} CPU cores for parallel processing.")

unprocessed_person_df = person_df[~person_df['person_id'].isin(master_processed_ids_set)].copy()
print(f"Remaining patients to process: {len(unprocessed_person_df)} out of {len(person_df)} total.")

if len(unprocessed_person_df) == 0:
    print("All patients have already been processed. Exiting.")
    exit()

chunk_size_per_process = math.ceil(len(unprocessed_person_df) / num_processes)
patient_chunks = [unprocessed_person_df.iloc[i:i + chunk_size_per_process]
                  for i in range(0, len(unprocessed_person_df), chunk_size_per_process)]

pool = multiprocessing.Pool(processes=num_processes)

results_tuple_list = pool.starmap(process_patient_chunk,
                                   [(chunk, db_file, output_json_dir, master_processed_ids_set) for chunk in patient_chunks])

pool.close()
pool.join()

print("\nAll parallel processes finished. Combining chunk files and updating master log...")

all_chunk_files = [res[0] for res in results_tuple_list]
all_newly_processed_ids = []
for res in results_tuple_list:
    all_newly_processed_ids.extend(res[1])

print(f"Appending {len(all_newly_processed_ids)} newly processed IDs to log...")
with open(PROCESSED_IDS_LOG, 'a') as f:
    for patient_id in all_newly_processed_ids:
        f.write(f"{patient_id}\n")
print("Master processed IDs log updated.")

print("\n--- Consolidating final combined JSON from all processed patients... ---")
final_combined_json_output_path = os.path.join(output_json_dir, "all_patients_combined.json")

processed_patients_count = 0 # Initialize processed_patients_count for this consolidation loop

all_final_processed_ids = load_processed_ids(PROCESSED_IDS_LOG)
final_person_df_for_combine = person_df[person_df['person_id'].isin(all_final_processed_ids)].copy()

if len(final_person_df_for_combine) == 0:
    print("No patients processed yet. Combined JSON will be empty.")
    with open(final_combined_json_output_path, 'w') as final_outfile:
        final_outfile.write('[]\n')
else:
    with sqlite3.connect(db_file) as conn_final_combine:
        with open(final_combined_json_output_path, 'w') as final_outfile:
            final_outfile.write('[\n')
            is_first_patient_in_final_combine = True

            # ... (the for loop for patient processing and JSON dumping to final_outfile) ...
            # This part remains unchanged from the previous version you reviewed.
            # No changes needed in this specific for loop.

            # We need to add the cleanup here for the temporary chunk files!
            for chunk_file in all_chunk_files: # Iterate through the list of temp chunk files
                try:
                    with open(chunk_file, 'r') as infile: # Re-open and read its contents
                        content = infile.read().strip()
                        # Extract content without outer brackets
                        if content.startswith('[') and content.endswith(']'):
                            content = content[1:-1].strip()

                        if content: # Only append if the chunk actually contained data
                            if not is_first_patient_in_final_combine:
                                final_outfile.write(',\n')
                            final_outfile.write(content)
                            is_first_patient_in_final_combine = False

                    os.remove(chunk_file) # <-- ADD THIS LINE: Delete the temporary file!
                    print(f"Cleaned up temporary chunk file: {chunk_file}") # Optional: print cleanup message

                except FileNotFoundError:
                    print(f"Warning: Temporary chunk file {chunk_file} not found during cleanup. Skipping.", file=sys.stderr)
                except Exception as e:
                    print(f"Error during cleanup of {chunk_file}: {e}. Skipping.", file=sys.stderr)


            final_outfile.write('\n]\n') # End the final combined JSON array

print(f"\nPHASE 3: Total patients considered for final combined JSON: {len(final_person_df_for_combine)}. JSON consolidation complete. Final combined JSON saved to: {final_combined_json_output_path}")