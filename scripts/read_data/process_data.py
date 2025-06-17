import pandas as pd
import os
import sqlite3
import json
import numpy as np
import multiprocessing # New import for parallel processing
import math # For splitting chunks

# Define paths (consistent with previous script)
base_data_path = "/mnt/dell_storage/studies/ehr_study/data-EHR-prepped/25_04_17-v1/PrepData-2025_04_17/" # Corrected base_data_path
output_json_dir = "./real_data/" # Your requested output directory
os.makedirs(output_json_dir, exist_ok=True)

db_file = os.path.join(output_json_dir, "ehr_data.db")

# --- DEFINITION OF load_csv_to_sqlite FUNCTION (MUST BE PRESENT!) ---
# Copy the corrected load_csv_to_sqlite function definition here:
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
# Load Person_Table (still into DataFrame, as it's smaller) - this can stay outside the comment block
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


# --- PHASE 3: JSON CONVERSION (OPTIMIZED WITH MULTI-PROCESSING!) ---
# This function will be executed by each parallel process
def process_patient_chunk(person_df_chunk, db_file, output_json_dir):
    """
    Function to be run by each process, processing a chunk of patients.
    Takes a DataFrame chunk, DB file path, and output directory.
    Each process writes its results to a temporary JSON part file.
    """
    processed_count_in_chunk = 0

    # Each process needs its own database connection
    with sqlite3.connect(db_file) as conn_query:
        # Generate a unique filename for this chunk's output
        # Using PID for uniqueness in case multiple runs or errors leave files
        chunk_output_file = os.path.join(output_json_dir, f"temp_chunk_part_{os.getpid()}_{np.random.randint(10000)}.json")

        with open(chunk_output_file, 'w') as outfile:
            outfile.write('[\n') # Start the JSON array for this chunk
            is_first_patient_in_chunk = True

            for index, person_row in person_df_chunk.iterrows():
                patient_id = person_row['person_id']

                # Handle NaNs in demographics
                demographics_dict = person_row.drop('person_id').replace({np.nan: None, pd.NA: None}).to_dict()

                patient_data = {
                    "patient_id": patient_id,
                    "demographics": demographics_dict,
                    "encounters": []
                }

                # --- Query for Patient's Encounters ---
                encounters_query = f"SELECT * FROM Encounter_Table WHERE PatientEpicId_SH = '{patient_id}'"
                patient_encounters_df = pd.read_sql_query(encounters_query, conn_query)

                if not patient_encounters_df.empty:
                    patient_encounters_df['EncounterId_SH'] = patient_encounters_df['EncounterId_SH'].astype(str)
                    patient_encounters_df = patient_encounters_df.replace({np.nan: None, pd.NA: None})

                    patient_encounter_ids = patient_encounters_df['EncounterId_SH'].tolist()

                    # --- Query for ALL Diagnoses, Medications, Procedures for this patient's encounters ---
                    if patient_encounter_ids:
                        # Protect against SQL IN clause limits if patient_encounter_ids is huge
                        # SQLite has a default limit of 999 for IN clause items, but can be higher depending on version/config
                        # For very large lists, consider multiple queries or a temp table if this becomes an issue.
                        # For now, assuming it fits.
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
                if processed_count_in_chunk % 1000 == 0:
                    print(f"Process {os.getpid()}: Processed {processed_count_in_chunk} patients in this chunk.")
            # End of if not patient_encounters_df.empty
        # End of for person_row loop

        outfile.write('\n]\n') # Close the JSON array for this chunk
    print(f"Process {os.getpid()} completed its chunk, processed {processed_count_in_chunk} patients.")
    return chunk_output_file # Return the path to the chunk file


print("\n--- PHASE 3: Starting patient-centric JSON conversion (optimized with Multi-Processing!)... ---")

# Determine number of CPU cores to use
# This gets the number of available CPUs on the machine
num_processes = multiprocessing.cpu_count()
print(f"Using {num_processes} CPU cores for parallel processing.")

# Split person_df into chunks for each process
# Using ceil to ensure all patients are covered even if it's not perfectly divisible
# The last chunk might be smaller
chunk_size_per_process = math.ceil(len(person_df) / num_processes)
patient_chunks = [person_df.iloc[i:i + chunk_size_per_process]
                  for i in range(0, len(person_df), chunk_size_per_process)]

# Create a pool of processes
pool = multiprocessing.Pool(processes=num_processes)

# Map the processing function to each chunk
# pool.starmap distributes the arguments to each process_patient_chunk call
results_chunk_files = pool.starmap(process_patient_chunk,
                                   [(chunk, db_file, output_json_dir) for chunk in patient_chunks])

pool.close() # No more tasks will be submitted to the pool
pool.join()  # Wait for all worker processes to complete their tasks

print("\nAll parallel processes finished. Combining chunk files...")

# --- Combine all temporary chunk files into one final combined JSON ---
final_combined_json_output_path = os.path.join(output_json_dir, "all_patients_combined.json")

with open(final_combined_json_output_path, 'w') as final_outfile:
    final_outfile.write('[\n') # Start the global JSON array
    is_first_global_content = True

    for chunk_file in results_chunk_files:
        try:
            with open(chunk_file, 'r') as infile:
                # Read the content of the chunk file
                content = infile.read().strip()
                # Remove the outer '[' and ']' that each chunk file has
                if content.startswith('[') and content.endswith(']'):
                    content = content[1:-1].strip()

                if content: # Only append if the chunk actually contained data
                    if not is_first_global_content:
                        final_outfile.write(',\n') # Add comma if not the very first content block
                    final_outfile.write(content)
                    is_first_global_content = False

            os.remove(chunk_file) # Clean up the temporary chunk file after combining
        except FileNotFoundError:
            print(f"Warning: Chunk file {chunk_file} not found. Skipping.")
        except Exception as e:
            print(f"Error combining chunk file {chunk_file}: {e}. Skipping.")

    final_outfile.write('\n]\n') # End the global JSON array

print(f"\nFinal combined JSON saved to: {final_combined_json_output_path}")
print(f"PHASE 3: Total processed patients: {len(person_df)}. JSON combining complete.")