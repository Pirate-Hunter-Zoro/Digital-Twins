import pandas as pd
import os

base_data_path = "/mnt/dell_storage/studies/ehr_study/data-EHR-prepped/25_04_17-v1/PrepData-2025_04_17"

person_table_path = os.path.join(base_data_path, "Person_Table-25_04_17-v1.csv")
encounter_table_path = os.path.join(base_data_path, "Encounter_Table-25_04_17-v1.csv")

try:
    person_df = pd.read_csv(person_table_path, low_memory=False)
    print(f"Successfully loaded Person_Table. Shape: {person_df.shape}")
    
    encounter_df = pd.read_csv(encounter_table_path, low_memory=False) # Apply low_memory=False here too for safety
    print(f"\nSuccessfully loaded Encounter_Table. Shape: {encounter_df.shape}")
    print("\nFirst 5 rows of Encounter_Table:")
    print(encounter_df.head())
    print("\nColumns in Encounter_Table:")
    print(encounter_df.columns.tolist())

    common_cols = list(set(person_df.columns) & set(encounter_df.columns))
    print(f"\nCommon columns before merge: {common_cols}")

    patient_data_df = person_df.copy()

    if 'person_id' in encounter_df.columns:
        print("\nMerging Person_Table and Encounter_Table on 'person_id'...")
        # Use an outer merge to see all rows from both, and identify missing IDs if any
        # Or 'left' to keep all persons and add encounters
        patient_data_df = pd.merge(person_df, encounter_df, on='person_id', how='left', suffixes=('_person', '_encounter'))
        print(f"Merged DataFrame shape after encounters: {patient_data_df.shape}")
        print("\nFirst 5 rows of Merged (Person + Encounter) Table:")
        print(patient_data_df.head())
    else:
        print("\n'person_id' column not found directly in Encounter_Table.")
        print("Please examine Encounter_Table.columns to find the correct patient identifier for merging.")
        print("Encounter_Table columns:", encounter_df.columns.tolist())

except FileNotFoundError:
    print(f"Error: Person_Table.csv not found at {person_table_path}")
    print("Please ensure the 'base_data_path' is correct and the file exists.")
except Exception as e:
    print(f"An error occurred while loading Person_Table: {e}")