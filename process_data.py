import pandas as pd
import os

base_data_path = "/mnt/dell_storage/studies/ehr_study/data-EHR-prepped/25_04_17-v1/PrepData-2025_04_17"

person_table_path = os.path.join(base_data_path, "Person_Table-25_04_17-v1.csv")

try:
    person_df = pd.read_csv(person_table_path, low_memory=False)
    print(f"Successfully loaded Person_Table. Shape: {person_df.shape}")
    print("\nFirst 5 rows of Person_Table:")
    print(person_df.head())
    print("\nColumns in Person_Table:")
    print(person_df.columns.tolist())
except FileNotFoundError:
    print(f"Error: Person_Table.csv not found at {person_table_path}")
    print("Please ensure the 'base_data_path' is correct and the file exists.")
except Exception as e:
    print(f"An error occurred while loading Person_Table: {e}")