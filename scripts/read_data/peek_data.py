import sqlite3
import pandas as pd

db_file = "/home/librad.laureateinstitute.org/mferguson/Digital-Twins/real_data/ehr_data.db"
patient_id_to_check = "6F5BA6D7889E3FB8CA01C3216F5EED43" # Use one of the IDs from your JSON

try:
    with sqlite3.connect(db_file) as conn:
        query = f"SELECT * FROM Encounter_Table WHERE PatientEpicId_SH = '{patient_id_to_check}'"
        df_check = pd.read_sql_query(query, conn)

        print(f"Querying for patient: {patient_id_to_check}")
        print(f"Resulting DataFrame shape: {df_check.shape}")
        if not df_check.empty:
            print("Found encounters for this patient!")
            print(df_check.head())
        else:
            print("NO encounters found for this patient in Encounter_Table.")

        # Also, check if PatientEpicId_SH column itself has issues in the table
        print("\nChecking unique PatientEpicId_SH values in Encounter_Table:")
        df_unique_ids = pd.read_sql_query("SELECT DISTINCT PatientEpicId_SH FROM Encounter_Table LIMIT 10;", conn)
        print(df_unique_ids)

except Exception as e:
    print(f"An error occurred during direct SQLite query: {e}")