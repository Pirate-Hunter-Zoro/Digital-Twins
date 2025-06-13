# LIBR EHR Data for Digital Twins: Summary & JSON Conversion Plan

This document summarizes the details of the LIBR Treatment Resistant Depression (TRD) study data, its structure, and the proposed strategy for converting it into patient-centric JSON format for the digital twins project, intended for LLM consumption.

## 1. Data Location & Files

The data is located at `/mnt/dell_storage/studies/ehr_study/data-EHR-prepped/25_04_17-v1/PrepData-2025_04_17`.

It consists of various tables in two formats:

* **`.csv` files (Comma Separated Values):** These are plain text files, straightforward to read. Examples include `Diagnosis_Table-25_04_17-v1.csv`, `Encounter_Table-25_04_17-v1.csv`, `Medication_Table-25_04_17-v1.csv`, `Person_Table-25_04_17-v1.csv`, and `Procedure_Table-25_04_17-v1.csv`.
* **`.rds` files (R Data Serialization format):** These are serialized R objects. Examples include `BD-Diagnosis_Table-25_04_17-v1.rds`, `Diagnosis_Table-25_04_17-v1.rds`, `Encounter_Table-25_04_17-v1.rds`, `Medication_Table-25_04_17-v1.rds`, `Person_Table-25_04_17-v1.rds`, and `Procedure_Table-25_04_17-v1.rds`. There are also `Sample_5000` versions for testing.

## 2. Data Content & Schema (Caboodle SFHS Dataset)

The dataset is from the Caboodle SHFS dataset, consisting of 6 separate queries. The files follow a naming convention: `TableName-YY_MM_DD-vX`.

* **`Person_Table`**:
    * **Content**: Patient demographics, race, ethnicity, age, postal code, sexual orientation, marital status, religion, smoking status, and flags for Depression, Bipolar, or Schizophrenia diagnoses.
    * **Filtering**: Limited by service area (`Service Area = 10`), patient status (current, valid, not test/historical/deceased), and age (18-110 years).
    * **Race/Ethnicity**: Original `Race` and `Ethnicity` columns are pivoted into multiple boolean/flag columns (e.g., `American_Indian_Or_Alaska_Native`, `Not_Hispanic_Or_Latino`).
    * **Diagnosis Flags**: `DepressionDiagnosis`, `BipolarDiagnosis`, `SchizophreniaDiagnosis` are generated based on specific ICD9 and ICD10 codes.
    * **Sampling**: Includes a 4:1 sampling of the depression group vs. the control group. The final sampled table is `##Final_4to1_Sample`.
    * **Key Identifier**: `ShortHashedPatientEpicId`.

* **`Encounter_Table`**:
    * **Content**: Details patient encounters (type, class, dates, department, place of service, and provider).
    * **Filtering**: Limited to encounters for patients in the `Patient` table, with `Complete` status, valid dates (less than or equal to today), specific patient classes (`Inpatient`, `Outpatient`, `Observation`, `Emergency`), and encounters with at least one diagnosis.
    * **Dependency**: Uses `##Final_4to1_Sample`.
    * **Key Identifier**: `EncounterId_SH`.

* **`Diagnosis_Table`**:
    * **Content**: Diagnosis details including ICD codes (ICD9/ICD10), descriptions, and diagnosis dates, linked to encounters.
    * **Filtering**: Limited to patients and encounters from the `Patient` and `Encounters` tables, and `EncounterKey > 0`.
    * **Structure**: Removes duplicate ICD codes per encounter and pivots them into up to 10 individual columns (e.g., `Diagnosis_1_Code`, `Diagnosis_2_Code`).
    * **Dependency**: Uses `##ENCOUNTER_TABLE`.

* **`Medication_Table`**:
    * **Content**: Medication events including order number, codes, names, strength, form, route, dose unit, start/end instants, frequency, and refill information.
    * **Filtering**: Limited to patients and encounters from the `Patient` and `Encounters` tables, and valid `PatientKey`. Excludes deleted medication events.
    * **Dependency**: Uses `##ENCOUNTER_TABLE`.

* **`Procedure_Table`**:
    * **Content**: Procedure events including CPT codes, descriptions, and start/end instants.
    * **Filtering**: Limited to patients and encounters from `Patient` and `Encounters` tables, CPT codes only, `General Procedure` type, and `EncounterKey > 0`.
    * **Dependency**: Uses `##ENCOUNTER_TABLE`.

* **`RXNorm_Table`**:
    * **Source**: Pulled from Clarity.
    * **Content**: Mapping between `Med