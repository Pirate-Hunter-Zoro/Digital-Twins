import pandas as pd
import os
from pathlib import Path
from enum import Enum
import numpy as np

from dotenv import load_dotenv
load_dotenv()

from scripts.data_loading.med_definitions import get_med_arm

class VectorSource(Enum):
    EMBEDDED = 0
    FEATURE = 1

VITAL_COLUMNS = ("bmi", "bp_sys", "bp_dias",)

# Explicit missingness indicators for the two vital-sign blocks, used only when the
# feature matrix is loaded WITH vitals. BMI and blood pressure are recorded
# independently, so they get one indicator each; bp_sys and bp_dias are absent together
# and share an indicator.
VITAL_MISSING_INDICATORS = {"bmi": "bmi_missing", "bp_sys": "bp_missing"}

def load_neighborhood_data() -> pd.DataFrame:
    """Helper method to load all of the TRD risk prediction neighborhood data

    Returns:
        pd.DataFrame: Resulting neighborhood information
    """
    return pd.concat([pd.read_csv(f) for f in Path(os.environ['RESULTS_DIR']).glob(f"neighbor_results_*.csv")], ignore_index=True)

def cast_to_int8(df: pd.DataFrame) -> pd.DataFrame:
    """Helper function to turn all values in a pandas array into np.int8 types

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Changed dataframe with np.int8 types
    """
    return df.astype(np.int8)

def load_trd_set() -> set[str]:
    """Return the set of patient IDs who are TRD positive

    Returns:
        set[str]: Resulting set of patient IDs
    """
    return set([s.strip('\"') for s in Path(os.environ['TRD_LIST_PATH']).read_text().splitlines()])

def load_feature_matrix(patient_ids: set[str], include_vitals: bool = False) -> pd.DataFrame:
    """Load and return all of the patient feature vectors in a dataframe

    The three within-patient mean vital signs are dropped by default. They are recorded
    for ~78% of the cohort, and a numeric column carrying NaN cannot pass through the
    StandardScaler in make_classifier, so the published feature arm never saw them --
    while the narrative renders them for every patient. include_vitals is the
    representation-parity path: it keeps the three columns and adds a boolean indicator
    per vital block, so the fact of a missing measurement stays available to the model
    rather than being silently imputed away. Pair it with make_classifier's
    impute_numeric, which is what fills the NaNs, fitted on training rows only.

    Args:
        patient_ids (set[str]): Relevant patient IDs
        include_vitals (bool, optional): Keep the vital-sign columns and add missingness
            indicators. Defaults to False, the published behaviour.

    Returns:
        pd.DataFrame: Resulting features of all patient IDs
    """
    parquet_path = Path(os.environ['FEATURE_DATAFRAME_PATH'])
    cohort_df = pd.read_parquet(parquet_path)
    obj_cols = cohort_df.select_dtypes(include='object').columns
    cohort_df[obj_cols] = cohort_df[obj_cols].astype('category')
    X = cohort_df.loc[sorted(list(patient_ids))]
    if include_vitals:
        for source_column, indicator_column in VITAL_MISSING_INDICATORS.items():
            X[indicator_column] = X[source_column].isna()
    else:
        X = X.drop(columns=list(VITAL_COLUMNS))
    return X

def get_AD_mappings() -> dict[str, str]:
    """For every patient, return which antidepressant arm their anchor date prescription belongs to

    Returns:
        dict[str, str]: Patient ID, AD prescription arm
    """
    med_dates = pd.read_csv(Path(os.environ['MDD_MED_DATE_CSV_PATH'])).set_index('PatientEpicId_SH')
    med_dates = med_dates.sort_values(by='MedStartInstant', ascending=True)
    earliest_mask = ~med_dates.index.duplicated(keep='first') # Indexed by patient ID
    med_dates = med_dates[earliest_mask]
    return med_dates['MedName'].apply(get_med_arm).to_dict()