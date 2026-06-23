import pandas as pd
import os
from pathlib import Path
from enum import Enum
import numpy as np

from dotenv import load_dotenv
load_dotenv()

class VectorSource(Enum):
    EMBEDDED = 0
    FEATURE = 1

VITAL_COLUMNS = ("bmi", "bp_sys", "bp_dias",)

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

def load_feature_matrix(patient_ids: set[str]) -> pd.DataFrame:
    """Load and return all of the patient feature vectors in a dataframe

    Args:
        patient_ids (set[str]): Relevant patient IDs

    Returns:
        pd.DataFrame: Resulting features of all patient IDs
    """
    parquet_path = Path(os.environ['FEATURE_DATAFRAME_PATH'])
    cohort_df = pd.read_parquet(parquet_path)
    obj_cols = cohort_df.select_dtypes(include='object').columns
    cohort_df[obj_cols] = cohort_df[obj_cols].astype('category')
    X = cohort_df.loc[sorted(list(patient_ids))]
    X = X.drop(columns=list(VITAL_COLUMNS))
    return X