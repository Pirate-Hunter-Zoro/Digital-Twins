import pandas as pd
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(os.environ['PREP_DATA_DIR'])
MDD_PATH = DATA_DIR / 'MDD_IDs.csv'
BD_PATH = DATA_DIR / 'BD_IDs.csv'
SCH_PATH = DATA_DIR / 'SCH_IDs.csv'

OUTPUT_CSV = Path(os.environ['COHORT_PATH'])

def extract_ids(file_path: Path) -> set[str]:
    """
    Docstring for extract_ids
    
    :param file_path: Location of 'csv' file to read
    :type file_path: Path
    :return: All patient IDs associated with said file
    :rtype: set[str]
    """
    df = pd.read_csv(file_path, escapechar='\\', low_memory=False, header=None)
    return set(df.iloc[:, 0].astype(str))

def create_cohort():
    """
    Docstring for create_cohort
    """
    mdd_set = extract_ids(file_path=MDD_PATH)
    bd_set = extract_ids(file_path=BD_PATH)
    sch_set = extract_ids(file_path=SCH_PATH)
    
    print(f"Found {len(mdd_set)} patients diagnosed with MDD...")
    valid_ids = mdd_set - (bd_set | sch_set)
    print(f"Removed {len(mdd_set)-len(valid_ids)} patients also diagnosed with bipolar disorder or schizophrenia...")
    
    final_cohort = list(valid_ids)
    print(f"Example of an ID in final cohort: {final_cohort[0]}")
    df = pd.DataFrame({
        'PatientEpicId_SH':final_cohort
    })
    df.to_csv(OUTPUT_CSV, index=False)