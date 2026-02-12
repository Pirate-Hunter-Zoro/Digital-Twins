import hashlib
import pandas as pd
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

def generate_string_id(text: str) -> str:
    """
    Generate unique ID pertaining to the input string
    
    :param text: String to give ID to
    :type text: str
    :return: Resulting ID
    :rtype: str
    """
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def load_neighborhood_data() -> pd.DataFrame:
    """Helper method to load all of the TRD risk prediction neighborhood data

    Returns:
        pd.DataFrame: Resulting neighborhood information
    """
    return pd.concat([pd.read_csv(f) for f in Path(os.environ['RESULTS_DIR']).glob("neighbor_results_*.csv")], ignore_index=True)