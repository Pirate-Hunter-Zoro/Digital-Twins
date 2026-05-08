import pandas as pd
import os
from pathlib import Path
from enum import Enum

from dotenv import load_dotenv
load_dotenv()

class VectorSource(Enum):
    EMBEDDED = 0
    FEATURE = 1

def load_neighborhood_data() -> pd.DataFrame:
    """Helper method to load all of the TRD risk prediction neighborhood data

    Returns:
        pd.DataFrame: Resulting neighborhood information
    """
    return pd.concat([pd.read_csv(f) for f in Path(os.environ['RESULTS_DIR']).glob(f"neighbor_results_*.csv")], ignore_index=True)