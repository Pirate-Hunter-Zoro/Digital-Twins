import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve

from dotenv import load_dotenv
load_dotenv()

def compute_density_metrics(neighbor_df: pd.DataFrame) -> pd.DataFrame:
    """_summary_

    Args:
        neighbor_df (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    pass