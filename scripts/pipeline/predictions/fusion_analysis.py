import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

from dotenv import load_dotenv
load_dotenv()

from scripts.pipeline.predictions.trd_prediction_computation import compute_metrics
from scripts.pipeline.neighbors.neighbor_scheme import NeighborScheme

def load_predictions() -> pd.DataFrame:
    """Load the predictions data frame that includes all the patients with their risk scores given weighting strategy and weighting scheme

    Returns:
        pd.DataFrame: Resulting information on all patients with their risk scores given different prediction strategies
    """
    return pd.read_csv(Path(os.environ['RESULTS_DIR']) / 'summary_predictions.csv', index_col=0)

def filter_df(df: pd.DataFrame) -> pd.DataFrame:
    """Filter the given predictions data frame to only have results where the neighbor scheme is nearest or farthest

    Args:
        df (pd.DataFrame): Dataframe containing all predictions

    Returns:
        pd.DataFrame: Filtered data frame
    """
    return df[(df['neighbor_scheme'] == NeighborScheme.NEAREST.name) | (df['neighbor_scheme'] == NeighborScheme.FARTHEST.name)]

def pivot_df(df: pd.DataFrame) -> pd.DataFrame:
    """Given a data frame filtered with patient predictions to only specified neighbor schemes, fuse the predictions into one row

    Args:
        df (pd.DataFrame): Inputed data frame

    Returns:
        pd.DataFrame: Fused data frame
    """
    