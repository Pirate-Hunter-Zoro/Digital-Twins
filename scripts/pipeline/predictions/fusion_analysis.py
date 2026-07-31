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
from scripts.pipeline.predictions.weighting_strategy import WeightingStrategy

def load_predictions() -> pd.DataFrame:
    """Load the predictions data frame that includes all the patients with their risk scores given weighting strategy and weighting scheme

    Returns:
        pd.DataFrame: Resulting information on all patients with their risk scores given different prediction strategies
    """
    return pd.read_csv(Path(os.environ['RESULTS_DIR']) / 'summary_predictions.csv', index_col=0)

def filter_df(df: pd.DataFrame) -> pd.DataFrame:
    """Filter the given predictions data frame to only have results where the neighbor scheme is nearest or farthest and the weighting scheme is uniform

    Args:
        df (pd.DataFrame): Dataframe containing all predictions

    Returns:
        pd.DataFrame: Filtered data frame
    """
    filtered_by_scheme = df[(df['neighbor_scheme'] == NeighborScheme.NEAREST.name) | (df['neighbor_scheme'] == NeighborScheme.FARTHEST.name)]
    filtered_by_weighting = filtered_by_scheme[filtered_by_scheme['weighting_strategy'] == WeightingStrategy.UNIFORM.name]
    return filtered_by_weighting

def pivot_df(df: pd.DataFrame) -> pd.DataFrame:
    """Given a data frame filtered with patient predictions to only specified neighbor schemes and one withing strategy, fuse the predictions into one row

    Args:
        df (pd.DataFrame): Inputed data frame

    Returns:
        pd.DataFrame: Fused data frame
    """
    # Collapse into one row per patient - for each patient, show the predicted risk scores associated with the nearest and farthest neighbor schemes
    collapsed = df.pivot(index='anchor_patient_id', columns='neighbor_scheme', values='predicted_risk')
    # Obtain the labels for each patient
    labels = df.groupby('anchor_patient_id')['true_label'].first()
    collapsed['true_label'] = labels
    return collapsed.rename(columns={'NEAREST':'p_near', 'FARTHEST':'p_far'})

def fuse_df(pivoted_df: pd.DataFrame) -> pd.DataFrame:
    """Given a dataframe already pivoted to record near and far risk scores over all patients, create the fused risk score in a new column

    Args:
        pivoted_df (pd.DataFrame): Pivoted data frame

    Returns:
        pd.DataFrame: Resulting data frame with the fused score
    """
    pivoted_df['p_fused'] = 0.5*(pivoted_df['p_near'] + (1 - pivoted_df['p_far']))
    pivoted_df['rank_score'] = pivoted_df['p_near'] - pivoted_df['p_far']
    return pivoted_df