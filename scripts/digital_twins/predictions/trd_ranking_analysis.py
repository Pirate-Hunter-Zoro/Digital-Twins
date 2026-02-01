import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from pathlib import Path
import os

from dotenv import load_dotenv
load_dotenv()

from scripts.digital_twins.predictions.trd_predictor import TRDPredictor

RESULTS_DIR = Path(os.environ['RESULTS_DIR'])
ALPHA = float(os.environ['WEIGHTING_EXPONENT'])
PREDICTOR = TRDPredictor()

def homophily_helper(anchor_data: pd.DataFrame) -> float:
    """Compute homophily of TRD flags of neighbor patients given the anchor patient's dataframe of neighbors

    Args:
        anchor_data (pd.DataFrame): Anchor patient's dataframe

    Returns:
        float: Resulting homophily score
    """
    trd_flag = PREDICTOR.get_trd_status(candidate_patient_id=anchor_data['anchor_patient_id'].iloc[0])
    if trd_flag == 1:
        # Measure neighbors' agreement WITH being TRD
        return anchor_data['neighbor_trd_label'].mean()
    else:
        # Measure neighbors' agreeming with NOT being TRD
        return 1 - anchor_data['neighbor_trd_label'].mean()

def agreement(results_df: pd.DataFrame, k_values: list[int]) -> pd.DataFrame:
    """Calculates the "Agreement" metric to verify homophily

    Args:
        results_df (pd.DataFrame): All the information on the anchor patient and their neighbors
        k_values (list[int]): Number of nearest neighbors incrementally used to judge agreement

    Returns:
        pd.DataFrame: Summary results for each anchor patient
    """
    for k in k_values:
        analysis_df = results_df.copy()
        
        # Cosine similarity ranking
        analysis_df.sort_values(by='cosine_sim', ascending=False) # Descending cosine similarity
        analysis_df = analysis_df.groupby(by='anchor_id').head(k) # Grab the top k neighbors
        k_result_cos = analysis_df.groupby(by='anchor_id').apply(homophily_helper)
        
        # LLM similarity ranking
        analysis_df = results_df.copy()
        analysis_df = analysis_df[analysis_df['llm_sim'].notna()]
        analysis_df.sort_values(by='llm_sim', ascending=False)
        analysis_df.groupby(by='anchor_id').head(k)
        k_result_llm = analysis_df.groupby(by='anchor_id').apply(homophily_helper)

def main():
    results_df = pd.concat([pd.read_csv(f) for f in RESULTS_DIR.glob("trd_evaluation_results_*.csv")], ignore_index=True)
    results_df['anchor_trd_label'] = results_df['anchor_patient_id'].apply(PREDICTOR.get_trd_status)
    # Remove self from neighbors
    results_df = results_df[results_df['anchor_id'] != results_df['neighbor_id']]
    # This should be redundant but just in case
    results_df = results_df[results_df['anchor_patient_id'] != results_df['neighbor_patient_id']]