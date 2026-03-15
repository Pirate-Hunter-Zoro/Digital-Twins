import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from seaborn import lineplot
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from pathlib import Path
import os
import json

from dotenv import load_dotenv
load_dotenv()

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from scripts.digital_twins.predictions.trd_predictor import TRDPredictor
from scripts.digital_twins.neighbors.neighbor_scheme import NeighborScheme
from scripts.shared.utils import load_neighborhood_data

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
    trd_flag = anchor_data['anchor_trd_label'].iloc[0]
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
    all_results = []
    analysis_df = results_df.copy()
        
    # Cosine similarity ranking
    sorted_by_cos = analysis_df.sort_values(by='cosine_sim', ascending=False) # Descending cosine similarity
    sorted_by_llm = analysis_df.sort_values(by='llm_sim', ascending=False) # Descending LLM similarity
    for k in k_values:
        k_analysis_df = sorted_by_cos.groupby(by='anchor_patient_id').head(k) # Grab the top k neighbors
        k_results_cos = k_analysis_df.groupby(by='anchor_patient_id').apply(homophily_helper)
        for anchor_id, score in k_results_cos.items():
            all_results.append({
                'k': k, 
                'Strategy': 'Cosine', 
                'anchor_patient_id': anchor_id, 
                'Agreement': score
            })
        
        # LLM similarity ranking
        k_analysis_df = sorted_by_llm[sorted_by_llm['llm_sim'].notna()].groupby(by='anchor_patient_id').head(k)
        k_results_llm = k_analysis_df.groupby(by='anchor_patient_id').apply(homophily_helper)
        for anchor_id, score in k_results_llm.items():
            all_results.append({
                'k': k,
                'Strategy': 'LLM',
                'anchor_patient_id': anchor_id,
                'Agreement': score
            })
            
    # Return accumulated results as a data frame
    return pd.DataFrame(all_results)

def compute_diagnostics(df: pd.DataFrame) -> dict:
    """Correlation statistics that we care about for llm vs cosine results

    Args:
        df (pd.DataFrame): Information on all anchor patients with their neighbor cosine and llm similarities

    Returns:
        dict: All relevant correlation calculation results
    """
    filtered_df = df[df['llm_sim'].notna()]
    rho, p_value = spearmanr(filtered_df['llm_sim'], filtered_df['cosine_sim'])
    results = {
        "spearman_rho": rho,
        "rho_p_value": p_value,
    }
    close = filtered_df[filtered_df['rank_cosine'] <= 5]
    far = filtered_df[filtered_df['rank_cosine'] >= 45]
    cos_closeness_labels = np.concatenate([np.ones(len(close)), np.zeros(len(far))])
    llm_sims = np.concatenate([close['llm_sim'].values, far['llm_sim'].values])
    results['roc_score_llm_predict_close'] = roc_auc_score(y_true=cos_closeness_labels, y_score=llm_sims)
    return results

def plot_agreement_curves(agreement_df: pd.DataFrame, prediction_scheme: str):
    """Helper method to plot the agreement curves associated with TRD flags of nearby patients compared to anchor patients

    Args:
        results_df (pd.DataFrame): Anchor patients and neighbors information
        prediction_scheme (str): Prediction scheme for how neighbors were selected
    """
    plt.figure(figsize=(10,6))
    lineplot(agreement_df, x='k', y='Agreement', hue='Strategy', marker='o')
    plt.axhline(y=0.5, linestyle='--') # Random
    plt.legend()
    plt.title('Agreement of Nearest Neighbors with Anchor TRD Label')
    fig_path = RESULTS_DIR / 'agreement_curves' / f'agreement_curve_{prediction_scheme}.png'
    os.makedirs(fig_path.parent, exist_ok=True)
    plt.savefig(str(fig_path))
    plt.close()

def run_trd_ranking_analysis():
    results_df = load_neighborhood_data()
    results_df['anchor_trd_label'] = results_df['anchor_patient_id'].apply(PREDICTOR.get_trd_status)
    # Remove self from neighbors
    results_df = results_df[results_df['anchor_patient_id'] != results_df['neighbor_id']]
    
    schemes = results_df['prediction_scheme'].unique()
    k_values = [5, 10, 25, 50]
    for scheme in schemes:
        # Find TRD agreement of anchor patient with nearby neighbors
        filtered_df = results_df[results_df['prediction_scheme'] == scheme]
        agreement_df = agreement(results_df=filtered_df, k_values=k_values)  
        agreement_path = RESULTS_DIR / 'agreements' / f'agreement_summary_{scheme}.csv'
        os.makedirs(agreement_path.parent, exist_ok=True)
        agreement_df.to_csv(agreement_path)
        plot_agreement_curves(agreement_df=agreement_df, prediction_scheme=scheme)
        
        # Compute correlation values for llm and cosine similarity
        correlation_diagnostics = compute_diagnostics(df=filtered_df)
        correlation_path = RESULTS_DIR / 'correlations' / f'correlation_results_cos_vs_llm_{scheme}.json'
        os.makedirs(correlation_path.parent, exist_ok=True)
        with open(correlation_path, 'w') as f:
            json.dump(correlation_diagnostics, f, indent=4)