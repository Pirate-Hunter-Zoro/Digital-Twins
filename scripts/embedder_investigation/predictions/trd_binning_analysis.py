import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
)
from scipy.stats import spearmanr
from pathlib import Path
import os

from dotenv import load_dotenv
load_dotenv()

from scripts.shared.utils import load_neighborhood_data
from scripts.embedder_investigation.neighbors.neighbor_scheme import RELEVANT_NEIGHBOR_SCHEMES
from scripts.embedder_investigation.predictions.weighting_strategy import RELEVANT_WEIGHTING_STRATS

def load_and_merge_data() -> pd.DataFrame:
    """Loads the neighbor data for each anchor patient, as well as the evaluation results for each anchor patient and merges them into a single array

    Returns:
        pd.DataFrame: Resulting array that for each anchor patient describes neighborhood similarity and the TRD flag and risk score information
    """
    predictions_results_df = pd.read_csv(Path(os.environ['RESULTS_DIR']) / f"summary_predictions.csv")
    neighbor_df = load_neighborhood_data()
    # Group by anchor id and prediction scheme but then turn back into regular column
    neighbor_df = neighbor_df.groupby(['anchor_patient_id', 'neighbor_scheme']).agg({\
                                        'cosine_sim': list,
                                        'chronological_length': 'first',\
                                        })\
                                    .reset_index()
    merged = pd.merge(predictions_results_df, neighbor_df, on=['anchor_patient_id', 'neighbor_scheme'])
    return merged

def compute_density_metrics(merged_df: pd.DataFrame) -> pd.DataFrame:
    """Computes various density metrics on each anchor patient with their neighborhoods

    Args:
        merged_df (pd.DataFrame): Neighborhood data and TRD flag and risk predictions for each anchor patient

    Returns:
        pd.DataFrame: Resulting information for each anchor patient on their neighborhood density
    """
    threshold = float(os.environ['HIGH_SIM_THRESHOLD'])
    def calculate_metrics(scores):
        scores = np.array(scores)
        knn_rad = 1 - np.mean(scores)
        high_sim_count = np.sum(scores > threshold)
        return pd.Series({
            "knn_radius": knn_rad, 
            "high_similarity_count": high_sim_count,
        })
    # Get data frame with knn_rad and high_sim_count for each anchor patient
    metrics_df = merged_df['cosine_sim'].apply(calculate_metrics)
    return pd.concat([merged_df, metrics_df], axis=1)
    

def stratify_by_density(
    df: pd.DataFrame, 
    metric_col: str = 'knn_radius', 
    n_bins: int = 5
) -> pd.DataFrame:
    """
    Bins the dataframe into quintiles based on the specified density metric (e.g., knn_radius).
    
    Adds a 'density_bin' column (e.g., 'Q1 (Dense)', 'Q5 (Sparse)').
    
    Returns the modified DataFrame.
    """
    df['density_bin'] = pd.qcut(df[metric_col], q=n_bins, duplicates='drop')
    return df

def compute_density_bin_scores(
    stratified_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Groups by 'density_bin' and calculates:
    - AUC (Area Under Curve)
    - Brier Score
    - ECE (Expected Calibration Error)
    - Count (N patients in bin)
    
    Returns a summary DataFrame indexed by bin.
    """
    grouped_df = stratified_df.groupby(['density_bin', 'weighting_strategy', 'neighbor_scheme'], observed=False)
    def extractor(df_bin):
        true_labels = df_bin['true_label']
        predicted_risks = df_bin['predicted_risk']
        # Find metrics on this - if it breaks, let it - I want to see the error
        roc_score = roc_auc_score(y_true=true_labels, y_score=predicted_risks)
        brier_score = brier_score_loss(y_true=true_labels, y_proba=predicted_risks)
        patient_count_in_bin = len(df_bin)
        return pd.Series([roc_score, brier_score, patient_count_in_bin], index=['roc_score', 'brier_score', 'patient_count_in_bin'])
    return grouped_df.apply(extractor, include_groups=False).reset_index()

def stratify_by_chronology(
    df: pd.DataFrame, 
    metric_col: str = 'chronological_length', 
    n_bins: int = 5
) -> pd.DataFrame:
    """
    Bins the dataframe into quintiles based on the pre-anchor history length in days.
    
    Adds a 'chronological_bin' column.
    
    Returns the modified DataFrame.
    """
    df['chronological_bin'] = pd.qcut(df[metric_col], q=n_bins, duplicates='drop')
    return df

def compute_chronological_bin_scores(
    stratified_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Groups by 'chronological_bin' and calculates:
    - AUC (Area Under Curve)
    - Brier Score
    - ECE (Expected Calibration Error)
    - Count (N patients in bin)
    
    Returns a summary DataFrame indexed by bin.
    """
    grouped_df = stratified_df.groupby(['chronological_bin', 'weighting_strategy', 'neighbor_scheme'], observed=False)
    def extractor(df_bin):
        true_labels = df_bin['true_label']
        predicted_risks = df_bin['predicted_risk']
        # Find metrics on this - if it breaks, let it - I want to see the error
        roc_score = roc_auc_score(y_true=true_labels, y_score=predicted_risks)
        brier_score = brier_score_loss(y_true=true_labels, y_proba=predicted_risks)
        patient_count_in_bin = len(df_bin)
        return pd.Series([roc_score, brier_score, patient_count_in_bin], index=['roc_score', 'brier_score', 'patient_count_in_bin'])
    return grouped_df.apply(extractor, include_groups=False).reset_index()

def plot_bin_impact(
    performance_summary: pd.DataFrame, 
    bin_type: str,
) -> None:
    """
    Generates a dual-axis line plot:
    - X-Axis: Specified Bin (e.g. Density, Chronological)
    - Y-Axis Left: AUC (Higher is better)
    - Y-Axis Right: ECE/Brier (Lower is better)
    - Computes Spearman Rho association between bin index and both metrics
    
    Saves the plot to output_path.
    """
    
    for scheme in RELEVANT_NEIGHBOR_SCHEMES:
        for strat in RELEVANT_WEIGHTING_STRATS:
            # Set up graph and filter performance summary according to strategy
            fig, ax1 = plt.subplots(figsize=(10,6))
            ax2 = ax1.twinx() # Share x axis
            filtered_df = performance_summary[(performance_summary['weighting_strategy'] == strat.name) & (performance_summary['neighbor_scheme'] == scheme.name)]
            
            # Extract roc scores, brier scores, and bin numbers
            roc_scores, brier_scores, bins, bin_counts = filtered_df['roc_score'],\
                                                filtered_df['brier_score'],\
                                                    np.arange(len(filtered_df[f'{bin_type}_bin'])),\
                                                        filtered_df['patient_count_in_bin']
            rho_roc, p_value_roc = spearmanr(roc_scores, bins)
            rho_brier, p_value_brier = spearmanr(brier_scores, bins)
            bin_labels = [f'{bin_interval}\nN={count}' for bin_interval, count in zip(filtered_df[f'{bin_type}_bin'].astype(str), bin_counts)]
            metrics = f"\
    Rho_ROC: {rho_roc:.2f}\n\
    P-Value_ROC: {p_value_roc:.2f}\n\
    Rho_BRIER: {rho_brier:.2f}\n\
    P-Value_BRIER: {p_value_brier:.2f}\
    "
            
            ax1.plot(bin_labels, filtered_df['roc_score'], color='green', linestyle='dashed', label='ROC Score')
            ax2.plot(bin_labels, filtered_df['brier_score'], color='red', linestyle='solid', label='Brier Score')
            ax1.set_xlabel(f'{bin_type.capitalize()} Bin')
            ax1.set_ylabel('ROC Score')
            ax1.legend(loc='upper left')
            ax2.set_ylabel('Brier Score')
            ax2.legend(loc='upper right')
            fig.suptitle(metrics)
            save_path = Path(os.environ['RESULTS_DIR']) / f"{bin_type}_binning" / f"scores_by_{bin_type}_{scheme.name}_{strat.name}.png"
            os.makedirs(save_path.parent, exist_ok=True)
            fig.tight_layout()
            fig.savefig(str(save_path))
            plt.close(fig)

def run_trd_bin_analysis():
    df = load_and_merge_data()
    
    density_metrics_df=compute_density_metrics(df) 
    density_stratified = stratify_by_density(density_metrics_df)
    density_scores_by_bin = compute_density_bin_scores(density_stratified)
    # Save the .csv so we can see the bin counts
    density_scores_by_bin.to_csv(Path(os.environ['RESULTS_DIR']) / f"density_performance_summary.csv")
    plot_bin_impact(density_scores_by_bin, bin_type='density')
    
    chronological_stratified = stratify_by_chronology(df)
    chronological_scores_by_bin = compute_chronological_bin_scores(chronological_stratified)
    # Again save the .csv so we can see the bin counts
    chronological_scores_by_bin.to_csv(Path(os.environ['RESULTS_DIR']) / f"chronology_performance_summary.csv")
    plot_bin_impact(chronological_scores_by_bin, bin_type='chronological')