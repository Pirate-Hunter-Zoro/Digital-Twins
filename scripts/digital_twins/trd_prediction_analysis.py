from pathlib import Path
import os
from sklearn.metrics import roc_auc_score, brier_score_loss
import pandas as pd
import ast
import matplotlib.pyplot as plt

from scripts.digital_twins.predictions.trd_predictor import TRDPredictor
from scripts.shared.plots import (
    plot_calibration, 
    plot_precision_recall, 
    plot_receiving_operator_characteristic, 
    plot_decision_curve_analysis, 
    plot_effective_sample_size_distribution,
    plot_optimal_confusion_matrix
)

from dotenv import load_dotenv
load_dotenv()

RESULTS_DIR = Path(os.environ['RESULTS_DIR'])

def run_analysis():
    """
    Function for to run performance analysis on the TRD prediction
    """
    # Run analysis on all prediction results from all workers together
    results_files = RESULTS_DIR.glob("trd_evaluation_results_*.csv")
    results_df = pd.concat([pd.read_csv(f) for f in results_files], ignore_index=True)
    
    # Create master lists of random LLM scores and LLM scores of nearby cosine neighbors
    all_nearest_scores = []
    all_random_scores = []
    for nearest_llm_scores, random_llm_scores in zip(results_df['nearest_llm_scores'], results_df['random_llm_scores']):
        all_nearest_scores.extend(ast.literal_eval(nearest_llm_scores))
        all_random_scores.extend(ast.literal_eval(random_llm_scores))
    plt.figure(figsize=(10,6))
    plt.hist(all_nearest_scores, alpha=0.5, density=True, label="Nearest (Cosine)")
    plt.hist(all_random_scores, alpha=0.5, density=True, label="Random")
    plt.title("LLM Similarity Scores of Random vs. Nearest Cosine Neighbor Patients")
    plt.savefig(str(RESULTS_DIR / f"similarity_score_distribution.png"))
    
    prediction_modes = ['llm', 'cosine', 'uniform']
    for mode in prediction_modes:
        # Determin column name
        column_name = f"trd_risk_score_{mode}"
        
        # Grab the actual TRD status and evaluate
        roc = roc_auc_score(y_true=results_df['actual_trd_status'], y_score=results_df[column_name])
        brier = brier_score_loss(y_true=results_df['actual_trd_status'], y_proba=results_df[column_name])
        mean_ess = results_df[f'ess_{mode}'].mean()
        results_txt = RESULTS_DIR / f'trd_evaluation_results_{mode}.txt'
        print(f"Writing TRD prediction evaluation results to {str(results_txt)}...")
        with open(results_txt, 'w') as f:
            f.write(f"TRD Prediction Evaluation Results\n")
            f.write(f"ROC AUC: {roc:.4f}\n")
            f.write(f"Brier Score: {brier:.4f}\n")
            f.write(f"Mean Effective Sample Size (ESS): {mean_ess:.2f}\n")

        # Generate and save plots
        print("Generating TRD prediction evaluation plots...", flush=True)
        plot_receiving_operator_characteristic(y_true=results_df['actual_trd_status'].to_numpy(), y_prob=results_df[column_name].to_numpy(), mode=mode)
        plot_precision_recall(y_true=results_df['actual_trd_status'].to_numpy(), y_prob=results_df[column_name].to_numpy(), mode=mode)
        plot_calibration(y_true=results_df['actual_trd_status'].to_numpy(), y_prob=results_df[column_name].to_numpy(), mode=mode)
        plot_decision_curve_analysis(y_true=results_df['actual_trd_status'].to_numpy(), y_prob=results_df[column_name].to_numpy(), mode=mode)
        plot_effective_sample_size_distribution(ess_values=results_df[f'ess_{mode}'].to_numpy(), mode=mode)
        plot_optimal_confusion_matrix(y_true=results_df['actual_trd_status'].to_numpy(), y_prob=results_df[column_name].to_numpy(), mode=mode)
        print("TRD prediction evaluation analysis complete.", flush=True)
        
    # Now we have all the plots and results for each mode saved in the results directory
    predictor = TRDPredictor() # So that we can use it's trd flag
    top_k_thresholds = [10, 25, 50, 100]
    enrichment_percentages = [
        {k : 0 for k in top_k_thresholds},
        {k : 0 for k in top_k_thresholds},
    ]
    for _, row in results_df.iterrows():
        neighbors_by_llm_score = ast.literal_eval(row['neighbors_by_llm_score'])
        neighbors_by_cosine_score = ast.literal_eval(row['neighbors_by_cos_score'])
        # Compute running sum of TRD-positive patients by index for both lists
        for k_value in top_k_thresholds:
            enrichment_percentages[0][k_value] += sum(
                [predictor.get_trd_status(candidate_id=id) for id in neighbors_by_cosine_score[:k_value]]
            ) / k_value
            enrichment_percentages[1][k_value] += sum(
                [predictor.get_trd_status(candidate_id=id) for id in neighbors_by_llm_score[:k_value]]
            ) / k_value
    # TODO - Now that we have the enrichment percentages, calculate average enrichment for every k...
    
if __name__=="__main__":
    run_analysis()