from pathlib import Path
import os
from sklearn.metrics import roc_auc_score, brier_score_loss
import pandas as pd

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
    
    roc = roc_auc_score(y_true=results_df['actual_trd_status'], y_score=results_df['trd_risk_score'])
    brier = brier_score_loss(y_true=results_df['actual_trd_status'], y_proba=results_df['trd_risk_score'])
    mean_ess = results_df['ess'].mean()
    results_txt = RESULTS_DIR / f'trd_evaluation_results.txt'
    print(f"Writing TRD prediction evaluation results to {str(results_txt)}...")
    with open(results_txt, 'w') as f:
        f.write(f"TRD Prediction Evaluation Results\n")
        f.write(f"ROC AUC: {roc:.4f}\n")
        f.write(f"Brier Score: {brier:.4f}\n")
        f.write(f"Mean Effective Sample Size (ESS): {mean_ess:.2f}\n")

    # Generate and save plots
    print("Generating TRD prediction evaluation plots...", flush=True)
    plot_receiving_operator_characteristic(y_true=results_df['actual_trd_status'].to_numpy(), y_prob=results_df['trd_risk_score'].to_numpy())
    plot_precision_recall(y_true=results_df['actual_trd_status'].to_numpy(), y_prob=results_df['trd_risk_score'].to_numpy())
    plot_calibration(y_true=results_df['actual_trd_status'].to_numpy(), y_prob=results_df['trd_risk_score'].to_numpy())
    plot_decision_curve_analysis(y_true=results_df['actual_trd_status'].to_numpy(), y_prob=results_df['trd_risk_score'].to_numpy())
    plot_effective_sample_size_distribution(ess_values=results_df['ess'].to_numpy())
    plot_optimal_confusion_matrix(y_true=results_df['actual_trd_status'].to_numpy(), y_prob=results_df['trd_risk_score'].to_numpy())
    print("TRD prediction evaluation analysis complete.", flush=True)
    
if __name__=="__main__":
    run_analysis()