import random
import pandas as pd
import sqlite3
from sklearn.metrics import roc_auc_score, brier_score_loss
from pathlib import Path
import os
from tqdm import tqdm

from scripts.digital_twins.predictions.trd_predictor import TRDPredictor
from scripts.shared.plots import plot_calibration, plot_precision_recall, plot_receiving_operator_characteristic, plot_decision_curve_analysis, plot_effective_sample_size_distribution

from dotenv import load_dotenv
load_dotenv()

def run():
    vector_db = Path(os.environ['VECTORS_DIR']) / 'vectors.db'
    connection = sqlite3.connect(vector_db)
    cursor = connection.cursor()
    # Randomly pick n patients but ensure that half are TRD positive and half are TRD negative
    cursor.execute("""
SELECT id, patient_id FROM vectors                          
"""
    ) 
    rows = cursor.fetchall()
    predictor = TRDPredictor()
    n = int(os.environ['TRD_TEST_COUNT'])
    trd_positive = [row for row in rows if predictor.get_trd_status(candidate_id=row[1]) == 1]
    trd_negative = [row for row in rows if predictor.get_trd_status(candidate_id=row[1]) == 0]
    patient_sample = random.sample(trd_positive, min(len(trd_positive), n//2)) + random.sample(trd_negative, min(len(trd_negative), n//2))
    results = []
    for narrative_hash_id, patient_id in tqdm(patient_sample):
        print("Predicting TRD risk for patient ID:", patient_id, flush=True)
        trd_status = predictor.get_trd_status(candidate_id=patient_id)
        prediction = predictor.predict_risk(index_id=narrative_hash_id)
        results.append(
            {
                'patient_id' : patient_id,
                'actual_trd_status' : trd_status,
                'trd_risk_score' : prediction['risk_score'],
                'ess' : prediction['ess']
            }
        )
    # Turn results into a dataframe
    results_df = pd.DataFrame(results)
    roc = roc_auc_score(y_true=results_df['actual_trd_status'], y_score=results_df['trd_risk_score'])
    brier = brier_score_loss(y_true=results_df['actual_trd_status'], y_prob=results_df['trd_risk_score'])
    mean_ess = results_df['ess'].mean()

    # Generate and save plots
    plot_receiving_operator_characteristic(y_true=results_df['actual_trd_status'].to_numpy(), y_prob=results_df['trd_risk_score'].to_numpy())
    plot_precision_recall(y_true=results_df['actual_trd_status'].to_numpy(), y_prob=results_df['trd_risk_score'].to_numpy())
    plot_calibration(y_true=results_df['actual_trd_status'].to_numpy(), y_prob=results_df['trd_risk_score'].to_numpy())
    plot_decision_curve_analysis(y_true=results_df['actual_trd_status'].to_numpy(), y_prob=results_df['trd_risk_score'].to_numpy())
    plot_effective_sample_size_distribution(ess_values=results_df['ess'].to_numpy())
    
    # Save dataframe to a .csv and the results to a .txt
    results_dir = Path(os.environ['RESULTS_DIR'])
    results_csv = results_dir / 'trd_evaluation_results.csv'
    results_txt = results_dir / 'trd_evaluation_summary.txt'
    results_df.to_csv(results_csv, index=False)
    with results_txt.open('w') as f:
        f.write(f"ROC AUC: {roc}\n")
        f.write(f"Brier Score: {brier}\n")
        f.write(f"Mean ESS: {mean_ess}\n")
    print("Summary of TRD Evaluation:\n", flush=True)
    print(f"ROC AUC: {roc}", flush=True)
    print(f"Brier Score: {brier}", flush=True)
    print(f"Mean ESS: {mean_ess}", flush=True)