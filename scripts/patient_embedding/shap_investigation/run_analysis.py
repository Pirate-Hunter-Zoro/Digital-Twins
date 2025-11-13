import shap
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import json
import multiprocessing
from typing import Dict, List, Tuple
from dotenv import load_dotenv

load_dotenv()

from scripts.patient_embedding.shap_investigation.feature_matrix import forge_feature_matrix_embedding, forge_feature_matrix_judging

def init_worker(X_train_global: np.array, X_test_global: np.array, y_train_global: np.array, y_test_global: np.array, feature_names_global: List[str]):
    global X_train
    global X_test
    global y_train
    global y_test
    global feature_names
    X_train = X_train_global
    X_test = X_test_global
    y_train = y_train_global
    y_test = y_test_global
    feature_names = feature_names_global
    
def run_model_analysis(job_config: Tuple[any]) -> Tuple[Dict[str, any], np.array]:
    model_name, model_object, explainer_class, explainer_args = job_config
    model = model_object
    results = {}
    
    # Calculate individual feature predictive strengths
    for i in range(X_train.shape[1]):
        X_train_features = X_train[:, i]
        X_train_features = X_train_features.reshape(-1,1)
        model.fit(X_train_features, y_train)
        X_test_features = X_test[:, i]
        X_test_features = X_test_features.reshape(-1,1)
        y_pred = model.predict(X_test_features)
        r2 = r2_score(y_test, y_pred)
        results[f'r2_feature_{i}'] = r2
    
    # Train the model on all features
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    results['r2_all_features'] = r2
    
    explainer = explainer_class(model, *explainer_args)
    shap_values = explainer.shap_values(X_test)
    
    # Finally so that the caller of this function can immediately see where to save the results
    results['model_name'] = model_name
    return (results, shap_values)
    
   
def run_embedding_shap_analysis():
    # Recall that each x in X is of the form [cos_narrative_value, cos_meds_value, cos_diags_value]
    # Each element in y is of the form [cos_full_text_value]
    X, y = forge_feature_matrix_embedding()
    feature_names = ['cos_narrative', 'cos_meds', 'cos_diags']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    jobs_to_run = [
        ("Linear", LinearRegression(), shap.LinearExplainer, [X_train]),
        ("RandomForest", RandomForestRegressor(n_estimators=100), shap.TreeExplainer, [])
    ]   
    
    with multiprocessing.Pool(processes=int(os.environ['NUM_SHAP_PROCESSES']), initializer=init_worker, initargs=(X_train, X_test, y_train, y_test, feature_names)) as pool:
        for results_and_values in pool.imap_unordered(run_model_analysis, jobs_to_run):
            save_dir = Path(os.environ['ANALYSIS_DIR']) / "shap_analysis" / f"embedding"
            os.makedirs(save_dir, exist_ok=True)
            
            # Obtain values
            shap_values = results_and_values[1]
            results = results_and_values[0]
            model_name = results['model_name']
            
            # "Bee swarm plot"
            shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
            plt.savefig(f"{str(save_dir)}/shap_{model_name}_summary.png")
            plt.clf()
            
            # Global feature importance
            shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type="bar", show=False)
            plt.savefig(f"{str(save_dir)}/shap_{model_name}_importance.png")
            plt.clf()
            
            # Results
            with open(save_dir / f'{model_name}_results.json', 'w') as f:
                json.dump(results, f, indent=4)
            
    print("Shap analysis for embedding with all regression models complete...", flush=True)

def run_judge_shap_analysis():
    # Each x in X is of the form [judge_narrative_value, judge_meds_value, judge_diags_value]
    # Each element in y is of the form [judge_full_text_value]
    X, y = forge_feature_matrix_judging()
    feature_names = ['judge_narrative', 'judge_meds', 'judge_diags']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    jobs_to_run = [
        ("Linear", LinearRegression(), shap.LinearExplainer, [X_train]),
        ("RandomForest", RandomForestRegressor(n_estimators=100), shap.TreeExplainer, [])
    ]   
    
    with multiprocessing.Pool(processes=int(os.environ['NUM_SHAP_PROCESSES']), initializer=init_worker, initargs=(X_train, X_test, y_train, y_test, feature_names)) as pool:
        for results_and_values in pool.imap_unordered(run_model_analysis, jobs_to_run):
            save_dir = Path(os.environ['ANALYSIS_DIR']) / "shap_analysis" / f"judging"
            os.makedirs(save_dir, exist_ok=True)
            
            # Obtain values
            shap_values = results_and_values[1]
            results = results_and_values[0]
            model_name = results['model_name']
            
            # "Bee swarm plot"
            shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
            plt.savefig(f"{str(save_dir)}/shap_{model_name}_summary.png")
            plt.clf()
            
            # Global feature importance
            shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type="bar", show=False)
            plt.savefig(f"{str(save_dir)}/shap_{model_name}_importance.png")
            plt.clf()
            
            # Results
            with open(save_dir / f'{model_name}_results.json', 'w') as f:
                json.dump(results, f, indent=4)
            
    print("Shap analysis for judging with all regression models complete...", flush=True)


def main():
    run_embedding_shap_analysis()
    run_judge_shap_analysis()
        
if __name__ == "__main__":
    main()