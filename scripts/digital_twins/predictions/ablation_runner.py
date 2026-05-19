import os
from pathlib import Path
import json
import subprocess
import csv
import joblib
import numpy as np
import matplotlib.pyplot as plt

from scripts.data_loading.ablation_registry import ABLATIONS
from scripts.digital_twins.predictions.create_train_test_split import create_train_test_split
from scripts.shared.utils import VectorSource
from scripts.shared.plots import (
    plot_receiving_operator_characteristic,
    plot_precision_recall,
    plot_calibration,
    plot_decision_curve_analysis,
    plot_optimal_confusion_matrix
)
from scripts.digital_twins.predictions.trd_prediction_computation import compute_metrics
from scripts.digital_twins.predictions.classical_ml import load_data_set

def plot_ablation_deltas(rows: list[dict], baseline_results_dir: Path):
    """Write one bar-chart PNG per delta metric into baseline_results_dir, x-axis = ablation specs, hue = classifiers.

    Args:
        rows (list[dict]): Deltas over all the metrics
        baseline_results_dir (Path): Specify where to build ablation delta results files from
    """
    spec_order = [abl["id"] for abl in ABLATIONS]
    classifier_order = ["logistic_regression", "random_forest", "gradient_boosting", "xgboost"]
    metrics_to_plot = ["roc_score", "auprc", "brier_score", "weighted_calibration_error", "calibration_slope", "calibration_intercept"]
    row_lookup = {
        (row["spec_id"], row["classifier"]): row
        for row in rows
    }
    n_classifiers = len(classifier_order)
    bar_width = 0.8 / n_classifiers # Leave 20% as gutter so all bars together take up 80% of total room
    x_base = np.arange(len(spec_order))
    for metric in metrics_to_plot:
        fig, ax = plt.subplots(figsize=(12,6))
        for i, classifier in enumerate(classifier_order):
            # X positions of the bars for this classifier's errors
            x_positions = x_base + bar_width * (i - (n_classifiers - 1) / 2)
            heights = [row_lookup[(spec_id, classifier)][f"delta_{metric}"] for spec_id in spec_order]
            if metric == "roc_score":
                # Error bands
                yerr_lower = [row_lookup[(spec_id, classifier)]["roc_score"] - row_lookup[(spec_id, classifier)]["roc_score_ci_low"] for spec_id in spec_order]
                yerr_upper = [row_lookup[(spec_id, classifier)]["roc_score_ci_high"] - row_lookup[(spec_id, classifier)]["roc_score"] for spec_id in spec_order]
                yerr = np.array([yerr_lower, yerr_upper])
            else:
                yerr = None
            ax.bar(
                x_positions,
                heights,
                width=bar_width,
                label=classifier,
                yerr=yerr,
            )
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--") # y-coordinate of horizontal line is 0
        ax.set_xticks(x_base)
        ax.set_xticklabels(spec_order, rotation=15, ha="right")
        ax.set_xlabel("Ablation Spec")
        ax.set_ylabel(f"Delta {metric}")
        ax.set_title(f"Ablation Delta — {metric}")
        ax.legend(title="Classifier") # The classifier is what the legend labels
        fig.tight_layout()
        fig.savefig(baseline_results_dir / f"ablation_delta_{metric}.png")
        plt.close(fig)

def emit_ablation_summary(baseline_results_dir: Path):
    """Given the embedded ML results directory, write the ablation results in the same location

    Args:
        baseline_results_dir (Path): Location of ML embedding results
    """
    baseline_metrics_path = baseline_results_dir / f"classical_ml_results_{VectorSource.EMBEDDED.name}.json"
    with open(baseline_metrics_path, 'r') as f:
        baseline_metrics = json.load(f)
    spec_metrics = {}
    for spec in ABLATIONS:
        spec_id = spec["id"]
        spec_metrics_path = baseline_results_dir / spec_id / f"classical_ml_results_{VectorSource.EMBEDDED.name}.json"
        with open(spec_metrics_path, 'r') as f:
            spec_metrics[spec_id] = json.load(f)
    rows = []
    for spec_id, classifier_metrics in spec_metrics.items():
        # For each ablation, each ML technique achieved some metrics
        for classifier_name, metrics in classifier_metrics.items():
            row = {
                "spec_id": spec_id,
                "classifier": classifier_name,
                **metrics
            }
            # Find this model's achieved metrics with no ablation
            baseline_classifier_metrics = baseline_metrics[classifier_name]
            for metric_name in metrics:
                if metric_name != "roc_score_ci_low" and metric_name != "roc_score_ci_high":
                    # Record the difference
                    row[f"delta_{metric_name}"] = metrics[metric_name] - baseline_classifier_metrics[metric_name] 
            rows.append(row)
    # Write the .csv
    summary_csv_path = baseline_results_dir / "ablation_summary.csv"
    fieldnames = rows[0].keys()
    with open(summary_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    plot_ablation_deltas(rows, baseline_results_dir)

def main():
    baseline_narratives_dir = Path(os.environ['NARRATIVES_DIR'])
    baseline_embeddings_dir = Path(os.environ['EMBEDDINGS_DIR'])
    baseline_results_dir = Path(os.environ['RESULTS_DIR'])
    (_, test_ids) = create_train_test_split()
    
    models = ["logistic_regression", "random_forest", "gradient_boosting", "xgboost"]
    baseline_searchers = {
        model: joblib.load(baseline_results_dir / "trained_models" / f"{model}_EMBEDDED.joblib")
        for model in models
    }
    for spec in ABLATIONS:
        spec_id = spec["id"]
        ablation_narrative_dir = baseline_narratives_dir / spec_id
        os.environ['NARRATIVES_DIR'] = str(ablation_narrative_dir)
        ablation_embeddings_dir = baseline_embeddings_dir / spec_id
        os.environ['EMBEDDINGS_DIR'] = str(ablation_embeddings_dir)
        os.makedirs(ablation_embeddings_dir, exist_ok=True)
        ablation_results_dir = baseline_results_dir / spec_id
        os.environ['RESULTS_DIR'] = str(ablation_results_dir)
        os.makedirs(ablation_results_dir, exist_ok=True)
        
        print(f"Ablating on {spec}...", flush=True)
        print(f"Narratives: {os.environ['NARRATIVES_DIR']}", flush=True)
        print(f"Embeddings: {os.environ['EMBEDDINGS_DIR']}", flush=True)
        print(f"Results: {os.environ['RESULTS_DIR']}", flush=True)
        
        # With all the proper .env changes made, forge the embeddings with the given ablation
        subprocess.run(
            ["python", '-m', 'scripts.digital_twins.embeddings.forge_embeddings'],
            check=True,
        )
        
        test_X, test_y = load_data_set(test_ids, source=VectorSource.EMBEDDED)
        print(f"Running EMBEDDED ML on {spec_id}...", flush=True)
        
        model_predictions = {
            model_name: searcher.predict_proba(X=test_X)[:, 1] # Second column is positive class probability, first is negative
            for model_name, searcher in baseline_searchers.items()
        }
        results = {}
        for model_name, predictions in model_predictions.items():
            metrics = compute_metrics(y_true=test_y, y_prob=predictions)
            _, roc_score_ci_low, roc_score_ci_high = plot_receiving_operator_characteristic(y_true=test_y, y_prob=predictions, mode=f"{model_name}_{VectorSource.EMBEDDED.name}")
            plot_precision_recall(y_true=test_y, y_prob=predictions, mode=f"{model_name}_{VectorSource.EMBEDDED.name}")
            plot_calibration(y_true=test_y, y_prob=predictions, mode=f"{model_name}_{VectorSource.EMBEDDED.name}")
            plot_decision_curve_analysis(y_true=test_y, y_prob=predictions, mode=f"{model_name}_{VectorSource.EMBEDDED.name}")
            plot_optimal_confusion_matrix(y_true=test_y, y_prob=predictions, mode=f"{model_name}_{VectorSource.EMBEDDED.name}")
        
            metrics['roc_score_ci_low'] = float(roc_score_ci_low)
            metrics['roc_score_ci_high'] = float(roc_score_ci_high)
            results[model_name.lower()] = metrics
        
        results_json_file = Path(os.environ['RESULTS_DIR']) / f'classical_ml_results_{VectorSource.EMBEDDED.name}.json'
        with open(results_json_file, 'w') as f:
            json.dump(results, f, indent=4)
    
    emit_ablation_summary(baseline_results_dir)

if __name__=="__main__":
    main()