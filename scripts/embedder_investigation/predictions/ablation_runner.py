import os
from pathlib import Path
import json
import csv
import joblib
import numpy as np
import matplotlib.pyplot as plt

from scripts.data_loading.ablation_registry import ABLATIONS
from scripts.embedder_investigation.predictions.create_train_test_split import create_train_test_split
from scripts.shared.utils import VectorSource
from scripts.shared.plots import (
    plot_receiving_operator_characteristic,
    plot_precision_recall,
    plot_calibration,
    plot_decision_curve_analysis,
    plot_optimal_confusion_matrix,
    display_ablated_roc_deltas
)
from scripts.embedder_investigation.predictions.trd_prediction_computation import compute_metrics
from scripts.embedder_investigation.predictions.classical_ml import load_data_set

def plot_ablation_deltas(rows: list[dict], baseline_results_dir: Path):
    """Write one bar-chart PNG per delta metric into baseline_results_dir, x-axis = ablation specs, hue = classifiers.

    Args:
        rows (list[dict]): Deltas over all the metrics
        baseline_results_dir (Path): Specify where to build ablation delta results files from
    """
    spec_order = [abl["id"] for abl in ABLATIONS]
    spec_labels = [abl['display'] for abl in ABLATIONS]
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
            ax.bar(
                x_positions,
                heights,
                width=bar_width,
                label=classifier
            )
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--") # y-coordinate of horizontal line is 0
        ax.set_xticks(x_base)
        ax.set_xticklabels(spec_labels, rotation=15, ha="right")
        ax.set_xlabel(f"Ablation Spec")
        ax.set_ylabel(f"Δ {metric} (ablated − baseline)")
        ax.set_title(f"Ablation Delta — {metric}" + ("" if metric != 'roc_score' else "\n95% paired-bootstrap CI on Δ AUC"))
        ax.legend(title="Classifier") # The classifier is what the legend labels
        fig.tight_layout()
        fig.savefig(baseline_results_dir / f"ablation_delta_{metric}.png")
        plt.close(fig)

def plot_ablation_roc_ci(rows: list[dict], baseline_metrics: dict[str, dict[str, tuple[float, float]]], baseline_results_dir: Path):
    """Create a forest plot showing all of the ROC score confidence interval bands over the different ablations over the different machine learning models

    Args:
        rows (list[dict]): Confidence interval ROC scores across the different classifiers for the different ablations
        baseline_metrics (dict[str, dict[str, tuple[float]]]): Confidence interval ROC scores given no ablation across the different models
        baseline_results_dir (Path): Location to store forest plot
    """
    spec_labels = {abl["id"]: abl['display'] for abl in ABLATIONS}
    classifier_order = ["logistic_regression", "random_forest", "gradient_boosting", "xgboost"]
    row_lookup = {
        (row["spec_id"], row["classifier"]): row
        for row in rows
    }
    
    diffs = []
    for spec in ABLATIONS:
        spec_id = spec["id"]
        lr_roc_score_abl = row_lookup[(spec_id, "logistic_regression")]['roc_score']
        diff = baseline_metrics['logistic_regression']['roc_score'] - lr_roc_score_abl
        diffs.append((spec_id, diff))
    diffs.sort(key=lambda x: x[1], reverse=True) # Sort by highest difference first
    ablation_labels = ["Baseline"] + [spec_labels[diff[0]] for diff in diffs]
    
    # Now plot the confidence intervals for each classifier over the different ablations
    fig, axes = plt.subplots(nrows=1, ncols=4, sharex=True, figsize=(20,len(ABLATIONS)+1))
    for i, classifier in enumerate(classifier_order):
        ax = axes[i]
        lows, mids, highs = np.zeros(len(ABLATIONS)+1),\
            np.zeros(len(ABLATIONS)+1),\
                np.zeros(len(ABLATIONS)+1)
        lows[0], mids[0], highs[0] = baseline_metrics[classifier]["roc_score_ci_low"],\
            baseline_metrics[classifier]["roc_score"],\
                baseline_metrics[classifier]["roc_score_ci_high"]
        for j, (spec_id, _) in enumerate(diffs):
            lows[j+1] = row_lookup[(spec_id, classifier)]['roc_score_ci_low']
            mids[j+1] = row_lookup[(spec_id, classifier)]['roc_score']
            highs[j+1] = row_lookup[(spec_id, classifier)]['roc_score_ci_high']
        
        # Array of confidence intervals
        errors = np.array([mids - lows, highs - mids])
        ax.errorbar(mids, np.arange(len(ABLATIONS)+1), xerr=errors, fmt='o', capsize=5)
        ax.axvline(baseline_metrics[classifier]["roc_score"], linestyle='--', linewidth=1, alpha=0.5)
        ax.set_yticks(np.arange(len(ABLATIONS)+1))
        ax.set_yticklabels(ablation_labels if i==0 else [""]*len(ablation_labels))
        ax.invert_yaxis() # Baseline row should go at top
        ax.set_title(classifier)
        ax.set_xlabel("ROC AUC")
    
    fig.tight_layout()
    fig.savefig(baseline_results_dir / f"ablation_roc_ci_{VectorSource.EMBEDDED.name}.png")
    plt.close(fig)

def emit_ablation_summary(baseline_results_dir: Path, delta_roc_score_ci: dict[str, dict[str, tuple[float, float]]]):
    """Given the embedded ML results directory, write the ablation results in the same location

    Args:
        baseline_results_dir (Path): Location of ML embedding results
        delta_roc_score_ci (dict[str, dict[str, tuple[float, float]]]): For each model, and each ablation, stores the lower and upper bounds for 95% confidence interval of the difference between ablated and baseline roc scores
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
            row["delta_roc_score_ci_low"] = delta_roc_score_ci[classifier_name][spec_id][0]
            row["delta_roc_score_ci_high"] = delta_roc_score_ci[classifier_name][spec_id][1]
            rows.append(row)
    # Write the .csv
    summary_csv_path = baseline_results_dir / "ablation_summary.csv"
    fieldnames = rows[0].keys()
    with open(summary_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    plot_ablation_deltas(rows, baseline_results_dir)
    plot_ablation_roc_ci(rows, baseline_metrics, baseline_results_dir)

def main():
    baseline_embeddings_dir = Path(os.environ['EMBEDDINGS_DIR'])
    baseline_results_dir = Path(os.environ['RESULTS_DIR'])
    (_, test_ids) = create_train_test_split()
    
    models = ["logistic_regression", "random_forest", "gradient_boosting", "xgboost"]
    baseline_searchers = {
        model: joblib.load(baseline_results_dir / "trained_models" / f"{model}_EMBEDDED.joblib")
        for model in models
    }
    
    (test_X, test_y) = load_data_set(test_ids, VectorSource.EMBEDDED)
    baseline_predictions = {
        model_name: searcher.predict_proba(X=test_X)[:, 1] # Second column is positive class probability, first is negative
        for model_name, searcher in baseline_searchers.items()
    }
    accumulator = {
        model_name: {}
        for model_name in models
    }
    
    for spec in ABLATIONS:
        spec_id = spec["id"]
        ablation_embeddings_dir = baseline_embeddings_dir / spec_id
        os.environ['EMBEDDINGS_DIR'] = str(ablation_embeddings_dir)
        os.makedirs(ablation_embeddings_dir, exist_ok=True)
        ablation_results_dir = baseline_results_dir / spec_id
        os.environ['RESULTS_DIR'] = str(ablation_results_dir)
        os.makedirs(ablation_results_dir, exist_ok=True)
        
        test_X, _ = load_data_set(test_ids, source=VectorSource.EMBEDDED)
        print(f"Running EMBEDDED ML on {spec_id}...", flush=True)
        print(f"Embeddings: {os.environ['EMBEDDINGS_DIR']}", flush=True)
        print(f"Results: {os.environ['RESULTS_DIR']}", flush=True)
        
        model_predictions = {
            model_name: searcher.predict_proba(X=test_X)[:, 1] # Second column is positive class probability, first is negative
            for model_name, searcher in baseline_searchers.items()
        }
        for model_name in models:
            accumulator[model_name][spec_id] = model_predictions[model_name]
        
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
    
    ablated_names = {spec["id"]: spec["display"] for spec in ABLATIONS}
    delta_ci_by_classifier = {
        model: display_ablated_roc_deltas(classifier_name=model, labels=test_y, probs=baseline_predictions[model], ablated_scores=accumulator[model], ablated_names=ablated_names)
        for model in models
    }
    
    emit_ablation_summary(baseline_results_dir, delta_ci_by_classifier)

if __name__=="__main__":
    main()