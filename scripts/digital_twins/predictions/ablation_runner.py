import os
from pathlib import Path
import json
import subprocess
import csv

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
from scripts.digital_twins.predictions.classical_ml import (
    load_data_set,
    evaluate_models
)

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

def main():
    baseline_narratives_dir = Path(os.environ['NARRATIVES_DIR'])
    baseline_embeddings_dir = Path(os.environ['EMBEDDINGS_DIR'])
    baseline_results_dir = Path(os.environ['RESULTS_DIR'])
    (train_ids, test_ids) = create_train_test_split()
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
        
        train_X, train_y = load_data_set(train_ids, source=VectorSource.EMBEDDED)
        test_X, test_y = load_data_set(test_ids, source=VectorSource.EMBEDDED)
        print(f"Running EMBEDDED ML on {spec_id}...", flush=True)
        (model_predictions, grid_search_results) = evaluate_models(train_X, train_y, test_X, VectorSource.EMBEDDED)
        with open(Path(os.environ['RESULTS_DIR']) / 'grid_search_ml_results_EMBEDDED.json', 'w') as f:
            json.dump(grid_search_results, f, indent=4)
       
        results = {}
        for model_name, predictions in model_predictions.items():
            metrics = compute_metrics(y_true=test_y, y_prob=predictions)
            _, roc_score_ci_low, roc_score_ci_high = plot_receiving_operator_characteristic(y_true=test_y, y_prob=predictions, mode=f"{model_name}_{VectorSource.EMBEDDED.name}")
            plot_precision_recall(y_true=test_y, y_prob=predictions, mode=f"{model_name}_{VectorSource.EMBEDDED.name}")
            plot_calibration(y_true=test_y, y_prob=predictions, mode=f"{model_name}_{VectorSource.EMBEDDED.name}")
            plot_decision_curve_analysis(y_true=test_y, y_prob=predictions, mode=f"{model_name}_{VectorSource.EMBEDDED.name}")
            plot_optimal_confusion_matrix(y_true=test_y, y_prob=predictions, mode=f"{model_name}_{VectorSource.EMBEDDED.name}")
        
            metrics['roc_score_ci_low'] = roc_score_ci_low
            metrics['roc_score_ci_high'] = roc_score_ci_high
            results[model_name.lower()] = metrics
        
        results_json_file = Path(os.environ['RESULTS_DIR']) / f'classical_ml_results_{VectorSource.EMBEDDED.name}.json'
        with open(results_json_file, 'w') as f:
            json.dump(results, f, indent=4)
    
    emit_ablation_summary(baseline_results_dir)

if __name__=="__main__":
    main()