import numpy as np
from typing import Tuple
from pathlib import Path
import os
import json
import time
import joblib
import sqlite3
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer, make_column_selector
from xgboost import XGBClassifier

from scripts.pipeline.predictions.create_train_test_split import create_train_test_split
from scripts.shared.plots import (
    plot_receiving_operator_characteristic,
    plot_precision_recall,
    plot_calibration,
    plot_decision_curve_analysis,
    plot_optimal_confusion_matrix
)
from scripts.shared.utils import (
    VectorSource,
    cast_to_int8,
    load_feature_matrix,
    load_trd_set
)
from scripts.pipeline.predictions.trd_prediction_computation import compute_metrics

def load_data_set(patient_ids: set[str], source: VectorSource=VectorSource.FEATURE) -> Tuple[pd.DataFrame, np.ndarray]:
    """Load all the patient vectors and find their labels

    Args:
        patient_ids (set[str]): Set of all patient IDs whose information is to be loaded
        source (VectorSource, optional): Specifier for feature vectors of embedded vectors. Defaults to VectorSource.FEATURE.

    Returns:
        Tuple[pd.DataFrame, np.ndarray]: Features paired with labels
    """
    if source == VectorSource.FEATURE:
        X = load_feature_matrix(patient_ids)
    else:
        embeddings_db_path = Path(os.environ['EMBEDDINGS_DIR']) / 'embeddings.db'
        connection = sqlite3.connect(embeddings_db_path)
        cursor = connection.cursor()
        placeholders = ",".join(["?"] * len(patient_ids))
        cursor.execute(
f"SELECT embedding FROM embeddings WHERE patient_id IN ({placeholders}) ORDER BY patient_id",
            sorted(list(patient_ids)) # Ensures the order is preserved
        )
        X = []
        for row in cursor.fetchall():
            X.append(np.frombuffer(row[0], dtype=np.float32))
        X = pd.DataFrame(np.array(X))
        connection.close()
    trd_ids = load_trd_set()
    y = np.array([1 if id in trd_ids else 0 for id in sorted(list(patient_ids))])
    print(f"Shape of X from source {source.name}: {X.shape}; Shape of y: {y.shape}", flush=True)
    return (X, y)

def make_classifier(model, impute_numeric: bool = False) -> Pipeline:
    """Return a Pipeline tailored with the given model

    Args:
        model (SKLEARN model): Underlying model
        impute_numeric (bool, optional): Prepend a median imputer to the numeric branch.
            Needed only when the matrix carries a numeric column with missing values --
            in practice the vital signs, which load_feature_matrix(include_vitals=True)
            keeps. The imputer is inside the Pipeline so its median is learnt from the
            training rows of each grid-search fold rather than from the whole cohort.
            Defaults to False, which reproduces the published pipeline exactly; neither
            representation as published carries a numeric NaN, so the flag changes nothing
            for them either way.

    Returns:
        Pipeline: Resulting data pre-processing and machine learning pipeline
    """
    # The default hands the ColumnTransformer a bare StandardScaler, exactly as the
    # published pipeline did, so nothing about the published fits changes structurally.
    numeric_transformer = StandardScaler()
    if impute_numeric:
        numeric_transformer = Pipeline(steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])
    return Pipeline(steps=[
                    ("preprocess",
                        ColumnTransformer(
                            transformers=[
                                ("num",
                                    numeric_transformer,
                                    make_column_selector(dtype_include="number")
                                ),
                                ("cat",
                                    # Collapse binary into single value, and unseen values become the all zero encoding though that should never happen
                                    OneHotEncoder(drop='if_binary', handle_unknown='ignore'),
                                    make_column_selector(dtype_include="category")
                                ),
                                ("bool",
                                    FunctionTransformer(func=cast_to_int8, feature_names_out="one-to-one"),
                                    make_column_selector(dtype_include="bool")
                                )
                            ],
                        )
                    ),
                    ("model", model)
                ])

HYPERPARAMETERS = {
    'logistic_regression': [
        {
            'model__penalty': ['l2'],
            'model__solver': ['lbfgs', 'liblinear', 'newton-cg'],
            'model__C': [0.001, 0.01, 0.1, 1, 10, 100]
        },
        {
            'model__penalty': ['l1'],
            'model__solver': ['liblinear'],
            'model__C': [0.001, 0.01, 0.1, 1, 10, 100]
        },
        {
            'model__penalty': ['elasticnet'],
            'model__solver': ['saga'],
            'model__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'model__l1_ratio': [0.25, 0.5, 0.75],
            'model__max_iter': [5000]
        },
        {
            'model__penalty': [None],
            'model__solver': ['lbfgs', 'newton-cg']
        }
    ],
    'random_forest': {
        'model__n_estimators': [200],
        'model__max_depth': [10, 50, 100],
        'model__min_samples_split': [2, 10],
        'model__min_samples_leaf': [2, 10]
    },
    'gradient_boosting': {
        'model__learning_rate': [0.01, 0.1, 0.3],
        'model__n_estimators': [300],
        'model__max_depth': [3, 5, 8],
        'model__min_samples_split': [2, 10]
    },
    'xgboost': {
        'model__n_estimators': [300],
        'model__learning_rate': [0.01, 0.1, 0.3],
        'model__max_depth': [3, 5, 8],
        'model__subsample': [0.5, 1.0],
    }
}

def model_cache_path(model_name: str, source: VectorSource) -> Path:
    """Determine model save path given its name and the vector source it was trained on

    Args:
        model_name (str): Name of model (e.g. 'logistic_regression')
        source (VectorSource): EMBEDDED or FEATURE

    Returns:
        Path: Resulting save path for model
    """
    save_path = Path(os.environ['RESULTS_DIR']) / "trained_models" / f"{model_name}_{source.name}.joblib"
    os.makedirs(save_path.parent, exist_ok=True)
    return save_path

def evaluate_models(X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame, source: VectorSource) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    """Obtain classification results from various ML models on the input data

    Args:
        X_train (pd.DataFrame): Train observations
        y_train (np.ndarray): Train labels
        X_test (pd.DataFrame): Test observations
        source (VectorSource): EMBEDDED or FEATURE

    Returns:
        tuple[dict[str, np.ndarray], dict[str, dict]]: Probability scores for each model as well as grid search results
    """
    classifiers = {
        "logistic_regression": make_classifier(LogisticRegression(max_iter=1000, random_state=int(os.environ['SEED']))),
        "random_forest": make_classifier(RandomForestClassifier(random_state=int(os.environ['SEED']))),
        "gradient_boosting": make_classifier(GradientBoostingClassifier(random_state=int(os.environ['SEED']))),
        "xgboost": make_classifier(XGBClassifier(random_state=int(os.environ['SEED']), eval_metric='logloss', n_jobs=1))
    }
    # Fit each classifier on the training data
    classifier_predictions = {}
    model_grid_search_results = {}
    for name, classifier in classifiers.items():
        cache_path = model_cache_path(name, source)
        if cache_path.exists() and int(os.environ['SCRUB_TRAINED_MODELS']) == 0:
            print(f"Loading {name} from cache for {source.name}...", flush=True)
            searcher = joblib.load(cache_path)
        else:
            start = time.perf_counter()
            print(f"Starting {name} classifier...", flush=True)
            param_grid = HYPERPARAMETERS[name]
            # Hyperparameter grid search to enable the model to perform as best it can
            searcher = GridSearchCV(classifier, param_grid, scoring='roc_auc', cv=5, n_jobs=16)
            searcher.fit(X=X_train, y=y_train)
            elapsed = time.perf_counter() - start
            print(f"{name} classifier finished in {elapsed:.1f} seconds running {len(searcher.cv_results_['params'])} different models...", flush=True)
            joblib.dump(searcher, cache_path)
        # Find model predictions
        predictions = searcher.predict_proba(X=X_test)[:, 1]
        classifier_predictions[name] = predictions
        model_grid_search_results[name] = {
            'Best Parameters': searcher.best_params_,
            'Best Score' : float(searcher.best_score_)
        }
    return classifier_predictions, model_grid_search_results

def write_test_predictions(test_ids: set[str], test_y: np.ndarray, model_predictions: dict[str, np.ndarray], source: VectorSource) -> Path:
    """Persist the per-patient held-out predicted probabilities for this representation.

    The metrics JSON records only summary statistics, which is enough to tabulate a
    model but not enough to compare two of them: a paired test needs both score
    vectors on the same patients in the same order. Row order here is
    sorted(test_ids), which is exactly the order load_data_set imposes on both
    VectorSources, so row i is the same patient in the EMBEDDED and FEATURE files
    and the two can be compared row-for-row without a join.

    Args:
        test_ids (set[str]): Held-out cohort IDs.
        test_y (np.ndarray): Held-out labels, shape (n_test,), aligned to sorted(test_ids).
        model_predictions (dict[str, np.ndarray]): Classifier name -> predicted TRD
            probability for each held-out patient, each of shape (n_test,).
        source (VectorSource): EMBEDDED or FEATURE.

    Returns:
        Path: Where the table was written.
    """
    ordered_ids = sorted(list(test_ids))
    predictions_df = pd.DataFrame({'patient_id': ordered_ids, 'true_label': test_y})
    for model_name, predictions in model_predictions.items():
        predictions_df[model_name.lower()] = predictions
    save_path = Path(os.environ['RESULTS_DIR']) / f'test_predictions_{source.name}.parquet'
    predictions_df.to_parquet(save_path, index=False)
    print(f"Wrote {save_path} with shape {predictions_df.shape}", flush=True)
    return save_path

def run_source_pass(source: VectorSource, train_ids: set[str], test_ids: set[str]):
    """Run a single ML pass for the given vector source: load, fit/predict, score, plot, persist.

    Args:
        source (VectorSource): EMBEDDED or FEATURE
        train_ids (set[str]): Training cohort IDs
        test_ids (set[str]): Held-out cohort IDs
    """
    (train_X, train_y) = load_data_set(train_ids, source=source)
    (test_X, test_y) = load_data_set(test_ids, source=source)
    print(f"Running ML on {source.name}...", flush=True)
    model_predictions, grid_search_results = evaluate_models(train_X, train_y, test_X, source)
    with open(Path(os.environ['RESULTS_DIR']) / f"grid_search_ml_results_{source.name}.json", 'w') as f:
        json.dump(grid_search_results, f, indent=4)
    write_test_predictions(test_ids, test_y, model_predictions, source)
    results = {}
    for model_name, predictions in model_predictions.items():
        metrics = compute_metrics(y_true=test_y, y_prob=predictions)
        _, roc_score_ci_low, roc_score_ci_high = plot_receiving_operator_characteristic(y_true=test_y, y_prob=predictions, mode=f"{model_name}_{source.name}")
        plot_precision_recall(y_true=test_y, y_prob=predictions, mode=f"{model_name}_{source.name}")
        plot_calibration(y_true=test_y, y_prob=predictions, mode=f"{model_name}_{source.name}")
        plot_decision_curve_analysis(y_true=test_y, y_prob=predictions, mode=f"{model_name}_{source.name}")
        plot_optimal_confusion_matrix(y_true=test_y, y_prob=predictions, mode=f"{model_name}_{source.name}")

        metrics['roc_score_ci_low'] = float(roc_score_ci_low)
        metrics['roc_score_ci_high'] = float(roc_score_ci_high)
        results[model_name.lower()] = metrics

    results_json_file = Path(os.environ['RESULTS_DIR']) / f'classical_ml_results_{source.name}.json'
    with open(results_json_file, 'w') as f:
        json.dump(results, f, indent=4)

def main():
    # Get training and test split
    (train_ids, test_ids) = create_train_test_split()

    run_source_pass(VectorSource.EMBEDDED, train_ids, test_ids)
    run_source_pass(VectorSource.FEATURE, train_ids, test_ids)

if __name__=="__main__":
    main()
