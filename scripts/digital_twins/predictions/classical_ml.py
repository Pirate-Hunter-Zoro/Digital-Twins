import numpy as np
from typing import Tuple
from pathlib import Path
import os
import json
import time
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

from scripts.digital_twins.predictions.trd_predictor import TRDPredictor
from scripts.digital_twins.predictions.create_train_test_split import create_train_test_split
from scripts.shared.plots import (
    plot_receiving_operator_characteristic,
    plot_precision_recall,
    plot_calibration

)
from scripts.shared.utils import VectorSource

def load_data_set(patient_ids: set[str], source: VectorSource=VectorSource.DETERMINISTIC) -> Tuple[pd.DataFrame, np.ndarray]:
    """Load all the patient vectors and find their labels

    Returns:
        Tuple[pd.DataFrame, np.ndarray]: vectors and labels
    """
    predictor = TRDPredictor() # We don't care about ID exclusion - only the ability to flag patients as TRD positive or negative
    if source == VectorSource.DETERMINISTIC:
        parquet_path = Path(os.environ['DETERMINISTIC_DATAFRAME_PATH'])
        cohort_df = pd.read_parquet(parquet_path)
        X = cohort_df.loc[sorted(list(patient_ids))]
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
    y = np.array([predictor.get_trd_status(id) for id in sorted(list(patient_ids))])
    print(f"Shape of X from source {source.name}: {X.shape}; Shape of y: {y.shape}", flush=True)
    return (X, y)

def make_classifier(model):
    return Pipeline(steps=[
                    ("preprocess", 
                        ColumnTransformer(
                            transformers=[
                                ("num", Pipeline([
                                        # Replace nan with column median
                                        ("fill", SimpleImputer(strategy="median")),
                                        # Mean 0 and unit variance
                                        ("scale", StandardScaler())
                                    ]),
                                    make_column_selector(dtype_include="number")
                                ),
                                ("cat", 
                                    # Collapse binary into single value, and unseen values become the all zero encoding though that should never happen
                                    OneHotEncoder(drop='if_binary', handle_unknown='ignore'),
                                    make_column_selector(dtype_include="category")
                                ),
                                ("bool", 
                                    FunctionTransformer(func=lambda df: df.astype(np.int8)),
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
        'model__n_estimators': [300],
        'model__learning_rate': [0.01, 0.1, 0.3],
        'model__max_depth': [3, 5, 8],
    },
    'xgboost': {
        'model__n_estimators': [300],
        'model__learning_rate': [0.01, 0.1, 0.3],
        'model__max_depth': [3, 5, 8],
        'model__subsample': [0.5, 1.0],
    }
}

def evaluate_models(X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    """Obtain classification results from various ML models on the input data

    Args:
        X_train (pd.DataFrame): Train observations
        y_train (np.ndarray): Train labels
        X_test (pd.DataFrame): Test observations

    Returns:
        tuple[dict[str, np.ndarray], dict[str, dict]]: Probability scores for each model as well as grid search results
    """
    classifiers = {
        "logistic_regression": make_classifier(LogisticRegression(max_iter=1000, random_state=int(os.environ['SEED']))),
        "random_forest": make_classifier(RandomForestClassifier(random_state=int(os.environ['SEED']))),
        "gradient_boosting": make_classifier(GradientBoostingClassifier(random_state=int(os.environ['SEED']))),
        "xgboost": make_classifier(XGBClassifier(random_state=int(os.environ['SEED']), eval_metric='logloss'))
    }
    # Fit each classifier on the training data
    classifier_predictions = {}
    model_grid_search_results = {}
    for name, classifier in classifiers.items():
        start = time.perf_counter()
        print(f"Starting {name} classifier...", flush=True)
        param_grid = HYPERPARAMETERS[name]
        # Hyperparameter grid search to enable the model to perform as best it can
        searcher = GridSearchCV(classifier, param_grid, scoring='roc_auc', cv=5, n_jobs=-1)
        searcher.fit(X=X_train, y=y_train)
        elapsed = time.perf_counter() - start
        predictions = searcher.predict_proba(X=X_test)[:, 1]
        classifier_predictions[name] = predictions
        model_grid_search_results[name] = {
            'Best Parameters': searcher.best_params_,
            'Best Score' : float(searcher.best_score_)
        }
        print(f"{name} classifier finished in {elapsed:.1f} seconds running {len(searcher.cv_results_['params'])} different models...", flush=True)
    return classifier_predictions, model_grid_search_results

def main():
    # Get training and test split
    (train_ids, test_ids) = create_train_test_split()
    
    for source in VectorSource:
        (train_X, train_y) = load_data_set(train_ids, source=source)
        (test_X, test_y) = load_data_set(test_ids, source=source)
        print(f"Running ML on {source.name}...", flush=True)
        model_predictions, grid_search_results = evaluate_models(train_X, train_y, test_X)
        with open(Path(os.environ['RESULTS_DIR']) / f"grid_search_ml_results_{source.name}.json", 'w') as f:
            json.dump(grid_search_results, f, indent=4)
        for model_name, predictions in model_predictions.items():
            plot_receiving_operator_characteristic(y_true=test_y, y_prob=predictions, mode=f"{model_name}_{source.name}")
            plot_precision_recall(y_true=test_y, y_prob=predictions, mode=f"{model_name}_{source.name}")
            plot_calibration(y_true=test_y, y_prob=predictions, mode=f"{model_name}_{source.name}")
        
if __name__=="__main__":
    main()