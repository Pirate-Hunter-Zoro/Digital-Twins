import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

from scripts.digital_twins.predictions.classical_ml import make_classifier
from scripts.shared.utils import VectorSource

from dotenv import load_dotenv
load_dotenv()

def load_best_params(model_name: str, source: VectorSource) -> dict:
    """From the recorded results, find the best parameters associated with the given model operating on the input vector source

    Args:
        model_name (str): Classifier
        source (VectorSource): Embedding or deterministic

    Returns:
        dict: Pipeline-prefixed parameters
    """
    results_json_path = Path(os.environ['RESULTS_DIR']) / f"grid_search_ml_results_{source.name}.json"
    with open(results_json_path, 'r') as f:
        results = json.load(f)
    return results[model_name]["Best Parameters"]

def refit_best_model(
    model_name: str,
    source: VectorSource,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
) -> Pipeline:
    """Given the model, load the best hyperparameters and use them to refit the model in the input model

    Args:
        model_name (str): Name of model to reference results with
        source (VectorSource): Deterministic or embedded vectors
        X_train (pd.DataFrame): Training inputs to fit with
        y_train (np.ndarray): Training outputs to fit with

    Returns:
        Pipeline: fitted sklearn Pipeline
    """
    seed = int(os.environ['SEED'])
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=seed),
        "random_forest": RandomForestClassifier(random_state=seed),
        "gradient_boosting": GradientBoostingClassifier(random_state=seed),
        "xgboost": XGBClassifier(random_state=seed, eval_metric="logloss")
    }
    base_estimator_pipeline = make_classifier(models[model_name])
    hyperparams = load_best_params(model_name, source)
    base_estimator_pipeline.set_params(**hyperparams)
    base_estimator_pipeline.fit(X_train, y_train)
    return base_estimator_pipeline

def extract_feature_importances(
    pipeline: Pipeline,
    model_name: str,
) -> tuple[np.ndarray, list[str]]:
    """From fitted pipeline, extract importance value for each feature

    Args:
        pipeline (Pipeline): Fitted model
        model_name (str): Name of model

    Returns:
        tuple[np.ndarray, list[str]]: Feature importances paired with their names
    """
    pass