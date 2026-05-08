import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
        source (VectorSource): Embedded or feature

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
        source (VectorSource): Feature or embedded vectors
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
    steps = pipeline.named_steps
    
    # This preprocessor has already been fitted with its learned categories from training
    preprocessor = steps['preprocess']
    # Post encoding labels - e.g. branch__column_value style for categorical fields
    feature_names = list(preprocessor.get_feature_names_out())
    
    # Grab the model's importances
    model = steps['model'] # Already fitted model
    if model_name == "logistic_regression":
        # The coefficients logistic regression applies to each feature ARE the importances
        importances = model.coef_[0] # Grab index zero because originally of shape (1, n_features)
    else:
        importances = model.feature_importances_
        
    return (importances, feature_names)

def plot_feature_importance(
    importances: np.ndarray,
    feature_names: list[str],
    model_name: str,
    top_k: int=20
):
    """Helper method to plot the different feature importances of the given model

    Args:
        importances (np.ndarray): Importances of each feature from the learning model
        feature_names (list[str]): Names of each feature
        model_name (str): Name of classifier - e.g. logistic_regression, etc.
        top_k (int, optional): How many bars (features) to display. Defaults to 20.
    """
    magnitudes = np.abs(importances)
    # Only grab the top k sorted (reverse order for higher magnitude first) indices
    sorted_mag_indices = np.argsort(magnitudes)[::-1][:top_k]
    top_importances = importances[sorted_mag_indices]
    top_names = [feature_names[i] for i in sorted_mag_indices]
    
    if model_name == "logistic_regression":
        colors = ['steelblue' if val >= 0 else 'firebrick' for val in top_importances]
    else:
        colors = 'steelblue' # One raw string works with matplotlib as well
    
    fig, ax = plt.subplots(figsize=(10, max(6, top_k * 0.3)))
    # Create bars of length corresponding to importance magnitudes
    ax.barh(range(len(top_names)), np.abs(top_importances), color=colors)
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names)
    ax.invert_yaxis()
    ax.set_xlabel("Feature importance (magnitude)")
    title = f"{model_name}: {" (blue: raises TRD risk, red: lowers TRD risk)" \
                                if model_name == "logistic_regression"\
                                    else "(direction unavailable for tree models)"}"
    ax.set_title(title)
    fig.tight_layout()
    save_path = Path(os.environ['RESULTS_DIR']) / "feature_importance" /\
        f"feature_importance_{model_name}.png"
    os.makedirs(save_path.parent, exist_ok=True)
    fig.savefig(str(save_path), dpi=120)
    plt.close(fig)