import os
import json
from pathlib import Path
import copy
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from scipy.stats import spearmanr

from scripts.digital_twins.predictions.classical_ml import make_classifier
from scripts.shared.utils import VectorSource
from scripts.digital_twins.predictions.classical_ml import load_data_set, model_cache_path
from scripts.digital_twins.predictions.create_train_test_split import create_train_test_split

from dotenv import load_dotenv
load_dotenv()


PCA_K_VALUES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
MODEL_NAMES = ("logistic_regression", "random_forest", "gradient_boosting", "xgboost")
TOP_K = 20

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
    """Given the model, load the cached fitted best estimator if available, otherwise load the best hyperparameters and use them to refit a fresh pipeline from the grid-search best params

    Args:
        model_name (str): Name of model to reference results with
        source (VectorSource): Feature or embedded vectors
        X_train (pd.DataFrame): Training inputs to fit with
        y_train (np.ndarray): Training outputs to fit with

    Returns:
        Pipeline: fitted sklearn Pipeline
    """
    cache_path = model_cache_path(model_name, source)
    if cache_path.exists():
        print(f"Loading cached best estimator for {model_name} on {source.name}...", flush=True)
        return joblib.load(cache_path).best_estimator_
    print(f"Cache miss for {model_name}_{source.name}; refitting from grid-search best params...", flush=True)
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
    top_k: int=20,
    direction_signs: np.ndarray | None = None
):
    """Helper method to plot the different feature importances of the given model

    Args:
        importances (np.ndarray): Importances of each feature from the learning model
        feature_names (list[str]): Names of each feature
        model_name (str): Name of classifier - e.g. logistic_regression, etc.
        direction_signs (np.ndarray | None, optional): Whether an increase in the numeric feature increases or decreases risk score. Defaults to None.
        top_k (int, optional): How many bars (features) to display. Defaults to 20.
    """
    magnitudes = np.abs(importances)
    # Only grab the top k sorted (reverse order for higher magnitude first) indices
    sorted_mag_indices = np.argsort(magnitudes)[::-1][:top_k]
    top_importances = importances[sorted_mag_indices]
    top_names = [feature_names[i] for i in sorted_mag_indices]
    top_directions = None
    if direction_signs is not None:
        top_directions = direction_signs[sorted_mag_indices]
    
    if model_name == "logistic_regression":
        colors = ['steelblue' if val >= 0 else 'firebrick' for val in top_importances]
    elif top_directions is None:
        colors = 'steelblue' # One raw string works with matplotlib as well
    else:
        colors = ['steelblue' if val >= 0 else 'firebrick' for val in top_directions.tolist()]
    
    fig, ax = plt.subplots(figsize=(10, max(6, top_k * 0.3)))
    # Create bars of length corresponding to importance magnitudes
    ax.barh(range(len(top_names)), np.abs(top_importances), color=colors)
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names)
    ax.invert_yaxis()
    ax.set_xlabel("Feature importance (magnitude)")
    title = f"{model_name}: " + "(blue: raises TRD risk, red: lowers TRD risk)" \
                                if model_name == "logistic_regression" or (direction_signs is not None)\
                                    else "(direction unspecified)"
    ax.set_title(title)
    fig.tight_layout()
    save_path = Path(os.environ['RESULTS_DIR']) / "feature_importance" /\
        f"feature_importance_{model_name}.png"
    os.makedirs(save_path.parent, exist_ok=True)
    fig.savefig(str(save_path), dpi=120)
    plt.close(fig)
    
def compute_univariate_spearman(
    X: pd.DataFrame,
    risk_scores: np.ndarray,
    feature_names: list[str],
) -> np.ndarray:
    """Find spearman correlation between each column of X and y

    Args:
        X (pd.DataFrame): post-ColumnTransformer feature matrix (imputed, scaled, one-hot-encoded, bool-cast) (completely numeric) 
        risk_scores (np.ndarray): TRD risk scores predicted by some model, shape (n_samples,)
        feature_names (list[str]): Column names of X.columns

    Returns:
        np.ndarray: (n_features,) - each spearman correlation over all attribute indices in X
    """
    if X.shape[1] != len(feature_names):
        raise ValueError(f"Error, expected {len(feature_names)} columns in X but found {X.shape[1]}")
    # For each column, compute spearman correlation between X[col] and y
    result = spearmanr(X, risk_scores).statistic # Works with X being 2D and y being 1D
    # Output is of shape (n_features + 1, n_features + 1) - we care about the last row, first n_features
    return result[result.shape[0]-1, 0:len(feature_names)]

def count_nonzero_lr_coefficients(pipeline: Pipeline) -> tuple[int,int]:
    """Given a trained logistic regression learning model, find the number of non-zero (or close to it) coefficients

    Args:
        pipeline (Pipeline): "model" step is a LogisticRegression

    Returns:
        tuple[int,int]: (nonzero_count, total_count)
    """
    model = pipeline.named_steps["model"]
    if not isinstance(model, LogisticRegression):
        raise TypeError(f"Expected model of type {LogisticRegression.__name__} but received {type(model).__name__}...")
    feature_coefficients = model.coef_[0] # index zero since shape is (1, n_features)
    total = len(feature_coefficients)
    nonzero = len(feature_coefficients[np.abs(feature_coefficients) > 1e-10])
    return (nonzero, total)
       
def plot_cumulative_magnitude_overlay(
    magnitudes_by_label: dict[str, np.ndarray],
    xlabel: str,
    ylabel: str,
    title: str,
    filename: str,
):
    """Plot of overlapping results of cumulative fraction of magnitude against output for each dimension over all the given model (or baseline) results

    Args:
        magnitudes_by_label (dict[str, np.ndarray]): Magnitude results for various models/schemes
        xlabel (str): x-axis text
        ylabel (str): y-axis text
        title (str): plot title text
        filename (str): final segment of save path
    """
    fig, ax = plt.subplots(figsize=(10,6))
    ax.axhline(0.8, linestyle='--', color='gray', alpha=0.5)
    ax.axhline(0.9, linestyle='--', color='gray', alpha=0.5)
    
    # Plot each model's correlation output
    for label, magnitudes in magnitudes_by_label.items():
        # Sort by decreasing magnitude
        sorted_magnitudes = np.sort(np.abs(magnitudes))[::-1]
        cumulative = np.cumsum(sorted_magnitudes) # Last rank holds total mass
        # Normalize to fraction of mass
        fraction = cumulative / cumulative[-1]
        # Create plot of increasing 'rank-1' on the x-axis, farther to the left is where we have added the highest remaining magnitude correlation
        ranks = np.arange(1, len(fraction)+1)
        knee_80 = np.searchsorted(fraction, 0.8) + 1
        knee_90 = np.searchsorted(fraction, 0.9) + 1
        ax.plot(ranks, fraction, linewidth=2, label=f"{label} (K₈₀={knee_80}, K₉₀={knee_90})")
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    save_path = Path(os.environ['RESULTS_DIR']) / "feature_importance" / filename
    os.makedirs(save_path.parent, exist_ok=True)
    fig.savefig(str(save_path), dpi=120)
    plt.close(fig)
   
def pca_cache_path(model_name: str, k: int) -> Path:
    """Given the model name and the number of PCA dimensions, return resulting path where the model should be saved

    Args:
        model_name (str): Name of model (e.g. 'logistic_regression')
        k (int): Number of PCA dimensions

    Returns:
        Path: Resulting save path
    """
    save_path = Path(os.environ['RESULTS_DIR']) / "trained_models_pca" / f"{model_name}_K={k}.joblib"
    os.makedirs(save_path.parent, exist_ok=True)
    return save_path
   
def plot_pca_k_vs_roc(
    model_name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
):
    """For each K in a pre-set list of values, refit one of the four classifiers on the embedded
  vectors projected to K principal components, score it on the held-out test set, and plot ROC AUC
  versus K

    Args:
        model_name (str): Specified ML model
        X_train (pd.DataFrame): Embedded vectors
        X_test (pd.DataFrame): Held-out embedded vectors
        y_train (np.ndarray): Train labels
        y_test (np.ndarray): Held-out labels
    """
    base_models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=int(os.environ['SEED'])),
        "random_forest": RandomForestClassifier(random_state=int(os.environ['SEED'])),
        "gradient_boosting": GradientBoostingClassifier(random_state=int(os.environ['SEED'])),
        "xgboost": XGBClassifier(random_state=int(os.environ['SEED']), eval_metric='logloss')
    }
    best_params = load_best_params(model_name, VectorSource.EMBEDDED)
    auc_scores = []
    for k in PCA_K_VALUES:
        # Different number of PCA dimensions each time
        pca_save_path = pca_cache_path(model_name, k)
        if pca_save_path.exists() and int(os.environ['SCRUB_TRAINED_MODELS']) == 0:
            print(f"Loading cached PCA-K{k} pipeline for {model_name}...", flush=True)
            pipeline = joblib.load(pca_save_path)
        else:
            pipeline = Pipeline(steps=\
                [
                    ("scale", StandardScaler()),
                    ("pca", PCA(n_components=k, random_state=int(os.environ['SEED']))),
                    ("model", base_models[model_name])
                ]
            )
            pipeline.set_params(**best_params)
            pipeline.fit(X_train, y_train)
            joblib.dump(pipeline, pca_save_path)
        y_pred = pipeline.predict_proba(X_test)[:,1]
        score = float(roc_auc_score(y_true=y_test, y_score=y_pred))
        auc_scores.append(score)
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(PCA_K_VALUES, auc_scores, marker='o', color='steelblue', linewidth=2)
    ax.set_xscale('log', base=2) # Logarithmic x-scale since k-values are powers of 2
    for k, auc in zip(PCA_K_VALUES, auc_scores):
        ax.text(k, auc, f"{auc:.3f}")
    ax.set_xlabel("Truncated PCA components (K)")
    ax.set_ylabel("Held-out ROC AUC")
    ax.set_title(f"PCA-K vs ROC AUC - {model_name} (EMBEDDED)")
    fig.tight_layout()
    plot_save_path = Path(os.environ['RESULTS_DIR']) / 'feature_importance' / f"feature_importance_pca_sweep_{model_name}_EMBEDDED.png"
    os.makedirs(plot_save_path.parent, exist_ok=True)
    fig.savefig(str(plot_save_path), dpi=120)
    plt.close(fig)

def write_feature_importance_summary(
    summary: dict[str, list[dict]]
):
    """Record name, feature importance, and feature importance direction of each feature over all models

    Args:
        summary (dict[str, list[dict]]): Results
    """
    save_path = Path(os.environ['RESULTS_DIR']) / "feature_importance" / "feature_importance_summary.json"
    os.makedirs(save_path.parent, exist_ok=True)
    cleaned_summary = copy.deepcopy(summary)
    for model in summary.keys():
        # JSON chokes on np.float64 and np.int64
        for row in cleaned_summary[model]:
            row["importance"] = float(row["importance"])
            row["sign"] = int(row["sign"])
    with open(save_path, 'w') as f:
        json.dump(cleaned_summary, f, indent=4)
        
def main():
    (train_ids, test_ids) = create_train_test_split()
    summary = {}
    for source in VectorSource:
        print(f"Feature importance pass: {source.name} running...", flush=True)
        (X_train, y_train) = load_data_set(train_ids, source)
        (X_test, y_test) = load_data_set(test_ids, source)
        
        if source == VectorSource.EMBEDDED:
            # Grab the feature labels, which are essentially embedding dimension indices
            correlations_by_label: dict[str, np.ndarray] = {}
            importances_by_label: dict[str, np.ndarray] = {}
            feature_name_dims = [str(col) for col in X_test.columns]
            label_correlations = compute_univariate_spearman(X_test, y_test, feature_name_dims)
            correlations_by_label['TRD label (model-agnostic)'] = label_correlations
            
        for model_name in MODEL_NAMES:
            print(f"Running {model_name} feature importance under {source.name} vectors...")
            model_pipeline = refit_best_model(model_name, source, X_train, y_train)
            if source == VectorSource.FEATURE:
                (importances, feature_names) = extract_feature_importances(model_pipeline, model_name)
                risk_scores = model_pipeline.predict_proba(X_test)[:, 1] # Extract second column
                # Preprocess X_test so that categoricals are one-hot encoded and categoricals, bools are int8, etc. (numeric vector)
                X_test_preprocessed = model_pipeline.named_steps['preprocess'].transform(X_test)
                if hasattr(X_test_preprocessed, 'toarray'):
                    X_test_preprocessed = X_test_preprocessed.toarray()
                correlations = compute_univariate_spearman(X_test_preprocessed, risk_scores, feature_names)
                direction_signs = np.sign(correlations)
                plot_feature_importance(importances, feature_names, model_name, direction_signs=direction_signs)
                # Grab top k most important features
                sorted_indices = np.argsort(np.abs(importances))[::-1][:TOP_K]
                classifier_top_rows = [{
                    "name": feature_names[i],
                    "importance": importances[i],
                    "sign": direction_signs[i]
                } for i in sorted_indices.tolist()]
                summary[model_name] = classifier_top_rows
            else:
                # EMBEDDED VectorSource
                (feature_importances, _) = extract_feature_importances(model_pipeline, model_name)
                importances_by_label[model_name] = feature_importances
                
                if model_name == "logistic_regression":
                    (nonzero_count, total_count) = count_nonzero_lr_coefficients(model_pipeline)
                    sparsity_path = Path(os.environ['RESULTS_DIR']) / "feature_importance_sparsity.json"
                    os.makedirs(sparsity_path.parent, exist_ok=True)
                    with open(sparsity_path, 'w') as f:
                        json.dump({
                            "nonzero_coefficients": nonzero_count,
                            "total_coefficients": total_count
                        }, f, indent=4)
                    
                risk_scores = model_pipeline.predict_proba(X_test)[:, 1]
                feature_names = [str(col) for col in X_test.columns]
                correlations = compute_univariate_spearman(X_test, risk_scores, feature_names)
                correlations_by_label[model_name] = correlations
                
                plot_pca_k_vs_roc(model_name, X_train, X_test, y_train, y_test)
    
        if source == VectorSource.EMBEDDED:
            # Plot feature correlations
            plot_cumulative_magnitude_overlay(
                magnitudes_by_label = correlations_by_label,
                xlabel = "Embedding dimension rank (by |Spearman ρ|, descending)",
                ylabel = f"Cumulative |Spearman ρ| fraction",
                title = f"Cumulative correlation curve ({VectorSource.EMBEDDED.name})",
                filename = f"feature_correlation_cumulative_{VectorSource.EMBEDDED.name}.png",
            )
            # Plot feature importances
            plot_cumulative_magnitude_overlay(
                magnitudes_by_label = importances_by_label,
                xlabel = "Embedding dimension rank (by importance, descending)",
                ylabel = f"Cumulative importance fraction",
                title = f"Cumulative importance curve ({VectorSource.EMBEDDED.name})",
                filename = f"feature_importance_cumulative_{VectorSource.EMBEDDED.name}.png",
            )
    
    write_feature_importance_summary(summary)
    
if __name__=="__main__":
    main()