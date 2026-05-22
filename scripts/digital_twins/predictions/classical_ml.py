import numpy as np
from typing import Dict, Tuple, Optional
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
    HistGradientBoostingClassifier
)
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer, make_column_selector
from xgboost import XGBClassifier

from scripts.digital_twins.predictions.trd_predictor import TRDPredictor
from scripts.digital_twins.predictions.create_train_test_split import create_train_test_split
from scripts.shared.plots import (
    plot_receiving_operator_characteristic,
    plot_precision_recall,
    plot_calibration,
    plot_decision_curve_analysis,
    plot_optimal_confusion_matrix
)
from scripts.shared.utils import (
    VectorSource,
    VitalsStrategy,
    ClassifierFamily,
    VITAL_COLUMNS, 
    cast_to_int8
)
from scripts.digital_twins.predictions.trd_prediction_computation import compute_metrics

def load_data_set(patient_ids: set[str], source: VectorSource=VectorSource.FEATURE, vitals_strategy: Optional[VitalsStrategy]=None, classifier_family: Optional[ClassifierFamily]=None) -> Tuple[pd.DataFrame, np.ndarray]:
    """Load all the patient vectors and find their labels

    Args:
        patient_ids (set[str]): Set of all patient IDs whose information is to be loaded
        source (VectorSource, optional): Specifier for feature vectors of embedded vectors. Defaults to VectorSource.FEATURE.
        vitals_strategy (Optional[VitalsStrategy], optional): Specifier for how to handle missing vitals. Defaults to None. Not applicable when using embedded vectors
        classifier_family (Optional[ClassiferFamily], optional): Specifier for what type of classifier the data excepts to be working with - only matters if vitals_strategy == VitalsStreategy.ASYMMETRIC

    Returns:
        Tuple[pd.DataFrame, np.ndarray]: Features paired with labels
    """
    predictor = TRDPredictor() # We don't care about ID exclusion - only the ability to flag patients as TRD positive or negative
    if source == VectorSource.FEATURE:
        parquet_path = Path(os.environ['FEATURE_DATAFRAME_PATH'])
        cohort_df = pd.read_parquet(parquet_path)
        obj_cols = cohort_df.select_dtypes(include='object').columns
        cohort_df[obj_cols] = cohort_df[obj_cols].astype('category')
        X = cohort_df.loc[sorted(list(patient_ids))]
        if (vitals_strategy == VitalsStrategy.DROP) or (vitals_strategy == VitalsStrategy.ASYMMETRIC and classifier_family == ClassifierFamily.LINEAR):
            X = X.drop(columns=list(VITAL_COLUMNS))
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

def make_numeric_transformer(vitals_strategy: Optional[VitalsStrategy], classifier_family: Optional[ClassifierFamily]):
    """Create a data transformer based on how vitals are handled and based on the type of classifier

    Args:
        vitals_strategy (Optional[VitalsStrategy]): How vitals are handled
        classifier_family (Optional[ClassifierFamily]): Tree based or logistic regression
    """
    if classifier_family == ClassifierFamily.TREE and vitals_strategy in {VitalsStrategy.INDICATOR, VitalsStrategy.ASYMMETRIC}:
        return "passthrough" # No data-pre-processing pipeline construction necessary
    if classifier_family == ClassifierFamily.LINEAR and vitals_strategy == VitalsStrategy.INDICATOR:
        return Pipeline(
            steps=[
                ("fill", SimpleImputer(strategy="median", add_indicator=True)), # Missing indicator
                ("scale", StandardScaler())
            ]
        )
    # Otherwise, replace with median and no indicator
    return Pipeline([
        # Replace nan with column median
        ("fill", SimpleImputer(strategy="median")),
        # Mean 0 and unit variance
        ("scale", StandardScaler())
    ])

def make_classifier(model, vitals_strategy: Optional[VitalsStrategy], classifier_family: Optional[ClassifierFamily]) -> Pipeline:
    """Return a Pipeline tailored with the given model, strategy on handling vitals, and classifier family

    Args:
        model (SKLEARN model): Underlying model
        vitals_strategy (Optional[VitalsStrategy]): How to handle vitals
        classifier_family (Optional[ClassifierFamily]): Tree based or linear

    Returns:
        Pipeline: Resulting data pre-processing and machine learning pipeline
    """
    return Pipeline(steps=[
                    ("preprocess", 
                        ColumnTransformer(
                            transformers=[
                                ("num", 
                                    make_numeric_transformer(vitals_strategy, classifier_family),
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

CLASSIFIER_FAMILY: Dict[str, ClassifierFamily] = {
    "logistic_regression": ClassifierFamily.LINEAR,
    "random_forest": ClassifierFamily.TREE,
    "gradient_boosting": ClassifierFamily.TREE,
    "xgboost": ClassifierFamily.TREE,
}

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
        'model__max_iter': [300],
        'model__max_leaf_nodes': [15, 31, 63],
        'model__l2_regularization': [0.0, 1.0]
    },
    'xgboost': {
        'model__n_estimators': [300],
        'model__learning_rate': [0.01, 0.1, 0.3],
        'model__max_depth': [3, 5, 8],
        'model__subsample': [0.5, 1.0],
    }
}

def model_cache_path(model_name: str, source: VectorSource, vitals_strategy: Optional[VitalsStrategy]=None) -> Path:
    """Determine model save path given its name and the vector source it was trained on

    Args:
        model_name (str): Name of model (e.g. 'logistic_regression')
        source (VectorSource): EMBEDDED or FEATURE
        vitals_strategy (Optional[VitalsStrategy]): Dictates how vitals are handled, defaults to None and ignored when source is EMBEDDED

    Returns:
        Path: Resulting save path for model
    """
    strategy_suffix = f"_{vitals_strategy.name}" if vitals_strategy is not None and source==VectorSource.FEATURE else ""
    save_path = Path(os.environ['RESULTS_DIR']) / "trained_models" / f"{model_name}_{source.name}{strategy_suffix}.joblib"
    os.makedirs(save_path.parent, exist_ok=True)
    return save_path

def evaluate_models(X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame, source: VectorSource, vitals_strategy: Optional[VitalsStrategy]=None, only_classifier_family: Optional[ClassifierFamily]=None) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    """Obtain classification results from various ML models on the input data

    Args:
        X_train (pd.DataFrame): Train observations
        y_train (np.ndarray): Train labels
        X_test (pd.DataFrame): Test observations
        source (VectorSource): EMBEDDED or FEATURE
        vitals_strategy (Optional[VitalsStrategy]): Forwarded to make_classifier to specify how vitals are handled - defaults to None and ignored with EMBEDDED vector source
        only_classifier_family (Optional[ClassifierFamily]): When set, only classifiers that match this family are fitted and returned - defaults to None

    Returns:
        tuple[dict[str, np.ndarray], dict[str, dict]]: Probability scores for each model as well as grid search results
    """
    classifiers = {
        "logistic_regression": make_classifier(LogisticRegression(max_iter=1000, random_state=int(os.environ['SEED'])), vitals_strategy, CLASSIFIER_FAMILY['logistic_regression']),
        "random_forest": make_classifier(RandomForestClassifier(random_state=int(os.environ['SEED'])), vitals_strategy, CLASSIFIER_FAMILY['random_forest']),
        "gradient_boosting": make_classifier(HistGradientBoostingClassifier(random_state=int(os.environ['SEED'])), vitals_strategy, CLASSIFIER_FAMILY['gradient_boosting']),
        "xgboost": make_classifier(XGBClassifier(random_state=int(os.environ['SEED']), eval_metric='logloss'), vitals_strategy, CLASSIFIER_FAMILY['xgboost'])
    }
    # Fit each classifier on the training data
    classifier_predictions = {}
    model_grid_search_results = {}
    for name, classifier in classifiers.items():
        if only_classifier_family is not None and CLASSIFIER_FAMILY[name] != only_classifier_family:
            # Not a model we care about fitting in this isntance
            continue
        cache_path = model_cache_path(name, source, vitals_strategy)
        if cache_path.exists() and int(os.environ['SCRUB_TRAINED_MODELS']) == 0:
            print(f"Loading {name} from cache for {source.name}...", flush=True)
            searcher = joblib.load(cache_path)
        else:
            start = time.perf_counter()
            print(f"Starting {name} classifier...", flush=True)
            param_grid = HYPERPARAMETERS[name]
            # Hyperparameter grid search to enable the model to perform as best it can
            searcher = GridSearchCV(classifier, param_grid, scoring='roc_auc', cv=5, n_jobs=-1)
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

def main():
    # Get training and test split
    (train_ids, test_ids) = create_train_test_split()
     
    # Run ML on embedded vectors
    source = VectorSource.EMBEDDED
    (train_X, train_y) = load_data_set(train_ids, source=source)
    (test_X, test_y) = load_data_set(test_ids, source=source)
    print(f"Running ML on {source.name}...", flush=True)
    model_predictions, grid_search_results = evaluate_models(train_X, train_y, test_X, source)
    with open(Path(os.environ['RESULTS_DIR']) / f"grid_search_ml_results_{source.name}.json", 'w') as f:
        json.dump(grid_search_results, f, indent=4)
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
    
    # Run ML on FEATURE vectors
    source = VectorSource.FEATURE
    for strategy in VitalsStrategy:
        strategy_suffix = f"_{strategy.name}"
        print(f"Running ML on {source.name} (vitals strategy: {strategy.name})...", flush=True)
        
        if strategy == VitalsStrategy.ASYMMETRIC:
            # LR and Tree-based treated differently
            (train_X_lr, train_y) = load_data_set(train_ids, source=source, vitals_strategy=strategy, classifier_family=ClassifierFamily.LINEAR)
            (test_X_lr, test_y) = load_data_set(test_ids, source=source, vitals_strategy=strategy, classifier_family=ClassifierFamily.LINEAR)
            (train_X_tree, _) = load_data_set(train_ids, source=source, vitals_strategy=strategy, classifier_family=ClassifierFamily.TREE)
            (test_X_tree, _) = load_data_set(test_ids, source=source, vitals_strategy=strategy, classifier_family=ClassifierFamily.TREE)
            
            # Fit LR on the LR set
            (lr_predictions, lr_grid_results) = evaluate_models(train_X_lr, train_y, test_X_lr, source, strategy, only_classifier_family=ClassifierFamily.LINEAR)
            (tree_predictions, tree_grid_results) = evaluate_models(train_X_tree, train_y, test_X_tree, source, strategy, only_classifier_family=ClassifierFamily.TREE)
            model_predictions, grid_search_results = {**lr_predictions, **tree_predictions}, {**lr_grid_results, **tree_grid_results}
        
        else:
            (train_X, train_y) = load_data_set(train_ids, source=source, vitals_strategy=strategy)
            (test_X, test_y) = load_data_set(test_ids, source=source, vitals_strategy=strategy)
            model_predictions, grid_search_results = evaluate_models(train_X, train_y, test_X, source, strategy)
        
        with open(Path(os.environ['RESULTS_DIR']) / f"grid_search_ml_results_{source.name}{strategy_suffix}.json", 'w') as f:
            json.dump(grid_search_results, f, indent=4)
        results = {}
        for model_name, predictions in model_predictions.items():
            metrics = compute_metrics(y_true=test_y, y_prob=predictions)
            _, roc_score_ci_low, roc_score_ci_high = plot_receiving_operator_characteristic(y_true=test_y, y_prob=predictions, mode=f"{model_name}_{source.name}{strategy_suffix}")
            plot_precision_recall(y_true=test_y, y_prob=predictions, mode=f"{model_name}_{source.name}{strategy_suffix}")
            plot_calibration(y_true=test_y, y_prob=predictions, mode=f"{model_name}_{source.name}{strategy_suffix}")
            plot_decision_curve_analysis(y_true=test_y, y_prob=predictions, mode=f"{model_name}_{source.name}{strategy_suffix}")
            plot_optimal_confusion_matrix(y_true=test_y, y_prob=predictions, mode=f"{model_name}_{source.name}{strategy_suffix}")
        
            metrics['roc_score_ci_low'] = float(roc_score_ci_low)
            metrics['roc_score_ci_high'] = float(roc_score_ci_high)
            results[model_name.lower()] = metrics
        
        results_json_file = Path(os.environ['RESULTS_DIR']) / f'classical_ml_results_{source.name}{strategy_suffix}.json'
        with open(results_json_file, 'w') as f:
            json.dump(results, f, indent=4)
        
if __name__=="__main__":
    main()